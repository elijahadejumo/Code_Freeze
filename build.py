"""Rebuild the dataset in data/ from GitHub and the git history (needs a GitHub token).

    git clone --filter=tree:0 --bare --single-branch --branch master \\
        https://github.com/kubernetes/kubernetes.git k8s.git

    python build.py crawl k8s.git          # PR metadata and verified label timelines -> data/raw/ (hours)
    python build.py dataset k8s.git        # data/analysis.csv.gz, data/links_reverts.csv, data/links_fixes.csv
    python build.py census-page OUT.json   # page data for hand-coding the fix links

crawl reads GITHUB_TOKEN from the environment or from a .env file in the project root (never
commit it). Add --shards N --shard i to split the crawl over N workers. It appends to
data/raw/ and resumes where it stopped.

GitHub's GraphQL API silently truncates timelineItems when many PRs are queried at once (no
error, no hasNextPage). crawl therefore re-fetches every timeline in batches of 10 and checks
each PR by replaying its label events against its current labels, retrying failures alone.
"""

import argparse
import http.client
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"

# ================================================================ crawl (GitHub GraphQL)

SINCE = "2018-01-01"
BATCH, TIMELINE_BATCH = 25, 10
PR_FIELDS = """
  number title createdAt mergedAt closedAt additions deletions changedFiles baseRefName body
  author { login } mergedBy { login }
  milestone { title }
  labels(first: 50) { nodes { name } }
  files(first: 100) { totalCount nodes { path additions deletions } }
  reviews { totalCount }
  reviewThreads { totalCount }
  comments { totalCount }
  commits(last: 1) { totalCount nodes { commit { committedDate } } }
  timelineItems(first: 100, itemTypes: [LABELED_EVENT, UNLABELED_EVENT, MILESTONED_EVENT,
                                        DEMILESTONED_EVENT, READY_FOR_REVIEW_EVENT]) {
    totalCount
    nodes {
      __typename
      ... on LabeledEvent { createdAt label { name } actor { login } }
      ... on UnlabeledEvent { createdAt label { name } actor { login } }
      ... on MilestonedEvent { createdAt milestoneTitle }
      ... on DemilestonedEvent { createdAt milestoneTitle }
      ... on ReadyForReviewEvent { createdAt }
    }
  }
"""
TIMELINE = """timelineItems(first: 100{after}, itemTypes: [LABELED_EVENT, UNLABELED_EVENT, MILESTONED_EVENT,
                                                  DEMILESTONED_EVENT, READY_FOR_REVIEW_EVENT]) {{
    pageInfo {{ hasNextPage endCursor }}
    nodes {{
      __typename
      ... on LabeledEvent {{ createdAt label {{ name }} actor {{ login }} }}
      ... on UnlabeledEvent {{ createdAt label {{ name }} actor {{ login }} }}
      ... on MilestonedEvent {{ createdAt milestoneTitle }}
      ... on DemilestonedEvent {{ createdAt milestoneTitle }}
      ... on ReadyForReviewEvent {{ createdAt }}
    }}
  }}"""
TRANSIENT = (OSError, http.client.HTTPException, ValueError)   # URLError, resets, timeouts, bad JSON


def load_token():
    if os.environ.get("GITHUB_TOKEN"):
        return os.environ["GITHUB_TOKEN"]
    env = ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        k, _, v = line.strip().partition("=")
        if k.strip() == "GITHUB_TOKEN":
            return v.strip().strip('"').strip("'")
    sys.exit("GITHUB_TOKEN not set (environment or .env)")


def post(token, q):
    req = urllib.request.Request(
        "https://api.github.com/graphql", data=json.dumps({"query": q}).encode(),
        headers={"Authorization": f"bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def respect_rate_limit(d):
    rl = (d.get("data") or {}).get("rateLimit") or {}
    if rl.get("remaining", 5000) < 200:
        reset = time.mktime(time.strptime(rl["resetAt"], "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
        time.sleep(max(0, reset - time.time()) + 30)


def fetch(token, numbers):
    """Fetch a batch of PRs; on timeouts or server errors, split it until single PRs fail alone."""
    body = "\n".join(f"p{n}: pullRequest(number: {n}) {{ {PR_FIELDS} }}" for n in numbers)
    q = f'{{ rateLimit {{ cost remaining resetAt }} repository(owner: "kubernetes", name: "kubernetes") {{ {body} }} }}'
    for attempt in range(3):
        try:
            d = post(token, q)
            break
        except TRANSIENT as e:
            if getattr(e, "code", None) in (403, 429):          # secondary rate limit
                time.sleep(60 * (attempt + 1))
                continue
            if len(numbers) > 1:
                mid = len(numbers) // 2
                return fetch(token, numbers[:mid]) + fetch(token, numbers[mid:])
            time.sleep(5 * (attempt + 1))
    else:
        return [{"number": n, "_error": "request failed"} for n in numbers]
    respect_rate_limit(d)
    repo = (d.get("data") or {}).get("repository") or {}
    return [repo.get(f"p{n}") or {"number": n, "_error": "missing"} for n in numbers]


def read_jsonl(pattern):
    """Records from data/raw/<pattern>, skipping a line cut off by a crash."""
    for path in sorted(RAW.glob(pattern)):
        with open(path) as f:
            for line in f:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


def crawl_prs(token, repo, shards, shard):
    """Every PR merged on master since 2018 (labels, files, reviews, bodies, first timeline page)."""
    merges = load_merges(repo)
    todo = sorted(merges.loc[merges.merged_at >= SINCE, "pr"].unique().tolist())
    suffix = f".s{shard}of{shards}" if shards > 1 else ""
    out_path, failed_path = RAW / f"prs_graphql{suffix}.jsonl", RAW / f"prs_failed{suffix}.txt"
    done = {d["number"] for d in read_jsonl("prs_graphql*.jsonl")}
    todo = [n for n in todo if n not in done and n % shards == shard]
    print(f"PRs: {len(done)} already crawled, {len(todo)} to go", flush=True)
    t0 = time.time()
    with open(out_path, "a") as out, open(failed_path, "a") as failed:
        for i in range(0, len(todo), BATCH):
            for rec in fetch(token, todo[i:i + BATCH]):
                if "_error" in rec:
                    failed.write(f"{rec['number']}\t{rec['_error']}\n")
                else:
                    out.write(json.dumps(rec) + "\n")
            out.flush()
            n = i + BATCH
            if n % 1000 < BATCH:
                rate = n / (time.time() - t0)
                print(f"{min(n, len(todo))}/{len(todo)}  ~{(len(todo) - n) / rate / 60:.0f} min left", flush=True)


def repo_labels(token=None):
    """Labels that exist in kubernetes/kubernetes today (cached in data/raw/repo_labels.json)."""
    path = RAW / "repo_labels.json"
    if path.exists():
        return set(json.loads(path.read_text()))
    names, after = [], "null"
    while True:
        d = post(token, f'{{ repository(owner: "kubernetes", name: "kubernetes") {{ labels(first: 100, '
                        f'after: {after}) {{ pageInfo {{ hasNextPage endCursor }} nodes {{ name }} }} }} }}')
        page = d["data"]["repository"]["labels"]
        names += [x["name"] for x in page["nodes"]]
        if not page["pageInfo"]["hasNextPage"]:
            break
        after = json.dumps(page["pageInfo"]["endCursor"])
    path.write_text(json.dumps(sorted(names), indent=0))
    return set(names)


def complete(events, labels, repo):
    """The label state replayed from the timeline equals the PR's current labels, over labels
    that still exist (deleted labels vanish from PRs without an unlabel event)."""
    last = {}
    for e in events:
        if e["__typename"] in ("LabeledEvent", "UnlabeledEvent") and e.get("label"):
            last[e["label"]["name"]] = e["__typename"]
    added = {name for name, kind in last.items() if kind == "LabeledEvent"}
    return added & repo == labels & repo


def load_timelines():
    """Last verified-timeline record per PR across all output files."""
    return {d["number"]: d for d in read_jsonl("timelines_full*.jsonl")}


def fetch_pages(token, cursors):
    """One query for {pr: cursor or None}; returns {pr: timelineItems} (missing on failure)."""
    parts = []
    for n, cur in cursors.items():
        after = f", after: {json.dumps(cur)}" if cur else ""
        parts.append(f"p{n}: pullRequest(number: {n}) {{ {TIMELINE.format(after=after)} }}")
    q = (f'{{ rateLimit {{ cost remaining resetAt }} repository(owner: "kubernetes", name: "kubernetes") '
         f'{{ {" ".join(parts)} }} }}')
    for attempt in range(3):
        try:
            d = post(token, q)
            break
        except TRANSIENT as e:
            time.sleep((60 if getattr(e, "code", None) in (403, 429) else 5) * (attempt + 1))
    else:
        return {}
    respect_rate_limit(d)
    repo = (d.get("data") or {}).get("repository") or {}
    return {n: repo[f"p{n}"]["timelineItems"] for n in cursors if repo.get(f"p{n}")}


def fetch_full(token, numbers):
    """Full timelines for a batch, following hasNextPage per PR."""
    events = {n: [] for n in numbers}
    pages = {n: 0 for n in numbers}
    cursors = {n: None for n in numbers}
    while cursors:
        got = fetch_pages(token, cursors)
        nxt = {}
        for n in cursors:
            if n not in got:
                events.pop(n, None)
                continue
            events[n] += got[n]["nodes"]
            pages[n] += 1
            if got[n]["pageInfo"]["hasNextPage"]:
                nxt[n] = got[n]["pageInfo"]["endCursor"]
        cursors = nxt
    return {n: (events[n], pages[n]) for n in events}


def refetch_timelines(token, shards, shard):
    """Full label timelines in batches of 10, each checked with complete(); failures retried alone
    up to 3 times. A re-run fetches PRs with no record and retries, alone, records that failed."""
    labels = {d["number"]: {x["name"] for x in d["labels"]["nodes"]} for d in read_jsonl("prs_graphql*.jsonl")}
    repo = repo_labels(token)
    suffix = f".s{shard}of{shards}" if shards > 1 else ""
    recs = load_timelines()
    mine = sorted(n for n in labels if n % shards == shard)
    todo = [n for n in mine if n not in recs]
    retry = [n for n in mine if n in recs and not recs[n].get("single")
             and not complete(recs[n]["events"], labels[n], repo)]
    print(f"timelines: {len(recs)} done, {len(todo)} to go, {len(retry)} to retry alone", flush=True)
    t0, n_ok = time.time(), 0
    with open(RAW / f"timelines_full{suffix}.jsonl", "a") as out:
        work = [todo[i:i + TIMELINE_BATCH] for i in range(0, len(todo), TIMELINE_BATCH)] + [[n] for n in retry]
        for i, batch in enumerate(work):
            res = fetch_full(token, batch)
            for n in batch:
                ok = n in res and complete(res[n][0], labels[n], repo)
                single = len(batch) == 1
                for _ in range(3):                      # retry failures alone
                    if ok:
                        break
                    got = fetch_full(token, [n])
                    single = True
                    if n in got:
                        res[n] = got[n]
                        ok = complete(res[n][0], labels[n], repo)
                if n not in res:
                    continue
                n_ok += ok
                out.write(json.dumps({"number": n, "events": res[n][0], "pages": res[n][1],
                                      "complete": ok, "single": single}) + "\n")
            out.flush()
            if (i + 1) % 100 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"{i + 1}/{len(work)} queries  complete {n_ok}  ~{(len(work) - i - 1) / rate / 60:.0f} min left",
                      flush=True)
    print(f"timelines done: {n_ok} complete", flush=True)


def crawl(repo, shards=1, shard=0):
    token = load_token()
    RAW.mkdir(parents=True, exist_ok=True)
    crawl_prs(token, repo, shards, shard)
    refetch_timelines(token, shards, shard)


# ================================================================ git history

MERGE_RE = re.compile(r"^Merge pull request #(\d+) from ([^/\s]+)/(\S+)")
REVERT_BRANCH_RE = re.compile(r"^revert-(\d+)-")
TITLE_PR_RE = re.compile(r"#(\d{3,6})")
REVERTS_COMMIT_RE = re.compile(r"This reverts commit ([0-9a-f]{40})")
BOTS = {"k8s-ci-robot", "dependabot", "k8s-infra-cherrypick-robot", "kubernetes-prow"}


def load_merges(repo):
    """First-parent merge commits on master: one row per merged PR."""
    fmt = "%x1e%H%x1f%cI%x1f%P%x1f%s%x1f%b"
    raw = subprocess.run(["git", "-C", repo, "log", "--first-parent", f"--format={fmt}", "master"],
                         capture_output=True, text=True, check=True).stdout
    rows = []
    for rec in raw.split("\x1e")[1:]:
        sha, date, parents, subject, body = rec.split("\x1f", 4)
        m = MERGE_RE.match(subject)
        if not m:
            continue
        parents = parents.split()
        rows.append({"sha": sha, "merged_at": pd.Timestamp(date).tz_convert("UTC"), "pr": int(m.group(1)),
                     "author": m.group(2), "branch": m.group(3),
                     "title": body.strip().splitlines()[0] if body.strip() else "",
                     "pr_head": parents[1] if len(parents) > 1 else None})
    return pd.DataFrame(rows)


def reverted_commits_by_pr(repo, df):
    """For revert PRs, collect 'This reverts commit X' lines from their own commits."""
    out = {}
    for r in df.itertuples():
        if r.pr_head is None:
            continue
        log = subprocess.run(["git", "-C", repo, "log", "--format=%B", f"{r.sha}^1..{r.pr_head}"],
                             capture_output=True, text=True).stdout
        shas = REVERTS_COMMIT_RE.findall(log)
        if shas:
            out[r.pr] = shas
    return out


def link_reverts_git(repo, df):
    """Revert -> original links from git: revert-button branch, '#N' in the title, the commit
    recorded by git revert, or the original's title quoted in 'Revert "..."'."""
    by_sha = dict(zip(df.sha, df.pr))
    by_title = {}
    for r in df.sort_values("merged_at").itertuples():
        by_title.setdefault(r.title.strip().lower(), r.pr)
    revs = df[df.title.str.match(r"(?i)^\s*revert\b") | df.branch.str.match(r"^revert-\d+-")]
    commits = reverted_commits_by_pr(repo, revs)
    links = []
    for r in revs.itertuples():
        targets = set()
        m = REVERT_BRANCH_RE.match(r.branch)
        if m:
            targets.add(int(m.group(1)))
        targets.update(int(x) for x in TITLE_PR_RE.findall(r.title))
        for s in commits.get(r.pr, []):
            if s in by_sha:
                targets.add(by_sha[s])
        inner = re.match(r'(?i)^\s*revert\s+"(.+)"\s*$', r.title)
        if inner and inner.group(1).strip().lower() in by_title:
            targets.add(by_title[inner.group(1).strip().lower()])
        links += [{"orig_pr": t, "revert_pr": r.pr} for t in targets if t != r.pr]
    return pd.DataFrame(links)


# ================================================================ dataset

WINDOW_WEEKS = 12
HORIZON_DAYS = 30         # Y1, Y2: within 30 days of merge
ESCAPE_DAYS = 90          # Y3: problems surfacing within 90 days after the release
LINK_MAX_DAYS = 365
SIZES = {"size/XS": 0, "size/S": 1, "size/M": 2, "size/L": 3, "size/XL": 4, "size/XXL": 5}
KINDS = ["feature", "bug", "cleanup", "api-change", "documentation", "regression",
         "failing-test", "flake", "deprecation"]
FIX_KINDS = {"kind/bug", "kind/regression", "kind/failing-test", "kind/flake"}
BLOCKING = re.compile(r"^(do-not-merge/.*|needs-rebase)$")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
PR_REF = re.compile(r"(?:(?<![\w/])#|kubernetes/kubernetes#|github\.com/kubernetes/kubernetes/pull/)(\d{4,6})\b")
REVERT_BODY = re.compile(r"(?i)\brevert(?:s|ing|ed)?\b[^\n]{0,40}?"
                         r"(?:kubernetes/kubernetes#|#|github\.com/kubernetes/kubernetes/pull/)(\d{4,6})")
FIX_CUES = re.compile(r"(?i)regress|introduced (?:in|by)|broke|broken|caused by|since #|flak|revert")


def ts(s):
    return pd.Timestamp(s) if s else pd.NaT


def area_dirs(paths):
    """Coarse code areas: pkg/X, cmd/X, staging/X, test/X, else the top-level dir."""
    out = set()
    for p in paths:
        parts = p.split("/")
        if parts[0] == "staging" and len(parts) > 3:
            out.add("staging/" + parts[3])
        elif parts[0] in ("pkg", "cmd", "test", "plugin", "hack") and len(parts) > 2:
            out.add("/".join(parts[:2]))
        else:
            out.add(parts[0])
    return out


def load_crawl():
    """Crawled PRs with their verified timelines (first record per PR)."""
    full, repo = load_timelines(), repo_labels()
    recs, seen = [], set()
    for d in read_jsonl("prs_graphql*.jsonl"):
        if d["number"] in seen:
            continue
        seen.add(d["number"])
        if d["number"] in full:
            d["events"] = full[d["number"]]["events"]
            d["timeline_complete"] = complete(d["events"], {x["name"] for x in d["labels"]["nodes"]}, repo)
        else:                       # no re-fetch: the first page only, flagged as unverified
            d["events"] = d["timelineItems"]["nodes"]
            d["timeline_complete"] = False
        recs.append(d)
    return recs


def readiness(d, merged):
    """Last moment the PR became mergeable: lgtm/approved added or a blocking label removed."""
    last = {"lgtm": pd.NaT, "approved": pd.NaT, "unblock": pd.NaT}
    lgtm_removed = 0
    for e in d["events"]:
        if e["__typename"] not in ("LabeledEvent", "UnlabeledEvent"):
            continue
        t, name = ts(e["createdAt"]), e["label"]["name"]
        if t > merged:
            continue
        if e["__typename"] == "LabeledEvent" and name in ("lgtm", "approved"):
            last[name] = t if pd.isna(last[name]) else max(last[name], t)
        elif e["__typename"] == "UnlabeledEvent":
            if name == "lgtm":
                lgtm_removed += 1
            if BLOCKING.match(name):
                last["unblock"] = t if pd.isna(last["unblock"]) else max(last["unblock"], t)
    # PRs with no lgtm/approved/unblock event fall back to merge time and are flagged. With the
    # verified timelines this is 0 PRs; the fallback stays for re-runs on partial data.
    found = [v for v in last.values() if not pd.isna(v)]
    if not pd.isna(last["lgtm"]) and not pd.isna(last["approved"]):
        quality = "full"
    elif found:
        quality = "partial"
    else:
        return merged, lgtm_removed, "imputed"
    return max(found), lgtm_removed, quality


def changed_files():
    """{pr: {path: (additions, deletions)}} from the crawl (last record per PR)."""
    return {d["number"]: {x["path"]: (x["additions"], x["deletions"]) for x in d["files"]["nodes"]}
            for d in read_jsonl("prs_graphql*.jsonl")}


def inverse(fa, fb):
    """Does the revert undo the original? "exact" (every file of the original, additions and
    deletions swapped), "partial" (some files swapped), "shared" (shared files, nothing
    swapped), "none" (no shared files), or "no data"."""
    if not fa or not fb:
        return "no data"
    common = set(fa) & set(fb)
    if not common:
        return "none"
    swapped = sum(fa[p] == fb[p][::-1] for p in common)
    if swapped == len(common) == len(fa):
        return "exact"
    return "partial" if swapped else "shared"


def dataset(repo):
    crawl_recs = load_crawl()
    # The data end on the day of the crawl's last merge; later events are not observed.
    observed_until = max(ts(d["mergedAt"]) for d in crawl_recs if d.get("mergedAt")).floor("D")
    cal = pd.read_csv(DATA / "k8s_freeze_calendar.csv", dtype={"release": str})
    cal["freeze"] = pd.to_datetime(cal.code_freeze_utc, utc=True)
    cal["release_at"] = pd.to_datetime(cal.release_date, utc=True) + pd.Timedelta(hours=23, minutes=59)
    cal = cal[cal.freeze < observed_until - pd.Timedelta(days=HORIZON_DAYS)]

    git = load_merges(repo)
    git_by_pr = git.set_index("pr")

    rows, bodies = [], {}
    for d in crawl_recs:
        n = d["number"]
        if n not in git_by_pr.index or not d.get("mergedAt"):
            continue
        merged = ts(d["mergedAt"])
        labels = {x["name"] for x in d["labels"]["nodes"]}
        ready, lgtm_removed, ready_quality = readiness(d, merged)
        last_commit = (d["commits"]["nodes"] or [{}])[-1].get("commit", {}).get("committedDate")
        body = HTML_COMMENT.sub("", d.get("body") or "")
        bodies[n] = (d["title"], body)
        rows.append({
            "pr": n, "author": git_by_pr.at[n, "author"], "branch": git_by_pr.at[n, "branch"],
            "title": d["title"], "created_at": ts(d["createdAt"]), "merged_at": merged, "ready_at": ready,
            "ready_quality": ready_quality,
            "last_commit_at": ts(last_commit), "additions": d["additions"], "deletions": d["deletions"],
            "files": d["changedFiles"], "files_truncated": d["files"]["totalCount"] > 100,
            "areas": sorted(area_dirs(f["path"] for f in d["files"]["nodes"])),
            "size": next((v for k, v in SIZES.items() if k in labels), np.nan),
            **{f"kind_{k}": f"kind/{k}" in labels for k in KINDS},
            "kind_none": not any(x.startswith("kind/") for x in labels),
            "n_sigs": sum(x.startswith("sig/") for x in labels),
            "has_milestone": d.get("milestone") is not None,
            "priority_critical": "priority/critical-urgent" in labels,
            "is_fix_kind": bool(labels & FIX_KINDS),
            "reviews": d["reviews"]["totalCount"], "review_threads": d["reviewThreads"]["totalCount"],
            "comments": d["comments"]["totalCount"], "lgtm_removed": lgtm_removed,
            "timeline_complete": d["timeline_complete"],
        })
    df = pd.DataFrame(rows)
    df["is_bot"] = df.author.isin(BOTS)
    df["is_revert"] = df.title.str.match(r"(?i)^\s*revert\b") | df.branch.str.match(r"^revert-\d+-")
    merged_at = dict(zip(df.pr, df.merged_at))
    files = changed_files()

    # ---- Y1: reverts (git heuristics + 'Reverts #N' in the revert PR body), checked by inverse()
    links = [(r.orig_pr, r.revert_pr, "git") for r in link_reverts_git(repo, git).itertuples()]
    for r in df[df.is_revert].itertuples():
        for m in REVERT_BODY.findall(bodies[r.pr][1]):
            links.append((int(m), r.pr, "body"))
    lr = pd.DataFrame(links, columns=["orig_pr", "revert_pr", "source"])
    lr = lr[lr.orig_pr.isin(merged_at) & lr.revert_pr.isin(merged_at) & (lr.orig_pr != lr.revert_pr)]
    lr = lr.groupby(["orig_pr", "revert_pr"]).source.agg(lambda s: "+".join(sorted(set(s)))).reset_index()
    lr["event_at"] = lr.revert_pr.map(merged_at)
    lr["lag_days"] = (lr.event_at - lr.orig_pr.map(merged_at)).dt.total_seconds() / 86400
    lr = lr[lr.lag_days.between(0, LINK_MAX_DAYS)].copy()
    lr["inverse"] = [inverse(files.get(a), files.get(b)) for a, b in zip(lr.orig_pr, lr.revert_pr)]
    lr["verified"] = lr.inverse.isin(["exact", "partial"])
    lr.to_csv(DATA / "links_reverts.csv", index=False)

    # ---- Y2: follow-up fixes (later fix-kind PR citing the original); cue words and file overlap
    fixes = []
    for r in df[df.is_fix_kind & ~df.is_revert].itertuples():
        title, body = bodies[r.pr]
        text = title + "\n" + body
        strict = bool(FIX_CUES.search(text))
        for m in sorted(set(PR_REF.findall(text)), key=int):    # sorted: a set's order varies between runs
            o = int(m)
            if o != r.pr and o in merged_at:
                fixes.append((o, r.pr, strict))
    lf = pd.DataFrame(fixes, columns=["orig_pr", "fix_pr", "strict"])
    lf["event_at"] = lf.fix_pr.map(merged_at)
    lf["lag_days"] = (lf.event_at - lf.orig_pr.map(merged_at)).dt.total_seconds() / 86400
    lf = lf[lf.lag_days.between(0, LINK_MAX_DAYS) & ~lf.orig_pr.isin(df.pr[df.is_revert])].copy()
    lf["file_overlap"] = [bool(set(files.get(a, {})) & set(files.get(b, {}))) for a, b in zip(lf.orig_pr, lf.fix_pr)]
    lf.to_csv(DATA / "links_fixes.csv", index=False)

    # ---- Units: ready within 12 weeks before a freeze; assign that cycle
    unit_rows = []
    for c in cal.itertuples():
        lo = c.freeze - pd.Timedelta(weeks=WINDOW_WEEKS)
        w = df[(df.ready_at >= lo) & (df.ready_at < c.freeze)].copy()
        w["release"], w["freeze"], w["release_at"] = c.release, c.freeze, c.release_at
        unit_rows.append(w)
    u = pd.concat(unit_rows).sort_values("ready_at")
    u["hours_to_freeze"] = (u.freeze - u.ready_at).dt.total_seconds() / 3600
    u = u.sort_values("hours_to_freeze").drop_duplicates("pr")
    u["week_before"] = (u.hours_to_freeze // 168).astype(int) + 1
    u["final_week"] = u.week_before == 1
    u["merged_after_freeze"] = u.merged_at >= u.freeze

    def first_event(links, max_lag=HORIZON_DAYS):
        return links[links.lag_days <= max_lag].groupby("orig_pr").event_at.min()

    rev_at, fix_at = first_event(lr), first_event(lf)
    fix_strict_at = first_event(lf[lf.strict])
    u["y1_revert"] = u.pr.isin(rev_at.index)
    u["y2_fix"] = u.pr.isin(fix_at.index)
    u["y2_fix_strict"] = u.pr.isin(fix_strict_at.index)
    u["first_problem_at"] = u.pr.map(pd.concat([rev_at, fix_at], axis=1).min(axis=1))
    u["any_problem"] = u.y1_revert | u.y2_fix
    u["problem_in_freeze"] = u.first_problem_at.between(u.freeze, u.release_at)

    # Y3 (problem after release): any revert or fix in (release, release + 90d]. Same calendar
    # window for every PR in a cycle. NaN if the window had not closed when the data end.
    events = pd.concat([lr[["orig_pr", "event_at"]], lf[["orig_pr", "event_at"]]])
    ev = events.merge(u[["pr", "release_at"]], left_on="orig_pr", right_on="pr")
    hit = ev[(ev.event_at > ev.release_at) & (ev.event_at <= ev.release_at + pd.Timedelta(days=ESCAPE_DAYS))]
    u["y3_post_release"] = u.pr.isin(hit.orig_pr).astype(float)
    u.loc[u.release_at + pd.Timedelta(days=ESCAPE_DAYS) > observed_until, "y3_post_release"] = np.nan

    # ---- Covariates and mediators
    hist = {a: np.sort(s.values) for a, s in git.groupby("author").merged_at}
    u["author_prior_prs"] = [int(np.searchsorted(hist[a], t.to_datetime64())) if a in hist else 0
                             for a, t in zip(u.author, u.ready_at.dt.tz_convert(None))]
    u["age_days"] = (u.ready_at - u.created_at).dt.total_seconds() / 86400
    u["lines"] = u.additions + u.deletions
    u["queue_wait_h"] = (u.merged_at - u.ready_at).dt.total_seconds() / 3600
    u["last_commit_to_ready_h"] = (u.ready_at - u.last_commit_at).dt.total_seconds() / 3600

    # ---- Congestion exposures (all master merges, bots included: they occupy the queue too)
    times = git.merged_at.sort_values().tolist()
    areas_of = dict(zip(df.pr, df.areas))
    by_time = df.sort_values("merged_at")[["pr", "merged_at"]]
    t_list, pr_list = by_time.merged_at.tolist(), by_time.pr.tolist()
    tot6, area12 = [], []
    for r in u.itertuples():
        h6, h12 = pd.Timedelta(hours=6), pd.Timedelta(hours=12)
        tot6.append(bisect_right(times, r.merged_at + h6) - bisect_left(times, r.merged_at - h6) - 1)
        lo, hi = bisect_left(t_list, r.merged_at - h12), bisect_right(t_list, r.merged_at + h12)
        mine = set(r.areas)
        area12.append(sum(1 for p in pr_list[lo:hi] if p != r.pr and mine & set(areas_of.get(p, ()))))
    u["conc_total_6h"] = tot6
    u["conc_area_12h"] = area12

    u = u[~u.is_bot & ~u.is_revert]
    u.drop(columns=["areas"]).to_csv(DATA / "analysis.csv.gz", index=False)
    print(f"data observed until {observed_until.date()} | crawled PRs: {len(df)} | units: {len(u)} in "
          f"{u.release.nunique()} cycles | readiness quality: {u.ready_quality.value_counts().to_dict()}")
    print(f"Y1 revert: {u.y1_revert.sum()}  Y2 fix: {u.y2_fix.sum()} (strict {u.y2_fix_strict.sum()})  "
          f"Y3 post-release: {int(u.y3_post_release.sum())} (censored {u.y3_post_release.isna().sum()})  "
          f"merged after freeze: {u.merged_after_freeze.sum()}")
    print(f"revert links: {len(lr)} (verified {lr.verified.mean():.1%})  fix links: {len(lf)} "
          f"(cue words {lf.strict.mean():.1%}, shared file {lf.file_overlap.mean():.1%})")


# ================================================================ hand coding of fix links

CENSUS_SEED, CENSUS_DAYS = 2027, 180
LABEL = re.compile(r"^(kind|area|sig|priority)/")


def census_links():
    """Every fix link the checked outcomes can use: cue words AND a shared file, on an analysis
    unit, within 180 days of merge (soak analysis) or within 90 days after the release (Y3)."""
    u = pd.read_csv(DATA / "analysis.csv.gz", usecols=["pr", "release_at"], parse_dates=["release_at"])
    lf = pd.read_csv(DATA / "links_fixes.csv", parse_dates=["event_at"])
    lf = lf[lf.strict & lf.file_overlap].merge(u, left_on="orig_pr", right_on="pr")
    after_release = (lf.event_at > lf.release_at) & (lf.event_at <= lf.release_at + pd.Timedelta(days=ESCAPE_DAYS))
    return lf[(lf.lag_days <= CENSUS_DAYS) | after_release][["orig_pr", "fix_pr"]].drop_duplicates()


def pr_text(numbers, body_chars=5000, max_files=60):
    """What the coder sees: number, title, description, kind/area/sig/priority labels, files.
    No dates, lags, or late/early group."""
    out = {}
    for d in read_jsonl("prs_graphql*.jsonl"):
        if d["number"] in numbers and d["number"] not in out:
            body = HTML_COMMENT.sub("", d.get("body") or "").strip()
            body = re.sub(r"\n{3,}", "\n\n", body)
            out[d["number"]] = {
                "number": d["number"], "title": d["title"],
                "labels": sorted(x["name"] for x in d["labels"]["nodes"] if LABEL.match(x["name"])),
                "description": body[:body_chars] + (" [...]" if len(body) > body_chars else ""),
                "files_changed": d["files"]["totalCount"],
                "files": [f"{x['path']} (+{x['additions']}/-{x['deletions']})" for x in d["files"]["nodes"][:max_files]]}
    return out


def census_page(out):
    """Write the coding page's data (JSON) for the census links. A new census shuffles the links
    under neutral ids (so a census stopped early is a random sample) and writes
    data/validation/human_census_key.csv; an existing key is reused, never overwritten."""
    links = census_links()
    key = DATA / "validation" / "human_census_key.csv"
    if key.exists():
        rows = pd.read_csv(key)
        if set(map(tuple, rows[["orig_pr", "fix_pr"]].values)) != set(map(tuple, links.values)):
            sys.exit(f"{key.name} covers different links than the current data; not overwritten.")
    else:
        rows = links.sort_values(["orig_pr", "fix_pr"]).sample(frac=1, random_state=CENSUS_SEED).reset_index(drop=True)
        rows.insert(0, "id", [f"H{i + 1:03d}" for i in range(len(rows))])
        key.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(key, index=False)
    prs = pr_text(set(rows.orig_pr) | set(rows.fix_pr))
    data = {"items": [{"id": r.id, "a": int(r.orig_pr), "b": int(r.fix_pr)} for r in rows.itertuples()],
            "prs": {str(k): v for k, v in prs.items()}}
    Path(out).write_text(json.dumps(data, separators=(",", ":")))
    print(f"{len(rows)} links, {len(prs)} PRs with text -> {key.relative_to(ROOT)}, {out}")


# ================================================================ main

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="step", required=True)
    p = sub.add_parser("crawl", help="PR metadata and verified label timelines -> data/raw/")
    p.add_argument("repo", help="bare clone of kubernetes/kubernetes")
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p = sub.add_parser("dataset", help="data/analysis.csv.gz and the outcome links")
    p.add_argument("repo", help="bare clone of kubernetes/kubernetes")
    p = sub.add_parser("census-page", help="page data for hand-coding the fix links")
    p.add_argument("out", help="output JSON")
    a = ap.parse_args()
    if a.step == "crawl":
        crawl(a.repo, a.shards, a.shard)
    elif a.step == "dataset":
        dataset(a.repo)
    else:
        census_page(a.out)
