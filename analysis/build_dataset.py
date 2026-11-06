"""Build the analysis dataset defined in PLAN.md (units, exposure, outcomes, covariates).

Inputs:  data/raw/prs_graphql.jsonl (crawl), data/raw/timeline_tail.jsonl (optional
         re-fetch of the last 100 timeline events for long PRs), the git history, and
         data/k8s_freeze_calendar.csv.
Outputs: data/analysis.csv.gz          one row per unit PR
         data/links_reverts.csv        revert -> original links (for validation)
         data/links_fixes.csv          follow-up fix -> original links (for validation)

Usage:
    python analysis/build_dataset.py path/to/k8s.git
"""

import json
import re
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pilot"))
sys.path.insert(0, str(ROOT / "crawl"))
from pilot_rush_reverts import BOTS, link_reverts, load_merges  # noqa: E402
from refetch_timelines import complete, load_records, repo_labels  # noqa: E402

RAW = ROOT / "data" / "raw"
WINDOW_WEEKS = 12
HORIZON_DAYS = 30
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
    # The main crawl's timelines are silently truncated (batched GraphQL). Prefer the verified
    # re-fetch (crawl/refetch_timelines.py); fall back to crawl + tail events where it is missing.
    full, repo = load_records(), repo_labels()
    tails = {}
    if (RAW / "timeline_tail.jsonl").exists():
        with open(RAW / "timeline_tail.jsonl") as f:
            for line in f:
                d = json.loads(line)
                tails[d["number"]] = d["timelineItems"]["nodes"]
    recs, seen_prs = [], set()
    lines = (line for path in sorted(RAW.glob("prs_graphql*.jsonl")) for line in open(path))
    for line in lines:
        try:
            d = json.loads(line)
        except ValueError:          # a line cut off by a crash
            continue
        if d["number"] in seen_prs:
            continue
        seen_prs.add(d["number"])
        if d["number"] in full:
            d["events"] = full[d["number"]]["events"]
            d["timeline_complete"] = complete(d["events"], {x["name"] for x in d["labels"]["nodes"]}, repo)
            recs.append(d)
            continue
        events = d["timelineItems"]["nodes"] + tails.get(d["number"], [])
        seen, uniq = set(), []
        for e in events:
            key = (e["__typename"], e.get("createdAt"), (e.get("label") or {}).get("name"))
            if key not in seen:
                seen.add(key)
                uniq.append(e)
        d["events"] = uniq
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


def main(repo):
    cal = pd.read_csv(ROOT / "data" / "k8s_freeze_calendar.csv", dtype={"release": str})
    cal["freeze"] = pd.to_datetime(cal.code_freeze_utc, utc=True)
    cal["release_at"] = pd.to_datetime(cal.release_date, utc=True) + pd.Timedelta(hours=23, minutes=59)
    cal = cal[cal.freeze < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=HORIZON_DAYS)]

    git = load_merges(repo)
    git_by_pr = git.set_index("pr")
    crawl = load_crawl()

    rows, bodies = [], {}
    for d in crawl:
        n = d["number"]
        if n not in git_by_pr.index or not d.get("mergedAt"):
            continue
        merged = ts(d["mergedAt"])
        labels = {l["name"] for l in d["labels"]["nodes"]}
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
            "kind_none": not any(l.startswith("kind/") for l in labels),
            "n_sigs": sum(l.startswith("sig/") for l in labels),
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

    # ---- Y1: reverts (git heuristics from the pilot + 'Reverts #N' in the revert PR body)
    git_links, _ = link_reverts(repo, git)
    links = [(r.orig_pr, r.revert_pr, "git") for r in git_links.itertuples()]
    for r in df[df.is_revert].itertuples():
        for m in REVERT_BODY.findall(bodies[r.pr][1]):
            links.append((int(m), r.pr, "body"))
    lr = pd.DataFrame(links, columns=["orig_pr", "revert_pr", "source"])
    lr = lr[lr.orig_pr.isin(merged_at) & lr.revert_pr.isin(merged_at) & (lr.orig_pr != lr.revert_pr)]
    lr = lr.groupby(["orig_pr", "revert_pr"]).source.agg(lambda s: "+".join(sorted(set(s)))).reset_index()
    lr["event_at"] = lr.revert_pr.map(merged_at)
    lr["lag_days"] = (lr.event_at - lr.orig_pr.map(merged_at)).dt.total_seconds() / 86400
    lr = lr[lr.lag_days.between(0, LINK_MAX_DAYS)]
    lr.to_csv(ROOT / "data" / "links_reverts.csv", index=False)

    # ---- Y2: follow-up fixes (later fix-kind PR citing the original), broad and strict
    fixes = []
    for r in df[df.is_fix_kind & ~df.is_revert].itertuples():
        title, body = bodies[r.pr]
        text = title + "\n" + body
        strict = bool(FIX_CUES.search(text))
        for m in set(PR_REF.findall(text)):
            o = int(m)
            if o != r.pr and o in merged_at:
                fixes.append((o, r.pr, strict))
    lf = pd.DataFrame(fixes, columns=["orig_pr", "fix_pr", "strict"])
    lf["event_at"] = lf.fix_pr.map(merged_at)
    lf["lag_days"] = (lf.event_at - lf.orig_pr.map(merged_at)).dt.total_seconds() / 86400
    lf = lf[lf.lag_days.between(0, LINK_MAX_DAYS) & ~lf.orig_pr.isin(df.pr[df.is_revert])]
    lf.to_csv(ROOT / "data" / "links_fixes.csv", index=False)

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

    # Y3 (escaped defect): any revert or fix in (release, release + 90d]. Same calendar
    # window for every PR in a cycle, outside the freeze. Censored if the window is open.
    events = pd.concat([lr[["orig_pr", "event_at"]], lf[["orig_pr", "event_at"]]])
    ev = events.merge(u[["pr", "release_at"]], left_on="orig_pr", right_on="pr")
    hit = ev[(ev.event_at > ev.release_at) & (ev.event_at <= ev.release_at + pd.Timedelta(days=ESCAPE_DAYS))]
    u["y3_post_release"] = u.pr.isin(hit.orig_pr).astype(float)
    open_window = u.release_at + pd.Timedelta(days=ESCAPE_DAYS) > pd.Timestamp.now(tz="UTC")
    u.loc[open_window, "y3_post_release"] = np.nan

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
    u.drop(columns=["areas"]).to_csv(ROOT / "data" / "analysis.csv.gz", index=False)
    print(f"crawled PRs: {len(df)} | units: {len(u)} in {u.release.nunique()} cycles | "
          f"readiness quality: {u.ready_quality.value_counts().to_dict()}")
    print(f"Y1 revert: {u.y1_revert.sum()}  Y2 fix: {u.y2_fix.sum()} (strict {u.y2_fix_strict.sum()})  "
          f"Y3 escaped (90d post-release): {int(u.y3_post_release.sum())} (censored {u.y3_post_release.isna().sum()})  merged after freeze: {u.merged_after_freeze.sum()}")
    print(f"revert links: {len(lr)} ({(lr.source != 'git').sum()} need the body)  fix links: {len(lf)}")


if __name__ == "__main__":
    main(sys.argv[1])
