"""Re-fetch the full label/milestone timeline of every crawled PR, with a completeness check.

Why: in batched GraphQL queries GitHub silently truncates timelineItems (no error, no
hasNextPage). In the main crawl (25 PRs per query) 45% of PRs that carry `lgtm` have no
`lgtm` event, and the dropped events are the late ones that define readiness. Batches of
10 came back complete in testing, so this script uses 10, pages with cursors, and checks
every PR:

    complete  <=>  the label state replayed from the timeline equals the PR's current labels,
                   over the labels that still exist in the repo

(every current label's last event is an add, and every label whose last event is an add
is current). Labels deleted from the repo vanish from PRs without an unlabel event, so they
are left out of the comparison (data/raw/repo_labels.json). PRs that fail are retried
alone, up to 3 times, and keep complete=false if they still fail. A re-run fetches PRs
with no record and retries, alone, records that fail the check and were never retried.
Output: data/raw/timelines_full[.sKofN].jsonl, one {number, events, pages, complete,
single} per PR; the last line per PR wins. build_dataset.py prefers these over the crawl.

Usage:
    python crawl/refetch_timelines.py [--shards 4 --shard 0]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crawl_prs  # noqa: E402

RAW = crawl_prs.RAW
BATCH = 10
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


def repo_labels(token=None):
    """Labels that exist in kubernetes/kubernetes today (cached in data/raw/repo_labels.json)."""
    path = RAW / "repo_labels.json"
    if path.exists():
        return set(json.loads(path.read_text()))
    names, after = [], "null"
    while True:
        d = crawl_prs.post(token, f'{{ repository(owner: "kubernetes", name: "kubernetes") {{ labels(first: 100, '
                                  f'after: {after}) {{ pageInfo {{ hasNextPage endCursor }} nodes {{ name }} }} }} }}')
        page = d["data"]["repository"]["labels"]
        names += [x["name"] for x in page["nodes"]]
        if not page["pageInfo"]["hasNextPage"]:
            break
        after = json.dumps(page["pageInfo"]["endCursor"])
    path.write_text(json.dumps(sorted(names), indent=0))
    return set(names)


def complete(events, labels, repo):
    last = {}
    for e in events:
        if e["__typename"] in ("LabeledEvent", "UnlabeledEvent") and e.get("label"):
            last[e["label"]["name"]] = e["__typename"]
    added = {name for name, kind in last.items() if kind == "LabeledEvent"}
    return added & repo == labels & repo


def load_records():
    """Last record per PR across all output files."""
    recs = {}
    for path in sorted(RAW.glob("timelines_full*.jsonl")):
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                recs[d["number"]] = d
    return recs


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
            d = crawl_prs.post(token, q)
            break
        except crawl_prs.TRANSIENT as e:
            time.sleep((60 if getattr(e, "code", None) in (403, 429) else 5) * (attempt + 1))
    else:
        return {}
    crawl_prs.respect_rate_limit(d)
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


def main(shards=1, shard=0):
    token = crawl_prs.load_token()
    labels = {}
    for path in sorted(RAW.glob("prs_graphql*.jsonl")):
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                labels[d["number"]] = {x["name"] for x in d["labels"]["nodes"]}
    repo = repo_labels(token)
    suffix = f".s{shard}of{shards}" if shards > 1 else ""
    out_path = RAW / f"timelines_full{suffix}.jsonl"
    recs = load_records()
    mine = sorted(n for n in labels if n % shards == shard)
    todo = [n for n in mine if n not in recs]
    retry = [n for n in mine if n in recs and not recs[n].get("single")
             and not complete(recs[n]["events"], labels[n], repo)]
    print(f"{len(recs)} done, {len(todo)} to go, {len(retry)} to retry alone", flush=True)

    t0, n_ok = time.time(), 0
    with open(out_path, "a") as out:
        # batches of new PRs, then the old failures as batches of one
        work = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)] + [[n] for n in retry]
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
    print(f"done: {n_ok} complete", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    a = ap.parse_args()
    main(a.shards, a.shard)
