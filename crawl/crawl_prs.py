"""Crawl PR metadata for every kubernetes/kubernetes PR merged on master since 2018.

Collects what the git-only pilot cannot see: labels (kind/size/sig/priority),
milestone, files touched, review counts, last-commit time, and the label timeline
(lgtm/approved timestamps, which mark when a PR was ready rather than when the
Prow queue merged it). Bodies are kept for linking follow-up fixes.

Output is append-only JSONL, so the crawl resumes where it stopped.

Usage:
    python crawl/crawl_prs.py path/to/k8s.git [--shards 4 --shard 0]
Reads GITHUB_TOKEN from .env in the project root. With --shards, each worker takes the
PRs where number % shards == shard and writes its own file; readers glob all of them.
"""

import argparse
import http.client
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pilot"))
from pilot_rush_reverts import load_merges  # noqa: E402

RAW = ROOT / "data" / "raw"
SINCE = "2018-01-01"
BATCH = 25

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


def load_token():
    for line in (ROOT / ".env").read_text().splitlines():
        k, _, v = line.strip().partition("=")
        if k.strip() == "GITHUB_TOKEN":
            return v.strip().strip('"').strip("'")
    sys.exit("GITHUB_TOKEN not found in .env")


TRANSIENT = (OSError, http.client.HTTPException, ValueError)   # URLError, resets, timeouts, bad JSON


def post(token, q):
    req = urllib.request.Request(
        "https://api.github.com/graphql", data=json.dumps({"query": q}).encode(),
        headers={"Authorization": f"bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def query(token, numbers):
    body = "\n".join(f"p{n}: pullRequest(number: {n}) {{ {PR_FIELDS} }}" for n in numbers)
    q = f'{{ rateLimit {{ cost remaining resetAt }} repository(owner: "kubernetes", name: "kubernetes") {{ {body} }} }}'
    return post(token, q)


def respect_rate_limit(d):
    rl = (d.get("data") or {}).get("rateLimit") or {}
    if rl.get("remaining", 5000) < 200:
        reset = time.mktime(time.strptime(rl["resetAt"], "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
        time.sleep(max(0, reset - time.time()) + 30)


def fetch(token, numbers):
    """Fetch a batch; on timeouts or server errors, split it until single PRs fail alone."""
    for attempt in range(3):
        try:
            d = query(token, numbers)
            break
        except TRANSIENT as e:
            code = getattr(e, "code", None)
            if code in (403, 429):                       # secondary rate limit
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


def crawled_numbers():
    done = set()
    for path in RAW.glob("prs_graphql*.jsonl"):
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["number"])
                except ValueError:          # a line cut off by a crash
                    pass
    return done


def main(repo, shards=1, shard=0):
    token = load_token()
    merges = load_merges(repo)
    todo = sorted(merges.loc[merges.merged_at >= SINCE, "pr"].unique().tolist())
    RAW.mkdir(parents=True, exist_ok=True)
    suffix = f".s{shard}of{shards}" if shards > 1 else ""
    OUT, FAILED = RAW / f"prs_graphql{suffix}.jsonl", RAW / f"prs_failed{suffix}.txt"
    done = crawled_numbers()
    todo = [n for n in todo if n not in done and n % shards == shard]
    print(f"{len(done)} already crawled, {len(todo)} to go", flush=True)

    t0 = time.time()
    with open(OUT, "a") as out, open(FAILED, "a") as failed:
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
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    a = ap.parse_args()
    main(a.repo, a.shards, a.shard)
