"""Re-fetch the LAST 100 timeline events for PRs whose timeline exceeds 100 items.

The main crawl takes the first 100 events. Readiness (the final lgtm/approved and the
removal of blocking labels) sits at the end of the timeline, so long PRs need their tail.
Appends to data/raw/timeline_tail.jsonl and can be re-run as the main crawl grows.

Usage:
    python crawl/refetch_timeline_tail.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crawl_prs  # noqa: E402

RAW = crawl_prs.ROOT / "data" / "raw"
OUT = RAW / "timeline_tail.jsonl"
TAIL_FIELDS = crawl_prs.PR_FIELDS[crawl_prs.PR_FIELDS.index("timelineItems"):].replace("first: 100", "last: 100")
TAIL_FIELDS = "number " + TAIL_FIELDS


def main():
    token = crawl_prs.load_token()
    done = set()
    if OUT.exists():
        with open(OUT) as f:
            done = {json.loads(line)["number"] for line in f}
    todo = []
    for path in sorted(RAW.glob("prs_graphql*.jsonl")):
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d["timelineItems"]["totalCount"] > 100 and d["number"] not in done:
                    todo.append(d["number"])
    print(f"{len(todo)} long timelines to re-fetch", flush=True)
    crawl_prs.PR_FIELDS = TAIL_FIELDS
    with open(OUT, "a") as out:
        for i in range(0, len(todo), crawl_prs.BATCH):
            for rec in crawl_prs.fetch(token, todo[i:i + crawl_prs.BATCH]):
                if "_error" not in rec:
                    out.write(json.dumps(rec) + "\n")
            out.flush()
            time.sleep(0.5)
    print("done", flush=True)


if __name__ == "__main__":
    main()
