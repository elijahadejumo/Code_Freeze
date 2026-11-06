"""Prepare blind inputs for LLM coding of fix links, and the human check subset.

Rows: the 50 fix links (fix_links_sample.csv) + the 60 post-release links
(escape_links_sample.csv), shuffled under neutral ids. Reverts are verified automatically
(auto_validate_links.py) and are not coded. Coders see each PR's title, description
(HTML comments removed, truncated), kind/area/sig/priority labels and changed files. They do
not see dates, the time between the PRs, or whether the original was a late PR, so the
coding cannot depend on the comparison being studied.

Writes
  <outdir>/chunk_{k}.jsonl          coder input (outside the project, so coders see nothing else)
  data/validation/llm_rows_key.csv  id -> sample, orig_pr, second_pr
  data/validation/human_check_20.csv  20 escape rows (10 late, 10 early, shuffled) for a person

Usage:
    python analysis/prepare_llm_coding.py OUTDIR [--chunks 3]
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "data" / "validation"
RAW = ROOT / "data" / "raw"
URL = "https://github.com/kubernetes/kubernetes/pull/{}"
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
LABEL = re.compile(r"^(kind|area|sig|priority)/")
SEED = 2027
BODY_CHARS = 1500
MAX_FILES = 25


def load_prs(numbers, body_chars=BODY_CHARS, max_files=MAX_FILES):
    out = {}
    for path in sorted(RAW.glob("prs_graphql*.jsonl")):
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d["number"] in numbers and d["number"] not in out:
                    body = HTML_COMMENT.sub("", d.get("body") or "").strip()
                    body = re.sub(r"\n{3,}", "\n\n", body)
                    out[d["number"]] = {
                        "number": d["number"], "title": d["title"],
                        "labels": sorted(l["name"] for l in d["labels"]["nodes"] if LABEL.match(l["name"])),
                        "description": body[:body_chars] + (" [...]" if len(body) > body_chars else ""),
                        "files_changed": d["files"]["totalCount"],
                        "files": [f"{x['path']} (+{x['additions']}/-{x['deletions']})"
                                  for x in d["files"]["nodes"][:max_files]]}
    return out


def main(outdir, chunks):
    fix = pd.read_csv(VAL / "fix_links_sample.csv").rename(columns={"fix_pr": "second_pr"}).assign(sample="fix")
    esc = pd.read_csv(VAL / "escape_links_sample.csv").assign(sample="escape")
    rows = pd.concat([fix[["sample", "orig_pr", "second_pr"]], esc[["sample", "orig_pr", "second_pr"]]])
    rows = rows.sample(frac=1, random_state=SEED).reset_index(drop=True)
    rows["id"] = [f"L{i + 1:03d}" for i in range(len(rows))]
    rows[["id", "sample", "orig_pr", "second_pr"]].to_csv(VAL / "llm_rows_key.csv", index=False)

    prs = load_prs(set(rows.orig_pr) | set(rows.second_pr))
    outdir.mkdir(parents=True, exist_ok=True)
    for k in range(chunks):
        with open(outdir / f"chunk_{k}.jsonl", "w") as f:
            for r in rows.iloc[k::chunks].itertuples():
                f.write(json.dumps({"id": r.id, "first_pr": prs.get(r.orig_pr), "second_pr": prs.get(r.second_pr)}) + "\n")

    key = pd.read_csv(VAL / "escape_links_key.csv")
    esc_ids = rows[rows["sample"] == "escape"].merge(key, on=["orig_pr", "second_pr"])
    human = pd.concat([g.sample(10, random_state=SEED) for _, g in esc_ids.groupby("group")]).sample(frac=1, random_state=SEED)
    human = human.assign(first_url=human.orig_pr.map(URL.format), second_url=human.second_pr.map(URL.format),
                         question="Does the second PR fix a defect introduced by the first?", your_answer="", note="")
    path = VAL / "human_check_20.csv"
    if path.exists():
        print(f"{path.name} exists, left untouched")
    else:
        human[["id", "first_url", "second_url", "question", "your_answer", "note"]].to_csv(path, index=False)
    print(f"{len(rows)} rows in {chunks} chunks -> {outdir}; human check: {len(human)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--chunks", type=int, default=3)
    a = ap.parse_args()
    main(a.outdir, a.chunks)
