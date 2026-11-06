"""Prepare the human census of fix links: every fix link the paper's outcomes can use.

Census rule (PLAN.md, Deviations, 2026-10-29): fix links that use cue words AND share a file
with the original (strict & file_overlap), on an analysis unit, dated within 180 days of merge
(soak analysis; Y2's 30 days included) or within 90 days after the release (Y3). Reverts are
not coded: they are verified automatically (auto_validate_links.py).

Links are shuffled under neutral ids (seed 2027), so a census stopped early is a random sample.
The coder sees each PR's number, title, description, kind/area/sig/priority labels and changed
files, as the LLM coders did (prepare_llm_coding.py), with longer descriptions. No dates, lags
or late/early group.

Writes
  data/validation/human_census_key.csv  id -> orig_pr, fix_pr
  <out>                                 page data (JSON) for the coding page

Usage:
    python analysis/prepare_human_census.py OUT_JSON
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_llm_coding import load_prs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "data" / "validation"
SEED = 2027
HORIZON_DAYS = 180
ESCAPE_DAYS = 90


def census_links():
    u = pd.read_csv(ROOT / "data" / "analysis.csv.gz", usecols=["pr", "release_at"], parse_dates=["release_at"])
    lf = pd.read_csv(ROOT / "data" / "links_fixes_checked.csv", parse_dates=["event_at"])
    lf = lf[lf.strict & lf.file_overlap].merge(u, left_on="orig_pr", right_on="pr")
    after_release = (lf.event_at > lf.release_at) & (lf.event_at <= lf.release_at + pd.Timedelta(days=ESCAPE_DAYS))
    return lf[(lf.lag_days <= HORIZON_DAYS) | after_release][["orig_pr", "fix_pr"]].drop_duplicates()


def main(out):
    rows = census_links().sample(frac=1, random_state=SEED).reset_index(drop=True)
    rows.insert(0, "id", [f"H{i + 1:03d}" for i in range(len(rows))])
    key = VAL / "human_census_key.csv"
    if key.exists() and not pd.read_csv(key).equals(rows):
        sys.exit(f"{key.name} exists and differs; coding may have started. Not overwritten.")
    rows.to_csv(key, index=False)

    prs = load_prs(set(rows.orig_pr) | set(rows.fix_pr), body_chars=5000, max_files=60)
    data = {"items": [{"id": r.id, "a": int(r.orig_pr), "b": int(r.fix_pr)} for r in rows.itertuples()],
            "prs": {str(k): v for k, v in prs.items()}}
    Path(out).write_text(json.dumps(data, separators=(",", ":")))

    coded = pd.read_csv(VAL / "llm_rows_key.csv").rename(columns={"second_pr": "fix_pr"})
    overlap = rows.merge(coded, on=["orig_pr", "fix_pr"])
    print(f"{len(rows)} links, {len(prs)} PRs with text -> {key.name}, {out}; "
          f"{len(overlap)} were also LLM-coded")


if __name__ == "__main__":
    main(sys.argv[1])
