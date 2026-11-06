"""Score the human census of fix links (PLAN.md, Deviations, 2026-10-29).

Unblinds the late/early group only here, after coding finished. Precision counts "?" as not
a fix.

Inputs
  data/validation/human_census_key.csv      id -> orig_pr, fix_pr
  data/validation/human_census_answers.csv  id -> answer (y / n / ?), exported from the coding page

Writes results/validation_human_census.csv.

Usage:
    python analysis/score_human_census.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_validation import prec  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "data" / "validation"


def main():
    rows = pd.read_csv(VAL / "human_census_key.csv").merge(pd.read_csv(VAL / "human_census_answers.csv"), on="id")
    if len(rows) < len(pd.read_csv(VAL / "human_census_key.csv")) or rows.answer.isna().any():
        sys.exit("census incomplete")
    u = pd.read_csv(ROOT / "data" / "analysis.csv.gz", usecols=["pr", "final_week", "merged_at", "release_at"],
                    parse_dates=["merged_at", "release_at"])
    lf = pd.read_csv(ROOT / "data" / "links_fixes_checked.csv", parse_dates=["event_at"])
    rows = rows.merge(u, left_on="orig_pr", right_on="pr").merge(lf[["orig_pr", "fix_pr", "event_at", "lag_days"]])
    rows["group"] = np.where(rows.final_week, "final week", "weeks 2-12")
    rows["after_release"] = rows.event_at > rows.release_at

    res = [{"set": "all census links", **prec(rows.answer)}]
    for g, x in rows.groupby("group"):
        res.append({"set": f"original {g}", **prec(x.answer)})
    for a, x in rows.groupby("after_release"):
        res.append({"set": "fix after release" if a else "fix before release", **prec(x.answer)})
    for (g, a), x in rows.groupby(["group", "after_release"]):
        res.append({"set": f"original {g}, fix {'after' if a else 'before'} release", **prec(x.answer)})
    res = pd.DataFrame(res)

    res.to_csv(ROOT / "results" / "validation_human_census.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
