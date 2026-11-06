"""Score link validation: LLM-coded fix links, the human check, and automatic revert checks.

Inputs
  CODED_DIR/coded_*.csv               LLM verdicts (id, verdict, confidence, reason); copied to
                                      data/validation/llm_coded.csv for provenance
  data/validation/llm_rows_key.csv    id -> sample, orig_pr, second_pr
  data/validation/escape_links_key.csv  late vs early group (unblinded only here)
  data/validation/human_check_20.csv  your_answer = y / n / ? (optional until filled in)
  data/links_reverts_checked.csv      automatic revert verification

Precision counts "?" as not correct (conservative); the share of "?" is reported.

Usage:
    python analysis/score_validation.py [CODED_DIR]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "data" / "validation"


def prec(v):
    k, n = int((v == "y").sum()), len(v)
    lo, hi = proportion_confint(k, n, method="wilson") if n else (np.nan, np.nan)
    return {"n": n, "correct": k, "unsure": int((v == "?").sum()), "precision": k / n if n else np.nan,
            "lo": lo, "hi": hi}


def kappa(a, b):
    cats = sorted(set(a) | set(b))
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def main(coded_dir=None):
    out = VAL / "llm_coded.csv"
    if coded_dir:
        coded = pd.concat([pd.read_csv(p) for p in sorted(Path(coded_dir).glob("coded_*.csv"))])
        coded.to_csv(out, index=False)
    coded = pd.read_csv(out)
    coded["verdict"] = coded.verdict.astype(str).str.strip().str.lower()
    key = pd.read_csv(VAL / "llm_rows_key.csv")
    rows = key.merge(coded, on="id", how="left")
    missing = rows.verdict.isna().sum()
    group = pd.read_csv(VAL / "escape_links_key.csv")
    rows = rows.merge(group, on=["orig_pr", "second_pr"], how="left")

    res = [{"set": "all fix links coded", **prec(rows.verdict)}]
    for s, g in rows.groupby("sample"):
        res.append({"set": f"sample: {s}", **prec(g.verdict)})
    for gname, g in rows[rows["sample"] == "escape"].groupby("group"):
        res.append({"set": f"post-release links, {gname}", **prec(g.verdict)})

    rv = pd.read_csv(ROOT / "data" / "links_reverts_checked.csv")
    res.append({"set": "reverts (automatic inverse check)", **prec(rv.verified.map({True: "y", False: "n"}))})
    res = pd.DataFrame(res)

    hum = pd.read_csv(VAL / "human_check_20.csv", dtype=str)
    hum["your_answer"] = hum.your_answer.fillna("").str.strip().str.lower()
    done = hum[hum.your_answer.isin(["y", "n", "?"])].merge(coded[["id", "verdict"]], on="id")
    agree = None
    if len(done):
        agree = {"human_rows": len(done), "agreement": float(np.mean(done.your_answer == done.verdict)),
                 "kappa": kappa(done.your_answer.values, done.verdict.values),
                 "human_precision": float(np.mean(done.your_answer == "y"))}

    res.to_csv(ROOT / "results" / "validation_precision.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print(f"\nLLM verdicts missing for {missing} rows; confidence: {coded.confidence.value_counts().to_dict()}")
    print("human check:", agree if agree else "not filled in yet")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
