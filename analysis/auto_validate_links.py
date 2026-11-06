"""Automatic checks on outcome links, from the files each PR changed (crawl data).

Reverts: a true revert touches the original's files with additions and deletions swapped.
    inverse = "exact" (every file of the original, swapped), "partial" (some files swapped),
    "shared" (shared files, nothing swapped), "none" (no shared files), or "no data".
Fixes: file_overlap = the fix touches at least one file the original touched. Not a proof
    either way (a true fix may only touch a test; a false link may share a busy file), but a
    proxy that should not differ between late and early PRs.

Writes data/links_reverts_checked.csv and data/links_fixes_checked.csv. The soak analysis
uses them for its robustness variants (soak_decomposition.py --links).

Usage:
    python analysis/auto_validate_links.py
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def load_files():
    files = {}
    for path in sorted(RAW.glob("prs_graphql*.jsonl")):
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                files[d["number"]] = {x["path"]: (x["additions"], x["deletions"]) for x in d["files"]["nodes"]}
    return files


def inverse(fa, fb):
    if not fa or not fb:
        return "no data"
    common = set(fa) & set(fb)
    if not common:
        return "none"
    swapped = sum(fa[p] == fb[p][::-1] for p in common)
    if swapped == len(common) == len(fa):
        return "exact"
    return "partial" if swapped else "shared"


def main():
    files = load_files()
    lr = pd.read_csv(ROOT / "data" / "links_reverts.csv")
    lr["inverse"] = [inverse(files.get(a), files.get(b)) for a, b in zip(lr.orig_pr, lr.revert_pr)]
    lr["verified"] = lr.inverse.isin(["exact", "partial"])
    lr.to_csv(ROOT / "data" / "links_reverts_checked.csv", index=False)

    lf = pd.read_csv(ROOT / "data" / "links_fixes.csv")
    lf["file_overlap"] = [bool(set(files.get(a, {})) & set(files.get(b, {}))) for a, b in zip(lf.orig_pr, lf.fix_pr)]
    lf.to_csv(ROOT / "data" / "links_fixes_checked.csv", index=False)

    print(f"reverts: {len(lr)} links | {lr.inverse.value_counts().to_dict()} | verified {lr.verified.mean():.1%}")
    print(f"fixes:   {len(lf)} links | file overlap {lf.file_overlap.mean():.1%} | strict {lf.strict.mean():.1%} "
          f"| both {(lf.file_overlap & lf.strict).mean():.1%}")


if __name__ == "__main__":
    main()
