"""Draw the manual validation samples for outcome links.

Pre-specified (PLAN.md): 50 revert and 50 fix links within 30 days (the Y1/Y2 outcomes).
Added 2026-10-29: 60 links to problems found after release within 180 days of merge (the
soak analysis), 30 for final-week PRs and 30 for earlier PRs, so precision can be compared
between the groups being contrasted.

Writes data/validation/{revert,fix,escape}_links_sample.csv with PR URLs and empty coding
columns. Code each row: correct = y/n, plus a note. Two independent coders allow kappa.
A file that already exists is left untouched, so coding in progress is never overwritten.

Usage:
    python analysis/sample_validation.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "validation"
URL = "https://github.com/kubernetes/kubernetes/pull/{}"
N = 50
N_ESCAPE = 30          # per group (late, early)
SEED = 2027
QUESTIONS = {"revert": "Does the second PR revert the first?",
             "fix": "Does the second PR fix a defect introduced by the first?"}


def finish(s, name, path):
    s = s.copy()
    s["original_url"] = s.orig_pr.map(URL.format)
    s["second_url"] = s.second_pr.map(URL.format)
    s["question"] = s.link_type.map(QUESTIONS)
    s["coder1_correct"] = ""
    s["coder2_correct"] = ""
    s["note"] = ""
    s.to_csv(path, index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    u = pd.read_csv(ROOT / "data" / "analysis.csv.gz", usecols=["pr", "release_at", "final_week", "week_before"])
    u["release_at"] = pd.to_datetime(u.release_at, utc=True)
    rev = pd.read_csv(ROOT / "data" / "links_reverts.csv").rename(columns={"revert_pr": "second_pr"})
    fix = pd.read_csv(ROOT / "data" / "links_fixes.csv").rename(columns={"fix_pr": "second_pr"})
    rev["link_type"], fix["link_type"] = "revert", "fix"

    taken = set()
    for name, links in [("revert", rev), ("fix", fix)]:
        path = OUT / f"{name}_links_sample.csv"
        links = links[links.orig_pr.isin(u.pr) & (links.lag_days <= 30)]
        s = links.sample(min(N, len(links)), random_state=SEED)
        taken |= set(zip(s.orig_pr, s.second_pr))
        if path.exists():
            print(f"{name}: {path.name} exists, left untouched")
            continue
        finish(s, name, path)
        print(f"{name}: sampled {len(s)} of {len(links)} links")

    path = OUT / "escape_links_sample.csv"
    if path.exists():
        print(f"escape: {path.name} exists, left untouched")
        return
    links = pd.concat([rev, fix]).merge(u, left_on="orig_pr", right_on="pr")
    links["event_at"] = pd.to_datetime(links.event_at, utc=True)
    links = links[(links.week_before <= 12) & (links.lag_days <= 180) & (links.event_at > links.release_at)]
    links = links[[(a, b) not in taken for a, b in zip(links.orig_pr, links.second_pr)]]
    links["group"] = links.final_week.map({True: "final week", False: "weeks 2-12"})
    s = pd.concat([g.sample(min(N_ESCAPE, len(g)), random_state=SEED) for _, g in links.groupby("group")])
    s = s.sample(frac=1, random_state=SEED)          # mix groups so coders can't infer them from order
    # coders stay blind to late vs early: the group goes to a separate key file
    s[["orig_pr", "second_pr", "group"]].to_csv(OUT / "escape_links_key.csv", index=False)
    finish(s.drop(columns=["pr", "release_at", "final_week", "week_before", "group"]), "escape", path)
    print(f"escape: sampled {len(s)} of {len(links)} post-release links ({s.group.value_counts().to_dict()})")


if __name__ == "__main__":
    main()
