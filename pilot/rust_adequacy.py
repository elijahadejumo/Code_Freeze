"""Data-adequacy checks for adding Rust (decision rule in README, fixed 2026-10-28).

Computes ONLY the three pre-registered checks. It does not compute rush or revert
gradients, so the inclusion decision cannot depend on the direction of results.

  1. beta-cut dates recoverable for >= 20 cycles (version-bump PRs on main,
     cross-checked against release dates in RELEASES.md)
  2. >= 80% of revert PRs linkable to the PR they revert
  3. >= 100 linked reverts inside the pre-cut windows

Usage:
    git clone --filter=tree:0 --bare --single-branch --branch main \
        https://github.com/rust-lang/rust.git rust.git
    python rust_adequacy.py path/to/rust.git
"""

import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "out"
AUTO_RE = re.compile(r"^Auto merge of #(\d+) - ([^:\s]+):(\S+?),? r=")
ROLLUP_MEMBER_RE = re.compile(r"^Rollup merge of #(\d+) - ([^:\s]+):(\S+?),? r=")
BUMP_RE = re.compile(r"(?i)\bbump\b.*\b(?:version|to)\b.*?\b1\.(\d{1,3})(?:\.0)?\b(?![.\d])")
RELEASE_RE = re.compile(r"^Version 1\.(\d+)\.0 \((\d{4}-\d{2}-\d{2})\)", re.M)
REVERT_BRANCH_RE = re.compile(r"^revert-(\d+)")
PR_REF_RE = re.compile(r"#(\d{3,6})")
REVERTS_COMMIT_RE = re.compile(r"This reverts commit ([0-9a-f]{40})")


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout


def first_line(body):
    for line in body.splitlines():
        if line.strip():
            return line.strip()
    return ""


def load_prs(repo):
    """Every PR merged on main: direct bors merges plus members of rollups."""
    fmt = "%x1e%H%x1f%cI%x1f%P%x1f%s%x1f%b"
    rows, rollups = [], []
    for rec in git(repo, "log", "--first-parent", "--since=2015-05-01", f"--format={fmt}", "main").split("\x1e")[1:]:
        sha, date, parents, subject, body = rec.split("\x1f", 4)
        m = AUTO_RE.match(subject)
        if not m:
            continue
        parents = parents.split()
        title = first_line(body)
        rows.append({"sha": sha, "merged_at": date, "pr": int(m.group(1)), "author": m.group(2),
                     "branch": m.group(3), "title": title, "p1": parents[0],
                     "p2": parents[1] if len(parents) > 1 else None, "via_rollup": False})
        if title.startswith("Rollup of") and len(parents) > 1:
            rollups.append((sha, date, parents[0], parents[1]))
    for sha, date, p1, p2 in rollups:
        for rec in git(repo, "log", f"--format={fmt}", f"{p1}..{p2}").split("\x1e")[1:]:
            msha, _, mparents, subject, body = rec.split("\x1f", 4)
            m = ROLLUP_MEMBER_RE.match(subject)
            if not m:
                continue
            mparents = mparents.split()
            rows.append({"sha": msha, "merged_at": date, "pr": int(m.group(1)), "author": m.group(2),
                         "branch": m.group(3), "title": first_line(body), "p1": mparents[0],
                         "p2": mparents[1] if len(mparents) > 1 else None, "via_rollup": True})
    df = pd.DataFrame(rows)
    df["merged_at"] = pd.to_datetime(df.merged_at, utc=True)
    # A PR can appear as a rollup member and again as a direct merge after a failed rollup; keep the last.
    return df.sort_values("merged_at").drop_duplicates("pr", keep="last").reset_index(drop=True)


def check_cutoffs(df):
    """Beta cut for 1.N ~ the PR on main that bumps the version to 1.(N+1)."""
    bumps = df[~df.via_rollup & df.title.str.contains(r"(?i)\bbump", regex=True)].copy()
    bumps["to_minor"] = bumps.title.map(lambda t: int(m.group(1)) if (m := BUMP_RE.search(t)) else None)
    bumps = bumps.dropna(subset=["to_minor"])
    bumps = bumps[~bumps.title.str.contains(r"(?i)cargo|stage0|bootstrap|cc@|libgit|llvm|crate", regex=True)]
    bumps = bumps.sort_values("merged_at").drop_duplicates("to_minor", keep="first")
    bumps["cut_for"] = bumps.to_minor.astype(int) - 1

    text = urllib.request.urlopen("https://raw.githubusercontent.com/rust-lang/rust/main/RELEASES.md").read().decode()
    rel = pd.DataFrame([(int(n), d) for n, d in RELEASE_RE.findall(text)], columns=["minor", "release_date"])
    rel["release_date"] = pd.to_datetime(rel.release_date, utc=True)
    cuts = bumps.merge(rel, left_on="cut_for", right_on="minor", how="left")
    # Beta for 1.N is promoted ~6 weeks before 1.N ships; accept cuts within 0-9 weeks of that release.
    cuts["days_before_release"] = (cuts.release_date - cuts.merged_at).dt.total_seconds() / 86400
    cuts["consistent"] = cuts.days_before_release.between(28, 63)
    return cuts[["cut_for", "sha", "merged_at", "pr", "title", "release_date", "days_before_release", "consistent"]]


def check_reverts(repo, df):
    by_sha = dict(zip(df.sha, df.pr))
    by_title = {}
    for r in df.itertuples():
        by_title.setdefault(r.title.lower(), r.pr)
    known = set(df.pr)
    is_rev = df.title.str.match(r"(?i)^\s*revert\b") | df.branch.str.match(r"^revert-\d+")
    revs = df[is_rev]
    links = []
    for r in revs.itertuples():
        targets = set()
        if m := REVERT_BRANCH_RE.match(r.branch):
            targets.add(int(m.group(1)))
        targets.update(int(x) for x in PR_REF_RE.findall(r.title))
        if r.p2:
            for s in REVERTS_COMMIT_RE.findall(git(repo, "log", "--format=%B", f"{r.p1}..{r.p2}")):
                if s in by_sha:
                    targets.add(by_sha[s])
        if (inner := re.match(r'(?i)^\s*revert\s+"(.+)"\s*$', r.title)) and inner.group(1).lower() in by_title:
            targets.add(by_title[inner.group(1).lower()])
        targets = {t for t in targets if t in known and t != r.pr}
        links.append({"revert_pr": r.pr, "revert_at": r.merged_at, "linked": bool(targets),
                      "targets": sorted(targets)})
    return pd.DataFrame(links)


def main(repo):
    OUT.mkdir(exist_ok=True)
    df = load_prs(repo)
    cuts = check_cutoffs(df)
    good = cuts[cuts.consistent]
    links = check_reverts(repo, df)

    # Check 3 counts linked reverts whose reverted PR merged within a 6-week pre-cut window.
    edges = good.merged_at.sort_values().tolist()
    lo, hi = edges[0] - pd.Timedelta(weeks=6), edges[-1]
    orig_at = dict(zip(df.pr, df.merged_at))
    in_win = 0
    for r in links[links.linked].itertuples():
        t = orig_at[r.targets[0]]
        if lo <= t < hi and any(e - pd.Timedelta(weeks=6) <= t < e for e in edges):
            in_win += 1

    c1, c2, c3 = len(good), links.linked.mean(), in_win
    lines = [
        f"PRs merged on main since 1.0: {len(df)} ({int(df.via_rollup.sum())} via rollups)",
        f"[1] beta cuts recovered: {len(cuts)}; consistent with RELEASES.md: {c1} "
        f"(1.{int(good.cut_for.min())}..1.{int(good.cut_for.max())})  -> {'PASS' if c1 >= 20 else 'FAIL'} (need >= 20)",
        f"[2] revert PRs: {len(links)}; linked to a known PR: {int(links.linked.sum())} ({100 * c2:.1f}%)"
        f"  -> {'PASS' if c2 >= 0.8 else 'FAIL'} (need >= 80%)",
        f"[3] linked reverts of PRs merged inside pre-cut windows: {c3}  -> {'PASS' if c3 >= 100 else 'FAIL'} (need >= 100)",
        f"inconsistent cut candidates (excluded): {(~cuts.consistent).sum()}",
    ]
    text = "\n".join(lines)
    print(text)
    (OUT / "rust_adequacy.txt").write_text(text + "\n")
    cuts.to_csv(OUT / "rust_beta_cuts.csv", index=False)
    links[~links.linked].merge(df[["pr", "title", "branch"]], left_on="revert_pr", right_on="pr") \
        .to_csv(OUT / "rust_unlinked_reverts.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
