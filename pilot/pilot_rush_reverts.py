"""Pilot: does the pre-freeze merge surge exist, and do rushed PRs get reverted more?

Uses only git metadata from kubernetes/kubernetes (first-parent merge commits on
master), so it runs without a GitHub token. Labels, reviews and CI data are NOT
used here; this is a feasibility check, not the causal analysis.

Usage:
    git clone --filter=tree:0 --bare --single-branch --branch master \
        https://github.com/kubernetes/kubernetes.git k8s.git
    python pilot_rush_reverts.py path/to/k8s.git
"""

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CALENDAR = HERE.parent / "data" / "k8s_freeze_calendar.csv"
OUT = HERE / "out"

MERGE_RE = re.compile(r"^Merge pull request #(\d+) from ([^/\s]+)/(\S+)")
REVERT_BRANCH_RE = re.compile(r"^revert-(\d+)-")
PR_REF_RE = re.compile(r"#(\d{3,6})")
REVERTS_COMMIT_RE = re.compile(r"This reverts commit ([0-9a-f]{40})")

REVERT_HORIZON_DAYS = 30   # outcome: reverted within 30 days of merge
WEEKS_BEFORE = 12          # analysis window: 12 weeks before each freeze
BOTS = {"k8s-ci-robot", "dependabot", "k8s-infra-cherrypick-robot", "kubernetes-prow"}


def load_merges(repo):
    fmt = "%x1e%H%x1f%cI%x1f%P%x1f%s%x1f%b"
    raw = subprocess.run(
        ["git", "-C", repo, "log", "--first-parent", f"--format={fmt}", "master"],
        capture_output=True, text=True, check=True,
    ).stdout
    rows = []
    for rec in raw.split("\x1e")[1:]:
        sha, date, parents, subject, body = rec.split("\x1f", 4)
        m = MERGE_RE.match(subject)
        if not m:
            continue
        parents = parents.split()
        rows.append({
            "sha": sha,
            "merged_at": pd.Timestamp(date).tz_convert("UTC"),
            "pr": int(m.group(1)),
            "author": m.group(2),
            "branch": m.group(3),
            "title": body.strip().splitlines()[0] if body.strip() else "",
            "pr_head": parents[1] if len(parents) > 1 else None,
        })
    return pd.DataFrame(rows)


def reverted_commits_by_pr(repo, df):
    """For revert PRs, collect 'This reverts commit X' lines from their own commits."""
    out = {}
    for r in df.itertuples():
        if r.pr_head is None:
            continue
        log = subprocess.run(
            ["git", "-C", repo, "log", "--format=%B", f"{r.sha}^1..{r.pr_head}"],
            capture_output=True, text=True,
        ).stdout
        shas = REVERTS_COMMIT_RE.findall(log)
        if shas:
            out[r.pr] = shas
    return out


def link_reverts(repo, df):
    """Return {original_pr: revert_merged_at} using four linking heuristics."""
    by_sha = dict(zip(df.sha, df.pr))
    by_title = {}
    for r in df.sort_values("merged_at").itertuples():
        by_title.setdefault(r.title.strip().lower(), r.pr)

    is_rev = df.title.str.match(r"(?i)^\s*revert\b") | df.branch.str.match(r"^revert-\d+-")
    revs = df[is_rev]
    commits = reverted_commits_by_pr(repo, revs)

    links = []
    for r in revs.itertuples():
        targets = set()
        m = REVERT_BRANCH_RE.match(r.branch)
        if m:
            targets.add(int(m.group(1)))                                  # GitHub "Revert" button
        targets.update(int(x) for x in PR_REF_RE.findall(r.title))        # 'Revert #123'
        for s in commits.get(r.pr, []):                                   # git revert of the merge commit
            if s in by_sha:
                targets.add(by_sha[s])
        inner = re.match(r'(?i)^\s*revert\s+"(.+)"\s*$', r.title)         # 'Revert "<original title>"'
        if inner and inner.group(1).strip().lower() in by_title:
            targets.add(by_title[inner.group(1).strip().lower()])
        for t in targets:
            if t != r.pr:
                links.append({"orig_pr": t, "revert_pr": r.pr, "revert_merged_at": r.merged_at})
    return pd.DataFrame(links), revs


def main(repo):
    OUT.mkdir(exist_ok=True)
    cal = pd.read_csv(CALENDAR, dtype={"release": str})
    cal["freeze"] = pd.to_datetime(cal.code_freeze_utc, utc=True)
    now = pd.Timestamp.now(tz="UTC")
    cal = cal[cal.freeze < now - pd.Timedelta(days=REVERT_HORIZON_DAYS)]

    df = load_merges(repo)
    links, revs = link_reverts(repo, df)
    df = df.merge(links, how="left", left_on="pr", right_on="orig_pr")
    df["lag_days"] = (df.revert_merged_at - df.merged_at).dt.total_seconds() / 86400
    df["reverted30"] = df.lag_days.between(0, REVERT_HORIZON_DAYS)
    df = (df.sort_values("reverted30", ascending=False)
            .drop_duplicates("pr")[["pr", "sha", "merged_at", "author", "branch", "title",
                                    "reverted30", "revert_pr", "lag_days"]])
    df["is_revert"] = df.pr.isin(revs.pr)
    df["is_bot"] = df.author.isin(BOTS)

    # Assign each merge to the next upcoming freeze within the window.
    rows = []
    for c in cal.itertuples():
        lo = c.freeze - pd.Timedelta(weeks=WEEKS_BEFORE)
        w = df[(df.merged_at >= lo) & (df.merged_at < c.freeze)].copy()
        w["release"] = c.release
        w["days_to_freeze"] = (c.freeze - w.merged_at).dt.total_seconds() / 86400
        w["week_before"] = (w.days_to_freeze // 7).astype(int) + 1    # 1 = final 7 days
        rows.append(w)
    win = pd.concat(rows)
    # A PR can fall in two windows only if cycles overlap; keep its nearest freeze.
    win = win.sort_values("days_to_freeze").drop_duplicates("pr")
    win.to_csv(OUT / "merges_in_window.csv", index=False)

    base = win[~win.is_revert & ~win.is_bot]
    weekly = (base.groupby(["release", "week_before"])
                  .agg(merged=("pr", "size"), reverted=("reverted30", "sum")).reset_index())
    weekly.to_csv(OUT / "weekly_by_release.csv", index=False)

    pooled = (base.groupby("week_before")
                  .agg(merged=("pr", "size"), reverted=("reverted30", "sum"),
                       releases=("release", "nunique")).reset_index())
    pooled["merged_per_release"] = pooled.merged / pooled.releases
    pooled["revert_rate_pct"] = 100 * pooled.reverted / pooled.merged
    pooled.to_csv(OUT / "pooled_by_week.csv", index=False)

    daily_last = base[base.days_to_freeze < 14].copy()
    daily_last["day_before"] = daily_last.days_to_freeze.astype(int) + 1
    daily = daily_last.groupby("day_before").agg(merged=("pr", "size"),
                                                 reverted=("reverted30", "sum")).reset_index()
    daily.to_csv(OUT / "pooled_last14_days.csv", index=False)

    # Same-author comparison: authors with merges both in the final week and in weeks 5-12.
    late = base[base.week_before == 1]
    early = base[base.week_before >= 5]
    both = set(late.author) & set(early.author)
    a_late = late[late.author.isin(both)]
    a_early = early[early.author.isin(both)]

    with open(OUT / "summary.txt", "w") as f:
        p = lambda *a: print(*a, file=f)
        p(f"merges on master (all time): {len(df)}; revert PRs: {df.is_revert.sum()}; "
          f"linked revert->original pairs: {len(links)}")
        p(f"releases analysed: {cal.release.min()}..{cal.release.max()} ({len(cal)})")
        p(f"non-bot, non-revert PRs in 12-week pre-freeze windows: {len(base)}; "
          f"reverted within {REVERT_HORIZON_DAYS}d: {int(base.reverted30.sum())}")
        p("\npooled by week before freeze (1 = final 7 days):")
        p(pooled.to_string(index=False))
        p("\nlast 14 days, pooled:")
        p(daily.to_string(index=False))
        p(f"\nauthors merging both in final week and in weeks 5-12: {len(both)}")
        p(f"  their final-week PRs: {len(a_late)}, reverted {int(a_late.reverted30.sum())} "
          f"({100*a_late.reverted30.mean():.2f}%)")
        p(f"  their week 5-12 PRs: {len(a_early)}, reverted {int(a_early.reverted30.sum())} "
          f"({100*a_early.reverted30.mean():.2f}%)")
        share = (weekly[weekly.week_before == 1].set_index("release").merged
                 / weekly.groupby("release").merged.sum())
        p("\nshare of each cycle's 12-week merges landing in the final week:")
        p(share.round(3).to_string())
    print((OUT / "summary.txt").read_text())


if __name__ == "__main__":
    main(sys.argv[1])
