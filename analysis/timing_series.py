"""DESCRIPTIVE timing series for the paper (added after the results; PLAN.md, Deviations, 2026-10-31).
No estimates here: counts, shares and medians that show how the rush unfolds.

Writes results/
  timing_rush_daily.csv       PRs per release by day before freeze (days 1-28); weeks 2-12 daily mean
  timing_rush_by_release.csv  per release: PRs, final-week PRs, final week as a multiple of an average earlier week
  timing_v131_weeks.csv       v1.31 PRs by week before the actual freeze (24 Jul 2024); the original
                              deadline (10 Jul) is the end of week 3
  timing_by_bin.csv           by time before freeze (the RQ1 bins): size, kind mix, review, author
                              experience, days to release, and problem rates (all links and hand-checked)
  timing_discovery.csv        share of problems found within k days of merge, late vs early
                              (verified reverts + hand-confirmed fixes, 180 days from merge)

Usage:
    python analysis/timing_series.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_rqs as R  # noqa: E402
import soak_decomposition as S  # noqa: E402

BINS = ["weeks 9-12", "weeks 5-8", "weeks 3-4", "week 2", "days 3-7", "last 48h"]


def bin_of(u):
    """The RQ1 bins (run_rqs.rq1)."""
    return np.select([u.week_before.between(9, 12), u.week_before.between(5, 8), u.week_before.between(3, 4),
                      u.week_before == 2, u.final_week & (u.hours_to_freeze >= 48), u.hours_to_freeze < 48],
                     BINS, default="")


def rate(x):
    x = x.dropna()
    lo, hi = proportion_confint(int(x.sum()), len(x), method="wilson")
    return pd.Series({"n": len(x), "events": int(x.sum()), "pct": 100 * x.mean(), "lo": 100 * lo, "hi": 100 * hi})


def main():
    u = R.load()
    u = u[u.week_before <= 12].copy()
    n_rel = u.release.nunique()
    u["days_to_release"] = (u.release_at - u.merged_at).dt.total_seconds() / 86400

    day = (u.hours_to_freeze // 24).astype(int) + 1
    daily = (day[day <= 28].value_counts().sort_index() / n_rel).rename("prs_per_release").reset_index()
    daily.columns = ["day_before", "prs_per_release"]
    daily["weeks_2_12_daily_mean"] = (u.week_before >= 2).sum() / n_rel / (11 * 7)
    daily.to_csv(R.RES / "timing_rush_daily.csv", index=False)

    c = u.groupby("release").agg(freeze=("freeze", "first"), prs=("pr", "size"), final_week=("final_week", "sum"))
    c["final_week_share"] = c.final_week / c.prs
    c["final_week_x_avg_week"] = c.final_week / ((c.prs - c.final_week) / 11)
    c["three_per_year"] = c.index.map(lambda r: int(r.split(".")[1]) >= 22)
    c.sort_values("freeze").to_csv(R.RES / "timing_rush_by_release.csv")

    v = u[u.release == "1.31"].week_before.value_counts().sort_index().rename("prs").reset_index()
    v.columns = ["week_before_actual_freeze", "prs"]
    v.to_csv(R.RES / "timing_v131_weeks.csv", index=False)

    u["bin"] = bin_of(u)
    h = R.load("human")
    h = h[h.week_before <= 12].assign(bin=lambda x: bin_of(x))
    rows = []
    for b in BINS:
        g, gh = u[u.bin == b], h[h.bin == b]
        row = {"bin": b, "prs": len(g), "lines_median": g.lines.median(), "feature_pct": 100 * g.kind_feature.mean(),
               "bug_pct": 100 * g.kind_bug.mean(), "review_threads_mean": g.review_threads.mean(),
               "reviews_median": g.reviews.median(), "author_prior_prs_median": g.author_prior_prs.median(),
               "age_days_median": g.age_days.median(), "days_to_release_median": g.days_to_release.median()}
        for name, x in [("post_release_all", g.y3_post_release), ("post_release_checked", gh.y3_post_release),
                        ("revert30_verified", gh.y1_revert)]:
            row.update({f"{name}_{k}": val for k, val in rate(x).items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(R.RES / "timing_by_bin.csv", index=False)

    s = S.add_outcomes(R.load(), 180, "human")
    s = s[(s.week_before <= 12) & (s.p_total == 1)]
    disc = pd.DataFrame({"days": range(181)})
    for name, g in [("late", s[s.final_week]), ("early", s[~s.final_week])]:
        disc[name] = [(g.first_lag <= k).mean() for k in disc.days]
        disc[f"{name}_problems"] = len(g)
        disc[f"{name}_median_days_to_release"] = g.days_to_release.median()
    disc.to_csv(R.RES / "timing_discovery.csv", index=False)

    pd.set_option("display.width", 220)
    print(daily.round(1).to_string(index=False))
    print(c.sort_values("freeze").round(2).to_string())
    print(v.to_string(index=False))
    print(pd.DataFrame(rows).round(2).T.to_string())
    print(disc.iloc[[3, 7, 14, 21, 30, 60, 90, 180]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
