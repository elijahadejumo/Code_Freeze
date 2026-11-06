"""EXPLORATORY (post hoc, not in the pre-specified plan): is the Y3 gap quality or soak time?

Y3 counts problems in (release, release + 90d]. A final-week PR reaches the release about 1
month after merge, an early PR about 2 months after, so the same defect is more likely to
surface after release for the late PR. Here every PR gets the same window from MERGE
(H days), and its first problem is split into found-before vs found-after release:

    total = pre + post.    More problems in total -> quality.    Same total, more post -> soak.

The RQ2 estimators (naive, ATT-weighted, doubly robust; cycle bootstrap) and the RQ3 author
fixed effects are applied unchanged. The last block checks the soak prediction directly:
the share of late PRs' problems found after release, if late PRs had the early PRs' timing
of problem discovery (lag since merge).

Writes results/exploratory_soak_h{H}[_{links}].csv and results/exploratory_soak_timing_h{H}[_{links}].csv.

Usage:
    python analysis/soak_decomposition.py [--horizon 180 --boot 400 --links all]
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_rqs as R  # noqa: E402

OUTS = {"p_total": "any problem", "p_pre": "found before release", "p_post": "found after release"}


def add_outcomes(u, horizon, links="all"):
    if links == "human" and horizon > 180:
        raise SystemExit("the human census covers links within 180 days of merge only")
    lr, lf = R.load_links(links)
    ev = pd.concat([lr, lf])[["orig_pr", "event_at", "lag_days"]]
    ev["event_at"] = pd.to_datetime(ev.event_at, utc=True)
    ev = ev[ev.lag_days <= horizon]
    first_at = ev.groupby("orig_pr").event_at.min()
    first_lag = ev.groupby("orig_pr").lag_days.min()
    closed = pd.to_datetime(u.merged_at, utc=True) + pd.Timedelta(days=horizon) < pd.Timestamp.now(tz="UTC")
    hit = u.pr.isin(first_at.index)
    post = u.pr.map(first_at) > pd.to_datetime(u.release_at, utc=True)
    u["p_total"] = np.where(closed, hit, np.nan)
    u["p_pre"] = np.where(closed, hit & ~post, np.nan)
    u["p_post"] = np.where(closed, hit & post, np.nan)
    u["first_lag"] = u.pr.map(first_lag)
    u["days_to_release"] = (pd.to_datetime(u.release_at, utc=True)
                            - pd.to_datetime(u.merged_at, utc=True)).dt.total_seconds() / 86400
    return u


def main(horizon, reps, links="all"):
    warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
    u = add_outcomes(R.load(), horizon, links)
    tag = f"h{horizon}" + ("" if links == "all" else f"_{links}")
    s = u[u.week_before <= 12].assign(late=lambda x: x.final_week.astype(int))
    d = u[u.final_week | u.week_before.between(5, 12)].assign(late=lambda x: x.final_week.astype(int))
    w, _ = R.att_weights(d)
    rows = []
    for y, name in OUTS.items():
        rr, lo, hi = R.rr_poisson(s, "late", y)
        naive, adj, dr = R.rq2_estimates(d, y)
        boot = R.cluster_boot(d, lambda b: R.rq2_estimates(b, y), reps)
        b, se, _, _ = R.rq3_author_fe(d, w, y)
        o = R.obs(s, y)
        rows.append({"outcome": name, "horizon_days": horizon, "links": links,
                     "late_pct": 100 * o.loc[o.late == 1, y].mean(), "early_pct": 100 * o.loc[o.late == 0, y].mean(),
                     "rr": rr, "rr_lo": lo, "rr_hi": hi,
                     "naive_pp": 100 * naive, "naive_ci": tuple(100 * np.array(R.ci(boot[:, 0]))),
                     "adjusted_pp": 100 * adj, "adjusted_ci": tuple(100 * np.array(R.ci(boot[:, 1]))),
                     "dr_pp": 100 * dr, "dr_ci": tuple(100 * np.array(R.ci(boot[:, 2]))),
                     "authorFE_pp": 100 * b, "authorFE_ci": (100 * (b - 1.96 * se), 100 * (b + 1.96 * se))})
    out = pd.DataFrame(rows)
    out.to_csv(R.RES / f"exploratory_soak_{tag}.csv", index=False)

    # Soak prediction: apply the early PRs' discovery-lag distribution to each group's
    # merge-to-release gap, and compare with the observed share found after release.
    o = R.obs(s, "p_total")
    prob = o[o.p_total == 1]
    early_lags = np.sort(prob.loc[prob.late == 0, "first_lag"].values)
    timing = []
    for late, g in prob.groupby("late"):
        pred = np.mean([np.mean(early_lags > t) for t in g.days_to_release])
        timing.append({"group": "final week" if late else "weeks 2-12", "problems": len(g),
                       "median_days_merge_to_release": g.days_to_release.median(),
                       "observed_share_after_release": g.p_post.mean(),
                       "predicted_share_if_early_timing": pred})
    timing = pd.DataFrame(timing)
    timing.to_csv(R.RES / f"exploratory_soak_timing_{tag}.csv", index=False)
    return out, timing


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=180)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--links", default="all", choices=R.LINKS)
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    out, timing = main(a.horizon, a.boot, a.links)
    print(out.round(3).to_string(index=False))
    print(timing.round(3).to_string(index=False))
