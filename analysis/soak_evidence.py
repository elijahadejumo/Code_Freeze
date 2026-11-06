"""EXPLORATORY: evidence for soak time, and further candidate explanations of the escape gap.

Soak evidence
  A. Discovery curve: how soon after merge problems surface.
  B. Gradient: escape rate by weeks before freeze, observed vs predicted by timing alone. The
     prediction for each bin uses the discovery curve of the OTHER bins, so no bin predicts
     itself. Soak predicts a smooth rise as time to release shrinks; pressure predicts a jump
     in the final week only.
  C. Rate vs timing: the composition-adjusted escape gap split into "more problems" and "a
     larger share of problems found after release".
  D. Across cycles: freeze-to-release length varies (3-7 weeks). Soak predicts a longer
     interval shrinks the late PRs' escape excess and moves problems to before release,
     with no change in the total.

Further candidate explanations
  E. Author overload: authors with several final-week PRs vs one.
  F. Hidden features: features often ship behind gates that are off by default, so soak
     cannot test them. If that drove the gap, it would appear only for kind/feature PRs.
  G. Freeze exceptions: final-week PRs that merged after the freeze vs before.

Writes results/exploratory_evidence_*_h{H}[_{links}].csv.

Usage:
    python analysis/soak_evidence.py [--horizon 180 --boot 200 --links all]
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_rqs as R  # noqa: E402
import soak_decomposition as S  # noqa: E402

BINS = [("week 1", 1, 1), ("week 2", 2, 2), ("weeks 3-4", 3, 4), ("weeks 5-6", 5, 6),
        ("weeks 7-8", 7, 8), ("weeks 9-12", 9, 12)]


def share_after(lags, days):
    """Share of problems expected after release, given discovery lags and days to release."""
    lags = np.sort(lags)
    return float(np.mean([1 - np.searchsorted(lags, t, side="right") / len(lags) for t in days]))


def discovery_curve(prob):
    return pd.DataFrame([{"within_days": k, "share_found": float((prob.first_lag <= k).mean())}
                         for k in (7, 14, 30, 60, 90, 180)])


def gradient(o):
    prob = o[o.p_total == 1]
    rows = []
    for name, lo, hi in BINS:
        inb = o.week_before.between(lo, hi)
        pin, pout = prob[prob.week_before.between(lo, hi)], prob[~prob.week_before.between(lo, hi)]
        pred = share_after(pout.first_lag.values, pin.days_to_release.values)
        rows.append({"bin": name, "n": int(inb.sum()), "median_days_to_release": o.loc[inb, "days_to_release"].median(),
                     "total_pct": 100 * o.loc[inb, "p_total"].mean(),
                     "escape_pct": 100 * o.loc[inb, "p_post"].mean(),
                     "escape_share_observed": pin.p_post.mean(), "escape_share_predicted": pred,
                     "escape_pct_predicted": 100 * o.loc[inb, "p_total"].mean() * pred})
    return pd.DataFrame(rows)


def rate_vs_timing(d, w):
    """Composition-adjusted escape gap = (rate part) + (timing part)."""
    d = R.obs(d, "p_total")
    w = w[d.index]
    late, early = d.late == 1, d.late == 0
    t_l, e_l = d.loc[late, "p_total"].mean(), d.loc[late, "p_post"].mean()
    t_e = np.average(d.loc[early, "p_total"], weights=w[early])
    e_e = np.average(d.loc[early, "p_post"], weights=w[early])
    s_l, s_e = e_l / t_l, e_e / t_e
    prob = d[d.p_total == 1]
    s_l_pred = share_after(prob.loc[prob.late == 0, "first_lag"].values,
                           prob.loc[prob.late == 1, "days_to_release"].values)
    gap = e_l - e_e
    return pd.DataFrame([{
        "late_total_pct": 100 * t_l, "early_total_pct_adjusted": 100 * t_e,
        "late_escape_share": s_l, "early_escape_share_adjusted": s_e, "late_escape_share_predicted": s_l_pred,
        "escape_gap_pp": 100 * gap,
        "rate_part_pp": 100 * (t_l - t_e) * s_l, "timing_part_pp": 100 * t_e * (s_l - s_e),
        "timing_part_predicted_pp": 100 * t_e * (s_l_pred - s_e),
        "timing_share_of_gap": t_e * (s_l - s_e) / gap}])


def lpm(df, y, terms, cluster="cycle"):
    X = pd.concat([df[terms].astype(float), R.cycle_dummies(df)], axis=1)
    return sm.OLS(df[y].astype(float), sm.add_constant(X, has_constant="add")).fit(
        cov_type="cluster", cov_kwds={"groups": df[cluster]})


def across_cycles(o):
    o = o.copy()
    o["f_weeks"] = (pd.to_datetime(o.release_at, utc=True) - pd.to_datetime(o.freeze, utc=True)).dt.days / 7
    o["late_x_f"] = o.late * (o.f_weeks - o.f_weeks.mean())
    o["late_x_kep"] = o.late * o.release.map(lambda r: int(r.split(".")[1]) >= 22)
    rows = []
    for y in ["p_total", "p_pre", "p_post"]:
        r = lpm(o, y, ["late", "late_x_f", "late_x_kep"])
        lo, hi = r.conf_int().loc["late_x_f"]
        rows.append({"outcome": S.OUTS[y], "late_effect_pp": 100 * r.params["late"],
                     "per_extra_week_freeze_to_release_pp": 100 * r.params["late_x_f"],
                     "lo_pp": 100 * lo, "hi_pp": 100 * hi, "p": r.pvalues["late_x_f"],
                     "f_weeks_range": f"{o.f_weeks.min():.1f}-{o.f_weeks.max():.1f}"})
    cyc = o.groupby("release").agg(f_weeks=("f_weeks", "first"), late_n=("late", "sum"))
    return pd.DataFrame(rows), cyc


def author_overload(o):
    late = o[o.late == 1].copy()
    late["author_final_week_prs"] = late.groupby(["release", "author"]).pr.transform("count")
    late["several"] = (late.author_final_week_prs >= 2).astype(int)
    rows = []
    for y in ["p_total", "p_post"]:
        r = lpm(late, y, ["several"] + R.COVARS)
        lo, hi = r.conf_int().loc["several"]
        rows.append({"outcome": S.OUTS[y], "share_with_several": late.several.mean(),
                     "rate_one_pct": 100 * late.loc[late.several == 0, y].mean(),
                     "rate_several_pct": 100 * late.loc[late.several == 1, y].mean(),
                     "adjusted_diff_pp": 100 * r.params["several"], "lo_pp": 100 * lo, "hi_pp": 100 * hi})
    return pd.DataFrame(rows)


def by_feature(d, reps):
    rows = []
    for label, sub in [("kind/feature", d[d.kind_feature == 1]), ("not a feature", d[d.kind_feature == 0])]:
        prob = R.obs(sub, "p_total")
        prob = prob[prob.p_total == 1]
        pred = share_after(prob.loc[prob.late == 0, "first_lag"].values,
                           prob.loc[prob.late == 1, "days_to_release"].values)
        for y in ["p_total", "p_post"]:
            naive, adj, dr = R.rq2_estimates(sub, y)
            boot = R.cluster_boot(sub, lambda b: R.rq2_estimates(b, y), reps)
            rows.append({"group": label, "outcome": S.OUTS[y], "n_late": int((sub.late == 1).sum()),
                         "naive_pp": 100 * naive, "dr_pp": 100 * dr,
                         "dr_lo": 100 * R.ci(boot[:, 2])[0], "dr_hi": 100 * R.ci(boot[:, 2])[1],
                         "late_escape_share_observed": prob.loc[prob.late == 1, "p_post"].mean(),
                         "late_escape_share_predicted": pred,
                         "early_escape_share_observed": prob.loc[prob.late == 0, "p_post"].mean()})
    return pd.DataFrame(rows)


def exceptions(o):
    late = o[o.late == 1]
    rows = []
    for label, g in [("merged before freeze", late[~late.merged_after_freeze.astype(bool)]),
                     ("merged after freeze", late[late.merged_after_freeze.astype(bool)])]:
        rows.append({"group": label, "n": len(g), "median_days_to_release": g.days_to_release.median(),
                     "total_pct": 100 * g.p_total.mean(), "escape_pct": 100 * g.p_post.mean()})
    return pd.DataFrame(rows)


def main(horizon, reps, links="all"):
    warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
    u = S.add_outcomes(R.load(), horizon, links)
    tag = f"h{horizon}" + ("" if links == "all" else f"_{links}")
    s = u[u.week_before <= 12].assign(late=lambda x: x.final_week.astype(int))
    o = R.obs(s, "p_total")
    d = u[u.final_week | u.week_before.between(5, 12)].assign(late=lambda x: x.final_week.astype(int))
    w, _ = R.att_weights(d)
    out = {"discovery": discovery_curve(o[o.p_total == 1]), "gradient": gradient(o),
           "rate_vs_timing": rate_vs_timing(d, w)}
    out["across_cycles"], cyc = across_cycles(o)
    out["author_overload"] = author_overload(o)
    out["exceptions"] = exceptions(o)
    out["by_feature"] = by_feature(d, reps)
    for k, v in out.items():
        v.to_csv(R.RES / f"exploratory_evidence_{k}_{tag}.csv", index=False)
        print(f"\n== {k}\n{v.round(4).to_string(index=False)}")
    print(f"\nfreeze-to-release weeks by cycle: {cyc.f_weeks.round(1).to_dict()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=180)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--links", default="all", choices=R.LINKS)
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    main(a.horizon, a.boot, a.links)
