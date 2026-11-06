"""Run RQ1-RQ5 exactly as specified in PLAN.md; write results/ and a summary table.

--links other than "all" rebuilds Y1-Y3 from a stricter link set (robustness rows, PLAN.md
Deviations 2026-10-29) and writes to results/links_<links>/.

Usage:
    python analysis/run_rqs.py [--boot 400] [--links all]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUTCOMES = {"y1_revert": "Y1 revert", "y2_fix": "Y2 follow-up fix", "y3_post_release": "Y3 post-release problem"}
KINDS = ["feature", "bug", "cleanup", "api-change", "documentation", "regression", "failing-test",
         "flake", "deprecation", "none"]
MEDIATORS = ["m_review_threads", "m_reviews", "m_lgtm_removed", "m_last_commit_to_ready"]
RNG = np.random.default_rng(2027)
LINKS = ["all", "overlap", "strict", "strict_overlap", "human"]
HORIZON_DAYS, ESCAPE_DAYS = 30, 90      # as in build_dataset.py


# ---------------------------------------------------------------- data

def load_links(links="all"):
    """Outcome links. Variants other than "all" keep only links that pass the automatic
    checks (analysis/auto_validate_links.py): verified reverts, plus fixes that share a file
    with the original ("overlap"), use cue words ("strict"), or both ("strict_overlap").
    "human" keeps the strict_overlap fixes the human census coded "y" ("?" is not a fix);
    the census covers only links within 180 days of merge or 90 days after the release."""
    if links == "all":
        return pd.read_csv(ROOT / "data" / "links_reverts.csv"), pd.read_csv(ROOT / "data" / "links_fixes.csv")
    lr = pd.read_csv(ROOT / "data" / "links_reverts_checked.csv")
    lf = pd.read_csv(ROOT / "data" / "links_fixes_checked.csv")
    if links == "human":
        key = pd.read_csv(ROOT / "data" / "validation" / "human_census_key.csv")
        ans = pd.read_csv(ROOT / "data" / "validation" / "human_census_answers.csv").set_index("id").answer
        return lr[lr.verified], lf.merge(key[key.id.map(ans) == "y"][["orig_pr", "fix_pr"]])
    keep = {"overlap": lf.file_overlap, "strict": lf.strict, "strict_overlap": lf.file_overlap & lf.strict}[links]
    return lr[lr.verified], lf[keep]


def outcomes_from_links(u, lr, lf):
    """Rebuild Y1-Y3 exactly as build_dataset.py does, from another set of links."""
    lr, lf = (x.assign(event_at=pd.to_datetime(x.event_at, utc=True)) for x in (lr, lf))
    rev_at, fix_at = (x[x.lag_days <= HORIZON_DAYS].groupby("orig_pr").event_at.min() for x in (lr, lf))
    u["y1_revert"] = u.pr.isin(rev_at.index).astype(int)
    u["y2_fix"] = u.pr.isin(fix_at.index).astype(int)
    u["first_problem_at"] = u.pr.map(pd.concat([rev_at, fix_at], axis=1).min(axis=1))
    u["any_problem"] = u.y1_revert | u.y2_fix
    u["problem_in_freeze"] = u.first_problem_at.between(u.freeze, u.release_at)
    ev = pd.concat([lr, lf])[["orig_pr", "event_at"]].merge(u[["pr", "release_at"]], left_on="orig_pr", right_on="pr")
    hit = ev[(ev.event_at > ev.release_at) & (ev.event_at <= ev.release_at + pd.Timedelta(days=ESCAPE_DAYS))]
    u["y3_post_release"] = np.where(u.y3_post_release.isna(), np.nan, u.pr.isin(hit.orig_pr).astype(float))
    return u


def load(links="all"):
    u = pd.read_csv(ROOT / "data" / "analysis.csv.gz", dtype={"release": str},
                    parse_dates=["ready_at", "merged_at", "freeze", "release_at", "created_at", "first_problem_at"])
    for c in ["y1_revert", "y2_fix", "y2_fix_strict", "any_problem"]:
        u[c] = u[c].astype(bool).astype(int)
    u["y3_post_release"] = u.y3_post_release.astype(float)   # NaN = window still open
    u["size_missing"] = u["size"].isna().astype(int)
    u["size"] = u["size"].fillna(u["size"].median())
    u["log_lines"] = np.log1p(u.lines.clip(lower=0))
    u["log_files"] = np.log1p(u.files.clip(lower=0))
    u["log_age"] = np.log1p(u.age_days.clip(lower=0))
    u["log_exp"] = np.log1p(u.author_prior_prs)
    u["m_review_threads"] = np.log1p(u.review_threads)
    u["m_reviews"] = np.log1p(u.reviews)
    u["m_lgtm_removed"] = (u.lgtm_removed > 0).astype(int)
    u["m_last_commit_to_ready"] = np.log1p(u.last_commit_to_ready_h.clip(lower=0).fillna(0))
    u["log_queue_wait"] = np.log1p(u.queue_wait_h.clip(lower=0))
    u["log_conc_area"] = np.log1p(u.conc_area_12h)
    u["log_conc_total"] = np.log1p(u.conc_total_6h)
    for k in KINDS:
        u[f"kind_{k}"] = u[f"kind_{k}"].astype(bool).astype(int)
    for c in ["has_milestone", "priority_critical"]:
        u[c] = u[c].astype(bool).astype(int)
    u["cycle"] = u.release.astype("category").cat.codes
    u["author_id"] = u.author.astype("category").cat.codes
    return u if links == "all" else outcomes_from_links(u, *load_links(links))


COVARS = (["size", "size_missing", "n_sigs", "has_milestone", "priority_critical", "log_lines",
           "log_files", "log_age", "log_exp"] + [f"kind_{k}" for k in KINDS if k != "none"])


def cycle_dummies(df):
    return pd.get_dummies(df.release, prefix="c", drop_first=True, dtype=float)


def design(df, extra=()):
    X = df[list(COVARS) + list(extra)].astype(float)
    return pd.concat([X, cycle_dummies(df)], axis=1)


def ols(y, X, groups, w=None):
    X = sm.add_constant(X, has_constant="add")
    m = sm.WLS(y, X, weights=w) if w is not None else sm.OLS(y, X)
    return m.fit(cov_type="cluster", cov_kwds={"groups": groups})


def cluster_boot(df, fn, reps):
    cycles = df.release.unique()
    out, failed = [], 0
    for _ in range(reps):
        pick = RNG.choice(cycles, len(cycles), replace=True)
        # fresh index: a cycle drawn twice would otherwise duplicate row labels
        b = pd.concat([df[df.release == c].assign(release=f"{c}#{i}") for i, c in enumerate(pick)],
                      ignore_index=True)
        try:
            out.append(fn(b))
        except Exception:
            failed += 1
    if failed:
        print(f"  cluster_boot: {failed}/{reps} replicates failed", flush=True)
    return np.array(out)


def ci(a):
    a = a[np.isfinite(a)]
    return (np.percentile(a, 2.5), np.percentile(a, 97.5)) if len(a) else (np.nan, np.nan)


# ---------------------------------------------------------------- RQ1

def obs(df, y):
    return df[df[y].notna()]


def rr_poisson(df, treat, y):
    df = obs(df, y)
    if df.loc[df[treat] == 1, y].sum() == 0 or df.loc[df[treat] == 0, y].sum() == 0:
        return np.nan, np.nan, np.nan
    X = sm.add_constant(df[[treat]].astype(float))
    r = sm.GLM(df[y], X, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": df.cycle})
    b, se = r.params[treat], r.bse[treat]
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)


def rq1(u):
    bins = [("weeks 9-12", u.week_before.between(9, 12)), ("weeks 5-8", u.week_before.between(5, 8)),
            ("weeks 3-4", u.week_before.between(3, 4)), ("week 2", u.week_before == 2),
            ("days 3-7", u.final_week & (u.hours_to_freeze >= 48)), ("last 48h", u.hours_to_freeze < 48)]
    rows = []
    for lab, m in bins:
        row = {"bin": lab, "n": int(m.sum())}
        for y in OUTCOMES:
            row[f"{y}_n"] = int(np.nansum(u.loc[m, y]))
            row[f"{y}_pct"] = 100 * u.loc[m, y].mean()
        rows.append(row)
    pd.DataFrame(rows).to_csv(RES / "rq1_bins.csv", index=False)

    s = u[u.week_before <= 12].assign(late=u.final_week.astype(int))
    placebo = u[(u.week_before >= 2)].assign(late=(u.week_before == 7).astype(int))
    docs = s[(s.kind_documentation == 1) & (s[[f"kind_{k}" for k in KINDS if k not in ("documentation", "none")]].sum(axis=1) == 0)]
    full = s[(s.ready_quality == "full") & s.timeline_complete]
    summ = []
    for y, name in OUTCOMES.items():
        for label, d in [("final week vs weeks 2-12", s), ("placebo: week 7 vs other weeks 2-12", placebo),
                         ("negative control: docs-only PRs", docs),
                         ("robustness: full readiness, verified timeline", full)]:
            rr, lo, hi = rr_poisson(d, "late", y)
            summ.append({"outcome": name, "comparison": label, "rr": rr, "lo": lo, "hi": hi,
                         "events_late": int(obs(d, y).query("late == 1")[y].sum()),
                         "events": int(obs(d, y)[y].sum()), "n": len(obs(d, y))})
    out = pd.DataFrame(summ)
    out.to_csv(RES / "rq1_rr.csv", index=False)
    return out


# ---------------------------------------------------------------- RQ2

def att_weights(d):
    """Odds weights making early PRs resemble late PRs on covariates (cycle FE included)."""
    X = StandardScaler().fit_transform(design(d))
    lr = LogisticRegression(C=10.0, max_iter=5000).fit(X, d.late)
    p = np.clip(lr.predict_proba(X)[:, 1], 0.01, 0.99)
    w = pd.Series(np.where(d.late == 1, 1.0, p / (1 - p)), index=d.index)
    early = d.late == 0
    w[early] *= (d.late == 1).sum() / w[early].sum()
    return w, p


def rq2_estimates(d, y):
    d = obs(d, y)
    w, _ = att_weights(d)
    late, early = d.late == 1, d.late == 0
    naive = d.loc[late, y].mean() - d.loc[early, y].mean()
    adj = d.loc[late, y].mean() - np.average(d.loc[early, y], weights=w[early])
    # Doubly robust: outcome model fitted on early PRs, residuals reweighted.
    X = design(d)
    om = sm.OLS(d.loc[early, y], sm.add_constant(X[early], has_constant="add")).fit()
    mu0 = om.predict(sm.add_constant(X, has_constant="add"))
    r = d[y] - mu0
    dr = r[late].mean() - np.average(r[early], weights=w[early])
    return naive, adj, dr


def smd(d, w):
    late, early = d.late == 1, d.late == 0
    rows = []
    for c in COVARS:
        x = d[c].astype(float)
        sd = np.sqrt((x[late].var() + x[early].var()) / 2) or 1.0
        rows.append({"covariate": c,
                     "smd_raw": (x[late].mean() - x[early].mean()) / sd,
                     "smd_weighted": (x[late].mean() - np.average(x[early], weights=w[early])) / sd})
    return pd.DataFrame(rows)


def rq2(u, reps):
    d = u[u.final_week | u.week_before.between(5, 12)].assign(late=lambda x: x.final_week.astype(int))
    w, p = att_weights(d)
    smd(d, w).to_csv(RES / "rq2_balance.csv", index=False)
    rows = []
    for y, name in OUTCOMES.items():
        naive, adj, dr = rq2_estimates(d, y)
        boot = cluster_boot(d, lambda b: rq2_estimates(b, y), reps)
        share = 1 - adj / naive if naive else np.nan
        bs = 1 - boot[:, 1] / boot[:, 0]
        rows.append({"outcome": name, "naive_gap_pp": 100 * naive, "naive_ci": tuple(100 * np.array(ci(boot[:, 0]))),
                     "adjusted_gap_pp": 100 * adj, "adjusted_ci": tuple(100 * np.array(ci(boot[:, 1]))),
                     "dr_gap_pp": 100 * dr, "dr_ci": tuple(100 * np.array(ci(boot[:, 2]))),
                     "composition_share": share, "share_ci": ci(bs)})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "rq2_composition.csv", index=False)
    pd.DataFrame({"propensity_late": p[d.late == 1]}).describe().to_csv(RES / "rq2_overlap_late.csv")
    pd.DataFrame({"propensity_early": p[d.late == 0]}).describe().to_csv(RES / "rq2_overlap_early.csv")
    return out, d, w


# ---------------------------------------------------------------- RQ3

def within(df, cols, group, w):
    """Weighted within-transformation (absorbs author fixed effects, FWL with weights)."""
    g = df[group].values
    out = {}
    W = pd.Series(w, index=df.index)
    for c in cols:
        x = df[c].astype(float)
        m = (x * W).groupby(g).transform("sum") / W.groupby(g).transform("sum")
        out[c] = x - m
    return pd.DataFrame(out, index=df.index)


def rq3_author_fe(d, w, y):
    d, w = obs(d, y), w[obs(d, y).index]
    multi = d.author_id.map(d.author_id.value_counts()) > 1
    dd, ww = d[multi], w[multi]
    X = design(dd).assign(late=dd.late.astype(float))
    Z = within(pd.concat([X, dd[[y, "author_id"]]], axis=1), list(X.columns) + [y], "author_id", ww)
    keep = [c for c in X.columns if Z[c].abs().sum() > 0]
    r = sm.WLS(Z[y], Z[keep], weights=ww).fit(
        cov_type="cluster", cov_kwds={"groups": np.column_stack([dd.cycle, dd.author_id])})
    return r.params["late"], r.bse["late"], int(multi.sum()), dd.author_id.nunique()


def rq3_mediation(d, w, y):
    d, w = obs(d, y), w[obs(d, y).index]
    X = design(d)
    total = ols(d[y], X.assign(late=d.late), d.cycle, w).params["late"]
    a = {m: ols(d[m], X.assign(late=d.late), d.cycle, w).params["late"] for m in MEDIATORS}
    b_fit = ols(d[y], X.assign(late=d.late, **{m: d[m] for m in MEDIATORS}), d.cycle, w)
    indirect = sum(a[m] * b_fit.params[m] for m in MEDIATORS)
    return total, indirect, b_fit.params["late"], a


def rq3_v131(u):
    """Pressure removed: v1.31 PRs ready in the 7 days before the ORIGINAL freeze (10 Jul 2024)."""
    orig = pd.Timestamp("2024-07-10T02:00Z")
    t131 = u[(u.release == "1.31") & (u.ready_at >= orig - pd.Timedelta(days=7)) & (u.ready_at < orig)]
    ctrl = u[(u.release != "1.31") & u.final_week]
    base = u[(u.release != "1.31") & u.week_before.between(2, 12)]
    act = u[(u.release == "1.31") & u.final_week]
    rows = []
    for y, name in OUTCOMES.items():
        for label, d in [("v1.31, 7d before original freeze (pressure removed)", t131),
                         ("v1.31, 7d before actual freeze", act),
                         ("other cycles, final week (pressure)", ctrl),
                         ("other cycles, weeks 2-12 (baseline)", base)]:
            rows.append({"outcome": name, "group": label, "n": int(d[y].notna().sum()), "events": int(d[y].sum()),
                         "rate_pct": 100 * d[y].mean() if len(d) else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "rq3_v131.csv", index=False)
    return out


def rq3_dose(u):
    cal = pd.read_csv(ROOT / "data" / "k8s_freeze_calendar.csv", dtype={"release": str})
    rows = []
    for rel, g in u.groupby("release"):
        late, early = g[g.final_week], g[g.week_before.between(5, 12)]
        row = {"release": rel, "n": len(g), "final_week_share": len(late) / len(g[g.week_before <= 12])}
        for y in OUTCOMES:
            row[f"gap_{y}"] = late[y].mean() - early[y].mean()
        rows.append(row)
    c = pd.DataFrame(rows).merge(cal[["release", "freeze_week"]], on="release", how="left")
    c["post_kep2572"] = c.release.map(lambda r: int(r.split(".")[1]) >= 22).astype(int)
    c.to_csv(RES / "rq3_dose_cycles.csv", index=False)
    res = []
    for dv in ["final_week_share"] + [f"gap_{y}" for y in OUTCOMES]:
        d = c.dropna(subset=["freeze_week", dv])
        r = sm.WLS(d[dv], sm.add_constant(d[["freeze_week", "post_kep2572"]].astype(float)), weights=d.n).fit(cov_type="HC1")
        for k in ["freeze_week", "post_kep2572"]:
            res.append({"dv": dv, "term": k, "coef": r.params[k], "lo": r.conf_int().loc[k, 0],
                        "hi": r.conf_int().loc[k, 1], "n_cycles": len(d)})
    out = pd.DataFrame(res)
    out.to_csv(RES / "rq3_dose.csv", index=False)
    return out


def rq3(u, d, w, reps):
    rows = []
    for y, name in OUTCOMES.items():
        b, se, n, na = rq3_author_fe(d, w, y)
        total, ind, direct, a = rq3_mediation(d, w, y)
        boot = cluster_boot(d, lambda bb: (lambda ww: rq3_mediation(bb, ww, y)[1])(att_weights(bb)[0]), reps)
        rows.append({"outcome": name, "late_effect_authorFE_pp": 100 * b,
                     "authorFE_ci": (100 * (b - 1.96 * se), 100 * (b + 1.96 * se)),
                     "n_authorFE": n, "authors": na, "total_pp": 100 * total, "indirect_pp": 100 * ind,
                     "indirect_ci": tuple(100 * np.array(ci(boot))), "direct_pp": 100 * direct,
                     "share_mediated": ind / total if total else np.nan,
                     **{f"a_{m}": v for m, v in a.items()}})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "rq3_pressure.csv", index=False)
    return out


# ---------------------------------------------------------------- RQ4

def rq4(u):
    expos = ["log_conc_area", "log_conc_total", "log_queue_wait"]
    rows = []
    for label, d, fe in [("final week (6h-bin FE)", u[u.final_week],
                          lambda x: pd.get_dummies((x.hours_to_freeze // 6).astype(int), prefix="h", drop_first=True, dtype=float)),
                         ("weeks 2-12 (week FE)", u[u.week_before.between(2, 12)],
                          lambda x: pd.get_dummies(x.week_before, prefix="w", drop_first=True, dtype=float))]:
        for y, name in OUTCOMES.items():
            dy = obs(d, y)
            if dy[y].sum() == 0:
                continue
            X = pd.concat([design(dy, expos), fe(dy)], axis=1)
            r = ols(dy[y], X, dy.cycle)
            for e in expos:
                iqr = d[e].quantile(0.75) - d[e].quantile(0.25)
                rows.append({"sample": label, "outcome": name, "exposure": e, "coef": r.params[e],
                             "per_iqr_pp": 100 * r.params[e] * iqr,
                             "lo_pp": 100 * (r.params[e] - 1.96 * r.bse[e]) * iqr,
                             "hi_pp": 100 * (r.params[e] + 1.96 * r.bse[e]) * iqr,
                             "n": len(dy), "events": int(dy[y].sum())})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "rq4_congestion.csv", index=False)
    return out


# ---------------------------------------------------------------- RQ5

def rq5(u):
    d = u[u.any_problem.astype(bool)].copy()
    d["revert_given_problem"] = d.y1_revert
    d["in_freeze"] = d.problem_in_freeze.astype(bool).astype(int)
    X = d[COVARS].astype(float).assign(in_freeze=d.in_freeze)
    r = ols(d.revert_given_problem, X, d.cycle)
    raw = d.groupby("in_freeze").revert_given_problem.agg(["mean", "size"])
    out = pd.DataFrame([{"n_problems": len(d), "in_freeze_n": int(d.in_freeze.sum()),
                         "revert_share_in_freeze": raw.loc[1, "mean"] if 1 in raw.index else np.nan,
                         "revert_share_outside": raw.loc[0, "mean"] if 0 in raw.index else np.nan,
                         "adj_diff_pp": 100 * r.params["in_freeze"],
                         "lo_pp": 100 * (r.params["in_freeze"] - 1.96 * r.bse["in_freeze"]),
                         "hi_pp": 100 * (r.params["in_freeze"] + 1.96 * r.bse["in_freeze"])}])
    out.to_csv(RES / "rq5_freeze_response.csv", index=False)
    return out


# ---------------------------------------------------------------- main

def main():
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--links", default="all", choices=LINKS)
    args = ap.parse_args()
    if args.links != "all":
        RES = RES / f"links_{args.links}"
    RES.mkdir(exist_ok=True)
    u = load(args.links)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(f"units: {len(u)} | cycles: {u.release.nunique()} | events: "
          + ", ".join(f"{OUTCOMES[y]}={int(u[y].sum())}" for y in OUTCOMES))
    print("\n== RQ1\n", rq1(u).round(3).to_string(index=False))
    r2, d, w = rq2(u, args.boot)
    print("\n== RQ2\n", r2.round(3).to_string(index=False))
    print("\n== RQ3 author FE + mediation\n", rq3(u, d, w, args.boot).round(3).to_string(index=False))
    print("\n== RQ3 v1.31\n", rq3_v131(u).round(3).to_string(index=False))
    print("\n== RQ3 dose\n", rq3_dose(u).round(4).to_string(index=False))
    print("\n== RQ4\n", rq4(u).round(3).to_string(index=False))
    print("\n== RQ5\n", rq5(u).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
