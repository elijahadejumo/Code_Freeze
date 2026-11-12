"""Reproduce every table and figure in the paper from the included data (no GitHub token needed).

    python analyze.py            # about 20 minutes

Reads data/ and writes results/ and figures/. See README.md for which output backs which part
of the paper.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")      # multithreaded BLAS makes the bootstrap several times slower

import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA, RES, FIG = ROOT / "data", ROOT / "results", ROOT / "figures"
SEED, BOOT = 2027, 400
HORIZON_DAYS, ESCAPE_DAYS, SOAK_DAYS = 30, 90, 180
LINK_SETS = {"all": "all_links", "checked": "checked_links"}
OUTCOMES = {"y1_revert": "Y1 revert", "y2_fix": "Y2 follow-up fix", "y3_post_release": "Y3 post-release problem"}
SOAK_OUTCOMES = {"p_total": "any problem", "p_pre": "found before release", "p_post": "found after release"}
KINDS = ["feature", "bug", "cleanup", "api-change", "documentation", "regression", "failing-test",
         "flake", "deprecation", "none"]
MEDIATORS = ["m_review_threads", "m_reviews", "m_lgtm_removed", "m_last_commit_to_ready"]
COVARS = (["size", "size_missing", "n_sigs", "has_milestone", "priority_critical", "log_lines",
           "log_files", "log_age", "log_exp"] + [f"kind_{k}" for k in KINDS if k != "none"])
TIME_BINS = ["weeks 9-12", "weeks 5-8", "weeks 3-4", "week 2", "days 3-7", "last 48h"]
SOAK_BINS = [("week 1", 1, 1), ("week 2", 2, 2), ("weeks 3-4", 3, 4), ("weeks 5-6", 5, 6),
             ("weeks 7-8", 7, 8), ("weeks 9-12", 9, 12)]
RNG = np.random.default_rng(SEED)


def save(df, path, index=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.map(lambda v: tuple(float(x) for x in v) if isinstance(v, tuple) else v)   # CIs as plain (lo, hi)
    df.to_csv(path, index=index)
    print(f"\n== {path.relative_to(ROOT)}\n{df.round(4).to_string(index=index)}", flush=True)


# ================================================================ data

def load_links(links="all"):
    """Outcome links. "all": every revert and fix link. "checked": verified reverts (they undo the
    original's changes) and fixes that use cue words, share a file with the original, and were
    confirmed in the hand coding ("?" is not a fix)."""
    lr = pd.read_csv(DATA / "links_reverts.csv")
    lf = pd.read_csv(DATA / "links_fixes.csv")
    if links == "all":
        return lr, lf
    key = pd.read_csv(DATA / "validation" / "human_census_key.csv")
    ans = pd.read_csv(DATA / "validation" / "human_census_answers.csv").set_index("id").answer
    return lr[lr.verified], lf.merge(key[key.id.map(ans) == "y"][["orig_pr", "fix_pr"]])


def outcomes_from_links(u, lr, lf):
    """Rebuild Y1-Y3 as build.py does, from another set of links."""
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
    u = pd.read_csv(DATA / "analysis.csv.gz", dtype={"release": str},
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


def observed_until():
    """The data end on the day of the last revert or fix they contain (the crawl, 28 Oct 2026).
    Taken from the data, not the clock, so a re-run later gives the same results."""
    ev = pd.concat([pd.read_csv(DATA / f, usecols=["event_at"]) for f in ("links_reverts.csv", "links_fixes.csv")])
    return pd.to_datetime(ev.event_at, utc=True).max().floor("D")


OBSERVED_UNTIL = observed_until()


def soak_outcomes(u, links="all", horizon=SOAK_DAYS):
    """Every PR followed for the same days from MERGE; its first problem split into found
    before vs after release. Windows not yet closed when the data end are NaN."""
    lr, lf = load_links(links)
    ev = pd.concat([lr, lf])[["orig_pr", "event_at", "lag_days"]]
    ev["event_at"] = pd.to_datetime(ev.event_at, utc=True)
    ev = ev[ev.lag_days <= horizon]
    first_at = ev.groupby("orig_pr").event_at.min()
    first_lag = ev.groupby("orig_pr").lag_days.min()
    closed = pd.to_datetime(u.merged_at, utc=True) + pd.Timedelta(days=horizon) < OBSERVED_UNTIL
    hit = u.pr.isin(first_at.index)
    post = u.pr.map(first_at) > pd.to_datetime(u.release_at, utc=True)
    u["p_total"] = np.where(closed, hit, np.nan)
    u["p_pre"] = np.where(closed, hit & ~post, np.nan)
    u["p_post"] = np.where(closed, hit & post, np.nan)
    u["first_lag"] = u.pr.map(first_lag)
    u["days_to_release"] = (pd.to_datetime(u.release_at, utc=True)
                            - pd.to_datetime(u.merged_at, utc=True)).dt.total_seconds() / 86400
    return u


def late_vs_early(u):
    """Late = ready in the final week; early = weeks 2-12."""
    return u[u.week_before <= 12].assign(late=lambda x: x.final_week.astype(int))


def late_vs_weeks_5_12(u):
    """Composition comparison: early PRs from weeks 5-12, before the weekly count starts to rise."""
    return u[u.final_week | u.week_before.between(5, 12)].assign(late=lambda x: x.final_week.astype(int))


# ================================================================ estimators

def cycle_dummies(df):
    return pd.get_dummies(df.release, prefix="c", drop_first=True, dtype=float)


def design(df, extra=()):
    X = df[list(COVARS) + list(extra)].astype(float)
    return pd.concat([X, cycle_dummies(df)], axis=1)


def ols(y, X, groups, w=None):
    X = sm.add_constant(X, has_constant="add")
    m = sm.WLS(y, X, weights=w) if w is not None else sm.OLS(y, X)
    return m.fit(cov_type="cluster", cov_kwds={"groups": groups})


def lpm(df, y, terms, cluster="cycle"):
    X = pd.concat([df[terms].astype(float), cycle_dummies(df)], axis=1)
    return sm.OLS(df[y].astype(float), sm.add_constant(X, has_constant="add")).fit(
        cov_type="cluster", cov_kwds={"groups": df[cluster]})


def cluster_boot(df, fn, reps):
    """Bootstrap over releases: resample whole release cycles with replacement."""
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


def obs(df, y):
    return df[df[y].notna()]


def rr_poisson(df, treat, y):
    """Risk ratio from a Poisson GLM, CIs clustered by release."""
    df = obs(df, y)
    if df.loc[df[treat] == 1, y].sum() == 0 or df.loc[df[treat] == 0, y].sum() == 0:
        return np.nan, np.nan, np.nan
    X = sm.add_constant(df[[treat]].astype(float))
    r = sm.GLM(df[y], X, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": df.cycle})
    b, se = r.params[treat], r.bse[treat]
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)


def att_weights(d):
    """Odds weights making early PRs resemble late PRs on covariates (release FE included)."""
    X = StandardScaler().fit_transform(design(d))
    lr = LogisticRegression(C=10.0, max_iter=5000).fit(X, d.late)
    p = np.clip(lr.predict_proba(X)[:, 1], 0.01, 0.99)
    w = pd.Series(np.where(d.late == 1, 1.0, p / (1 - p)), index=d.index)
    early = d.late == 0
    w[early] *= (d.late == 1).sum() / w[early].sum()
    return w, p


def gap_estimates(d, y):
    """Late - early gap: naive, ATT-weighted, and doubly robust."""
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


def author_fe(d, w, y):
    """Late effect within author (authors with 2+ PRs), clustered by release and author."""
    d, w = obs(d, y), w[obs(d, y).index]
    multi = d.author_id.map(d.author_id.value_counts()) > 1
    dd, ww = d[multi], w[multi]
    X = design(dd).assign(late=dd.late.astype(float))
    Z = within(pd.concat([X, dd[[y, "author_id"]]], axis=1), list(X.columns) + [y], "author_id", ww)
    keep = [c for c in X.columns if Z[c].abs().sum() > 0]
    r = sm.WLS(Z[y], Z[keep], weights=ww).fit(
        cov_type="cluster", cov_kwds={"groups": np.column_stack([dd.cycle, dd.author_id])})
    return r.params["late"], r.bse["late"], int(multi.sum()), dd.author_id.nunique()


def mediation(d, w, y):
    d, w = obs(d, y), w[obs(d, y).index]
    X = design(d)
    total = ols(d[y], X.assign(late=d.late), d.cycle, w).params["late"]
    a = {m: ols(d[m], X.assign(late=d.late), d.cycle, w).params["late"] for m in MEDIATORS}
    b_fit = ols(d[y], X.assign(late=d.late, **{m: d[m] for m in MEDIATORS}), d.cycle, w)
    indirect = sum(a[m] * b_fit.params[m] for m in MEDIATORS)
    return total, indirect, b_fit.params["late"], a


def share_after(lags, days):
    """Share of problems expected after release, given discovery lags and days to release."""
    lags = np.sort(lags)
    return float(np.mean([1 - np.searchsorted(lags, t, side="right") / len(lags) for t in days]))


# ================================================================ RQ1: the impact

def impact(u):
    """Risk ratios, late vs early, and a placebo deadline 6 weeks before the freeze."""
    s = late_vs_early(u)
    placebo = u[(u.week_before >= 2)].assign(late=(u.week_before == 7).astype(int))
    rows = []
    for y, name in OUTCOMES.items():
        for label, d in [("final week vs weeks 2-12", s), ("placebo: week 7 vs other weeks 2-12", placebo)]:
            rr, lo, hi = rr_poisson(d, "late", y)
            rows.append({"outcome": name, "comparison": label, "rr": rr, "lo": lo, "hi": hi,
                         "events_late": int(obs(d, y).query("late == 1")[y].sum()),
                         "events": int(obs(d, y)[y].sum()), "n": len(obs(d, y))})
    return pd.DataFrame(rows)


def three_releases_a_year(u):
    """Per-release regressions (weighted by release size) on the move to three releases a year
    (v1.22, KEP-2572), adjusting for the week of the cycle the freeze fell in."""
    cal = pd.read_csv(DATA / "k8s_freeze_calendar.csv", dtype={"release": str})
    rows = []
    for rel, g in u.groupby("release"):
        late, early = g[g.final_week], g[g.week_before.between(5, 12)]
        row = {"release": rel, "n": len(g), "final_week_share": len(late) / len(g[g.week_before <= 12])}
        for y in OUTCOMES:
            row[f"gap_{y}"] = late[y].mean() - early[y].mean()
        rows.append(row)
    c = pd.DataFrame(rows).merge(cal[["release", "freeze_week"]], on="release", how="left")
    c["post_kep2572"] = c.release.map(lambda r: int(r.split(".")[1]) >= 22).astype(int)
    res = []
    for dv in ["final_week_share"] + [f"gap_{y}" for y in OUTCOMES]:
        d = c.dropna(subset=["freeze_week", dv])
        r = sm.WLS(d[dv], sm.add_constant(d[["freeze_week", "post_kep2572"]].astype(float)), weights=d.n).fit(cov_type="HC1")
        for k in ["freeze_week", "post_kep2572"]:
            res.append({"dv": dv, "term": k, "coef": r.params[k], "lo": r.conf_int().loc[k, 0],
                        "hi": r.conf_int().loc[k, 1], "n_cycles": len(d)})
    return pd.DataFrame(res)


def v131(u):
    """v1.31's freeze moved from 10 to 24 July 2024: problems in the week before each deadline."""
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
    return pd.DataFrame(rows)


# ================================================================ RQ2: four explanations

def composition(d, reps):
    """What lands late: share of the gap removed by reweighting early PRs to look like late ones."""
    rows = []
    for y, name in OUTCOMES.items():
        naive, adj, dr = gap_estimates(d, y)
        boot = cluster_boot(d, lambda b: gap_estimates(b, y), reps)
        share = 1 - adj / naive if naive else np.nan
        bs = 1 - boot[:, 1] / boot[:, 0]
        rows.append({"outcome": name, "naive_gap_pp": 100 * naive, "naive_ci": tuple(100 * np.array(ci(boot[:, 0]))),
                     "adjusted_gap_pp": 100 * adj, "adjusted_ci": tuple(100 * np.array(ci(boot[:, 1]))),
                     "dr_gap_pp": 100 * dr, "dr_ci": tuple(100 * np.array(ci(boot[:, 2]))),
                     "composition_share": share, "share_ci": ci(bs)})
    return pd.DataFrame(rows)


def review_pressure(d, w, reps):
    """Pressure on review: late effect within author, and the part mediated by review measures."""
    rows = []
    for y, name in OUTCOMES.items():
        b, se, n, na = author_fe(d, w, y)
        total, ind, direct, a = mediation(d, w, y)
        boot = cluster_boot(d, lambda bb: (lambda ww: mediation(bb, ww, y)[1])(att_weights(bb)[0]), reps)
        rows.append({"outcome": name, "late_effect_authorFE_pp": 100 * b,
                     "authorFE_ci": (100 * (b - 1.96 * se), 100 * (b + 1.96 * se)),
                     "n_authorFE": n, "authors": na, "total_pp": 100 * total, "indirect_pp": 100 * ind,
                     "indirect_ci": tuple(100 * np.array(ci(boot))), "direct_pp": 100 * direct,
                     "share_mediated": ind / total if total else np.nan,
                     **{f"a_{m}": v for m, v in a.items()}})
    return pd.DataFrame(rows)


def merge_queue(u):
    """A crowded merge queue: problems vs concurrent merges, final week and (as control) weeks 2-12."""
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
    return pd.DataFrame(rows)


def freeze_response(u):
    """The freeze itself: are problems surfacing during the freeze reverted more often than fixed?"""
    d = u[u.any_problem.astype(bool)].copy()
    d["revert_given_problem"] = d.y1_revert
    d["in_freeze"] = d.problem_in_freeze.astype(bool).astype(int)
    X = d[COVARS].astype(float).assign(in_freeze=d.in_freeze)
    r = ols(d.revert_given_problem, X, d.cycle)
    raw = d.groupby("in_freeze").revert_given_problem.agg(["mean", "size"])
    return pd.DataFrame([{"n_problems": len(d), "in_freeze_n": int(d.in_freeze.sum()),
                          "revert_share_in_freeze": raw.loc[1, "mean"] if 1 in raw.index else np.nan,
                          "revert_share_outside": raw.loc[0, "mean"] if 0 in raw.index else np.nan,
                          "adj_diff_pp": 100 * r.params["in_freeze"],
                          "lo_pp": 100 * (r.params["in_freeze"] - 1.96 * r.bse["in_freeze"]),
                          "hi_pp": 100 * (r.params["in_freeze"] + 1.96 * r.bse["in_freeze"])}])


# ================================================================ RQ2: soak time

def soak_total_vs_timing(u, links, reps):
    """180 days from merge: problems in total, found before release, and found after release."""
    s, d = late_vs_early(u), late_vs_weeks_5_12(u)
    w, _ = att_weights(d)
    rows = []
    for y, name in SOAK_OUTCOMES.items():
        rr, lo, hi = rr_poisson(s, "late", y)
        naive, adj, dr = gap_estimates(d, y)
        boot = cluster_boot(d, lambda b: gap_estimates(b, y), reps)
        b, se, _, _ = author_fe(d, w, y)
        o = obs(s, y)
        rows.append({"outcome": name, "horizon_days": SOAK_DAYS, "links": links,
                     "late_pct": 100 * o.loc[o.late == 1, y].mean(), "early_pct": 100 * o.loc[o.late == 0, y].mean(),
                     "rr": rr, "rr_lo": lo, "rr_hi": hi,
                     "naive_pp": 100 * naive, "naive_ci": tuple(100 * np.array(ci(boot[:, 0]))),
                     "adjusted_pp": 100 * adj, "adjusted_ci": tuple(100 * np.array(ci(boot[:, 1]))),
                     "dr_pp": 100 * dr, "dr_ci": tuple(100 * np.array(ci(boot[:, 2]))),
                     "authorFE_pp": 100 * b, "authorFE_ci": (100 * (b - 1.96 * se), 100 * (b + 1.96 * se))})
    return pd.DataFrame(rows)


def soak_prediction(u):
    """Share of problems found after release: observed, and predicted from the early PRs'
    discovery lags applied to each group's days from merge to release."""
    o = obs(late_vs_early(u), "p_total")
    prob = o[o.p_total == 1]
    early_lags = np.sort(prob.loc[prob.late == 0, "first_lag"].values)
    rows = []
    for late, g in prob.groupby("late"):
        pred = np.mean([np.mean(early_lags > t) for t in g.days_to_release])
        rows.append({"group": "final week" if late else "weeks 2-12", "problems": len(g),
                     "median_days_merge_to_release": g.days_to_release.median(),
                     "observed_share_after_release": g.p_post.mean(),
                     "predicted_share_if_early_timing": pred})
    return pd.DataFrame(rows)


def soak_by_week(u):
    """Per week before freeze: share found after release, observed vs predicted from the OTHER
    weeks' discovery lags (no week predicts itself)."""
    o = obs(late_vs_early(u), "p_total")
    prob = o[o.p_total == 1]
    rows = []
    for name, lo, hi in SOAK_BINS:
        inb = o.week_before.between(lo, hi)
        pin, pout = prob[prob.week_before.between(lo, hi)], prob[~prob.week_before.between(lo, hi)]
        pred = share_after(pout.first_lag.values, pin.days_to_release.values)
        rows.append({"bin": name, "n": int(inb.sum()), "median_days_to_release": o.loc[inb, "days_to_release"].median(),
                     "total_pct": 100 * o.loc[inb, "p_total"].mean(),
                     "escape_pct": 100 * o.loc[inb, "p_post"].mean(),
                     "escape_share_observed": pin.p_post.mean(), "escape_share_predicted": pred,
                     "escape_pct_predicted": 100 * o.loc[inb, "p_total"].mean() * pred})
    return pd.DataFrame(rows)


def soak_across_releases(u):
    """Direct test (Future Work): does a longer freeze-to-release interval shrink the late excess?"""
    o = obs(late_vs_early(u), "p_total").copy()
    o["f_weeks"] = (pd.to_datetime(o.release_at, utc=True) - pd.to_datetime(o.freeze, utc=True)).dt.days / 7
    o["late_x_f"] = o.late * (o.f_weeks - o.f_weeks.mean())
    o["late_x_kep"] = o.late * o.release.map(lambda r: int(r.split(".")[1]) >= 22)
    rows = []
    for y in SOAK_OUTCOMES:
        r = lpm(o, y, ["late", "late_x_f", "late_x_kep"])
        lo, hi = r.conf_int().loc["late_x_f"]
        rows.append({"outcome": SOAK_OUTCOMES[y], "late_effect_pp": 100 * r.params["late"],
                     "per_extra_week_freeze_to_release_pp": 100 * r.params["late_x_f"],
                     "lo_pp": 100 * lo, "hi_pp": 100 * hi, "p": r.pvalues["late_x_f"],
                     "f_weeks_range": f"{o.f_weeks.min():.1f}-{o.f_weeks.max():.1f}"})
    return pd.DataFrame(rows)


# ================================================================ descriptive series (RQ1, Figs. 2-4)

def time_bin(u):
    return np.select([u.week_before.between(9, 12), u.week_before.between(5, 8), u.week_before.between(3, 4),
                      u.week_before == 2, u.final_week & (u.hours_to_freeze >= 48), u.hours_to_freeze < 48],
                     TIME_BINS, default="")


def wilson(x):
    x = x.dropna()
    lo, hi = proportion_confint(int(x.sum()), len(x), method="wilson")
    return pd.Series({"n": len(x), "events": int(x.sum()), "pct": 100 * x.mean(), "lo": 100 * lo, "hi": 100 * hi})


def descriptive():
    u = load()
    u = u[u.week_before <= 12].copy()
    n_rel = u.release.nunique()
    u["days_to_release"] = (u.release_at - u.merged_at).dt.total_seconds() / 86400

    day = (u.hours_to_freeze // 24).astype(int) + 1
    daily = (day[day <= 28].value_counts().sort_index() / n_rel).rename("prs_per_release").reset_index()
    daily.columns = ["day_before", "prs_per_release"]
    daily["weeks_2_12_daily_mean"] = (u.week_before >= 2).sum() / n_rel / (11 * 7)
    save(daily, RES / "rush_daily.csv")

    c = u.groupby("release").agg(freeze=("freeze", "first"), prs=("pr", "size"), final_week=("final_week", "sum"))
    c["final_week_share"] = c.final_week / c.prs
    c["final_week_x_avg_week"] = c.final_week / ((c.prs - c.final_week) / 11)
    c["three_per_year"] = c.index.map(lambda r: int(r.split(".")[1]) >= 22)
    save(c.sort_values("freeze"), RES / "rush_by_release.csv", index=True)

    v = u[u.release == "1.31"].week_before.value_counts().sort_index().rename("prs").reset_index()
    v.columns = ["week_before_actual_freeze", "prs"]
    save(v, RES / "rush_v131_weeks.csv")

    u["bin"] = time_bin(u)
    h = load("checked")
    h = h[h.week_before <= 12].assign(bin=lambda x: time_bin(x))
    rows = []
    for b in TIME_BINS:
        g, gh = u[u.bin == b], h[h.bin == b]
        row = {"bin": b, "prs": len(g), "lines_median": g.lines.median(), "feature_pct": 100 * g.kind_feature.mean(),
               "bug_pct": 100 * g.kind_bug.mean(), "review_threads_mean": g.review_threads.mean(),
               "reviews_median": g.reviews.median(), "author_prior_prs_median": g.author_prior_prs.median(),
               "age_days_median": g.age_days.median(), "days_to_release_median": g.days_to_release.median()}
        for name, x in [("post_release_all", g.y3_post_release), ("post_release_checked", gh.y3_post_release),
                        ("revert30_verified", gh.y1_revert)]:
            row.update({f"{name}_{k}": val for k, val in wilson(x).items()})
        rows.append(row)
    save(pd.DataFrame(rows), RES / "by_time_before_freeze.csv")

    s = soak_outcomes(load(), "checked")
    s = s[(s.week_before <= 12) & (s.p_total == 1)]
    disc = pd.DataFrame({"days": range(SOAK_DAYS + 1)})
    for name, g in [("late", s[s.final_week]), ("early", s[~s.final_week])]:
        disc[name] = [(g.first_lag <= k).mean() for k in disc.days]
        disc[f"{name}_problems"] = len(g)
        disc[f"{name}_median_days_to_release"] = g.days_to_release.median()
    disc.to_csv(RES / "discovery_curve.csv", index=False)
    print(f"\n== results/discovery_curve.csv\n{disc.iloc[[3, 7, 14, 21, 30, 60, 90, 180]].round(3).to_string(index=False)}")


def link_check_precision():
    """Hand coding of fix links: share confirmed, overall and by late/early group."""
    def prec(v):
        k, n = int((v == "y").sum()), len(v)
        lo, hi = proportion_confint(k, n, method="wilson") if n else (np.nan, np.nan)
        return {"n": n, "correct": k, "unsure": int((v == "?").sum()), "precision": k / n if n else np.nan,
                "lo": lo, "hi": hi}

    val = DATA / "validation"
    rows = pd.read_csv(val / "human_census_key.csv").merge(pd.read_csv(val / "human_census_answers.csv"), on="id")
    u = pd.read_csv(DATA / "analysis.csv.gz", usecols=["pr", "final_week", "merged_at", "release_at"],
                    parse_dates=["merged_at", "release_at"])
    lf = pd.read_csv(DATA / "links_fixes.csv", parse_dates=["event_at"])
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
    save(pd.DataFrame(res), RES / "link_check_precision.csv")


# ================================================================ figures

BLUE, ORANGE, GREY, INK, MUTED, GRID = "#2a6fd6", "#e8692f", "#8a8f98", "#1d2125", "#5b6168", "#e3e5e8"
BIN_LABELS = ["9–12w", "5–8w", "3–4w", "2w", "3–7d", "48h"]
LATE_BINS = {4, 5}


def style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "axes.titlesize": 8.5,
                         "axes.titlelocation": "left", "axes.titlecolor": INK})


def grid(ax):
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save_fig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig((FIG / name).with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {(FIG / name).relative_to(ROOT)}")


def fig_rush():
    """Fig. 2: (a) PRs per day before freeze, (b) v1.31's moved deadline, (c) the final week across releases."""
    daily = pd.read_csv(RES / "rush_daily.csv").query("day_before <= 21")
    v131w = pd.read_csv(RES / "rush_v131_weeks.csv")
    rel = pd.read_csv(RES / "rush_by_release.csv", parse_dates=["freeze"])
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.16, 2.05), gridspec_kw={"width_ratios": [1.2, 1, 1.2], "wspace": 0.42})

    days = list(range(21, 0, -1))
    per = daily.set_index("day_before").prs_per_release
    a.bar(range(21), [per[d] for d in days], width=0.75, color=[ORANGE if d <= 7 else BLUE for d in days],
          edgecolor="white", linewidth=0.5)
    avg = daily.weeks_2_12_daily_mean.iloc[0]
    a.axhline(avg, color=GREY, lw=0.8, ls="--")
    a.text(0, per.max() * 1.1, f"dashed: weeks 2–12, {avg:.0f} per day", color=MUTED, fontsize=6.5)
    for d in (1, 2):
        a.text(21 - d, per[d] + 1, f"{per[d]:.0f}", ha="center", color=INK, fontsize=7)
    a.set_xticks([0, 7, 14, 20], ["21", "14", "7", "1"])
    a.set_xlabel("days before code freeze")
    a.set_ylabel("PRs per release")
    a.set_ylim(0, per.max() * 1.18)
    a.set_title("(a) The rush before code freeze")
    grid(a)

    weeks = list(range(12, 0, -1))
    cnt = v131w.set_index("week_before_actual_freeze").prs
    b.bar(range(12), [cnt[w] for w in weeks], width=0.72, color=[ORANGE if w == 1 else BLUE for w in weeks],
          edgecolor="white", linewidth=0.5)
    top = cnt.max() * 1.32
    for x, lab, dx, ha in [(9.5, "original deadline", -0.12, "right"), (11.5, "actual freeze", 0.12, "left")]:
        b.axvline(x, color=INK, lw=0.8, ls=":")
        b.text(x + dx, top * 0.98, lab, ha=ha, va="top", rotation=90, fontsize=6.5, color=MUTED)
    b.set_xlim(-0.6, 12.3)
    for w in (1, 3):
        b.text(12 - w, cnt[w] + 2, f"{cnt[w]}", ha="center", color=INK, fontsize=7)
    b.set_xticks([0, 4, 8, 11], ["12", "8", "4", "1"])
    b.set_xlabel("weeks before actual freeze")
    b.set_ylabel("PRs")
    b.set_ylim(0, top)
    b.set_title("(b) v1.31: the deadline moved")
    grid(b)

    cut = rel.loc[rel.three_per_year, "freeze"].min()
    c.plot(rel.freeze, rel.final_week_x_avg_week, color=GREY, lw=0.8, zorder=2)
    c.scatter(rel.freeze, rel.final_week_x_avg_week, s=16, color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)
    for flag, g in rel.groupby("three_per_year"):
        m = g.final_week_x_avg_week.mean()
        c.hlines(m, g.freeze.min(), g.freeze.max(), color=INK, lw=1.0, ls="--")
        c.text(g.freeze.min() + pd.Timedelta(days=40), m - 0.32 if flag else m + 0.12, f"mean {m:.1f}×", fontsize=6.5,
               color=MUTED)
    c.axvline(cut, color=INK, lw=0.8, ls=":")
    c.text(cut, 3.75, " three releases\n a year", fontsize=6.5, color=MUTED, va="top")
    c.axhline(1, color=GREY, lw=0.6)
    c.text(rel.freeze.min(), 1.07, "no rush", fontsize=6.5, color=MUTED)
    c.set_ylim(0, 3.9)
    c.set_ylabel("final week ÷ average earlier week")
    c.set_xlabel("code freeze")
    c.xaxis.set_major_locator(mdates.YearLocator(2))
    c.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    c.set_title("(c) The rush across releases")
    grid(c)
    save_fig(fig, "fig_rush.pdf")


def fig_changes():
    """Fig. 3: as the freeze approaches, problems after release, reverts, size, features, review, days to release."""
    t = pd.read_csv(RES / "by_time_before_freeze.csv")
    x = list(range(len(t)))
    colors = [ORANGE if i in LATE_BINS else BLUE for i in x]
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.5), gridspec_kw={"hspace": 0.62, "wspace": 0.3})
    panels = [("(a) Problem found after release (%)", "post_release"), ("(b) Reverted within 30 days (%)", "revert30_verified"),
              ("(c) Lines changed (median)", "lines_median"), ("(d) Features (%)", "feature_pct"),
              ("(e) Review threads per PR (mean)", "review_threads_mean"),
              ("(f) Days from merge to release (median)", "days_to_release_median")]
    for ax, (title, col) in zip(axes.flat, panels):
        ax.axvspan(3.5, 5.5, color=ORANGE, alpha=0.09, lw=0)
        if col == "post_release":
            for name, dx, fill, ls in [("all", -0.09, True, "-"), ("checked", 0.09, False, "--")]:
                xs = [i + dx for i in x]
                y, lo, hi = t[f"post_release_{name}_pct"], t[f"post_release_{name}_lo"], t[f"post_release_{name}_hi"]
                ax.plot(xs, y, color=GREY, lw=0.9, ls=ls, zorder=2)
                ax.vlines(xs, lo, hi, color=colors, lw=0.8, zorder=2)
                ax.scatter(xs, y, s=18, zorder=3, color=colors if fill else "white", edgecolor=colors, linewidth=1.0)
            ax.legend(handles=[Line2D([], [], color=GREY, marker="o", markerfacecolor=BLUE, markeredgecolor=BLUE, ms=4,
                                      lw=0.9, label="all links"),
                               Line2D([], [], color=GREY, ls="--", marker="o", markerfacecolor="white",
                                      markeredgecolor=BLUE, ms=4, lw=0.9, label="checked links")],
                      frameon=False, fontsize=6.5, loc="upper left")
            ax.text(4.5, 2.75, "final week", ha="center", fontsize=6.5, color=MUTED)
            ax.set_ylim(0, 3.0)
        else:
            y = t[f"{col}_pct"] if col == "revert30_verified" else t[col]
            ax.plot(x, y, color=GREY, lw=0.9, zorder=2)
            if col == "revert30_verified":
                ax.vlines(x, t[f"{col}_lo"], t[f"{col}_hi"], color=colors, lw=0.8, zorder=2)
            ax.scatter(x, y, s=18, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
            ax.set_ylim(0, max((t[f"{col}_hi"] if col == "revert30_verified" else y).max() * 1.15, 1))
        ax.set_xticks(x, BIN_LABELS, fontsize=6.5)
        ax.set_title(title, fontsize=8)
        grid(ax)
    fig.supxlabel("time before code freeze when ready (w weeks, d days, h hours)", fontsize=7.5, y=0.0)
    save_fig(fig, "fig_changes.pdf")


def fig_soak():
    """Fig. 4: (a) how fast problems are found, late vs early, (b) share found after release,
    observed vs predicted from soak time (checked links)."""
    d = pd.read_csv(RES / "discovery_curve.csv")
    g = pd.read_csv(RES / LINK_SETS["checked"] / "soak_by_week.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.15), gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.28})

    for grp, color, label in [("early", BLUE, "weeks 2–12"), ("late", ORANGE, "final week")]:
        a.step(d.days, 100 * d[grp], where="post", color=color, lw=1.4,
               label=f"{label} ({d[f'{grp}_problems'].iloc[0]} problems)")
        rel = d[f"{grp}_median_days_to_release"].iloc[0]
        found = 100 * d.loc[d.days <= rel, grp].iloc[-1]
        a.axvline(rel, color=color, lw=0.8, ls=":")
        a.scatter([rel], [found], s=22, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        a.annotate(f"{found:.0f}%", (rel, found), textcoords="offset points", xytext=(4, -10), fontsize=7, color=INK)
        # Late label on three lines so it ends before the early release line.
        a.text(rel + 2, 4 if grp == "late" else 20,
               f"{grp} code\nreleased\n({rel:.0f} d)" if grp == "late" else f"{grp} code\nreleased ({rel:.0f} d)",
               fontsize=6.5, color=MUTED)
    a.set_xlim(0, 180)
    a.set_ylim(0, 102)
    a.set_xlabel("days from merge")
    a.set_ylabel("problems found so far (%)")
    a.legend(frameon=False, fontsize=6.5, loc="center right")
    a.set_title("(a) How fast problems are found")
    grid(a)

    x = g.median_days_to_release
    late = g.bin == "week 1"
    b.plot(x, 100 * g.escape_share_predicted, color=GREY, lw=1.4, ls="--")
    b.scatter(x, 100 * g.escape_share_observed, s=26, color=[ORANGE if l else BLUE for l in late],
              edgecolor="white", linewidth=0.8, zorder=3)
    for xi, yi, lab in zip(x, 100 * g.escape_share_observed, g.bin):
        # "week 2" sits just below the dashed line, so its label goes to the left of the dot.
        off, ha = ((-5, -2), "right") if lab == "week 2" else ((5, 4), "left")
        b.annotate(lab, (xi, yi), textcoords="offset points", xytext=off, ha=ha, fontsize=6.5, color=MUTED)
    b.legend(handles=[Line2D([], [], color=GREY, ls="--", lw=1.4, label="predicted from soak time"),
                      Line2D([], [], color="none", marker="o", markerfacecolor=ORANGE, markeredgecolor="white", ms=6,
                             label="observed, final week"),
                      Line2D([], [], color="none", marker="o", markerfacecolor=BLUE, markeredgecolor="white", ms=6,
                             label="observed, earlier weeks")],
             frameon=False, fontsize=6.5, loc="upper right")
    b.set_xlabel("median days from merge to release")
    b.set_ylabel("problems found after release (%)")
    b.set_ylim(0, 35)
    b.set_xlim(20, 105)
    b.set_title("(b) Share of problems found after release")
    grid(b)
    save_fig(fig, "fig_soak.pdf")


# ================================================================ main

def main():
    global RNG
    warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(f"data observed until {OBSERVED_UNTIL.date()}")

    descriptive()
    link_check_precision()

    for links, folder in LINK_SETS.items():
        out = RES / folder
        u = load(links)
        print(f"\n######## {links} links | units: {len(u)} | cycles: {u.release.nunique()} | events: "
              + ", ".join(f"{OUTCOMES[y]}={int(u[y].sum())}" for y in OUTCOMES), flush=True)
        save(impact(u), out / "impact.csv")
        save(three_releases_a_year(u), out / "three_releases_a_year.csv")
        save(v131(u), out / "v131_problems.csv")

        RNG = np.random.default_rng(SEED)
        d = late_vs_weeks_5_12(u)
        w, p = att_weights(d)
        save(smd(d, w), out / "composition_balance.csv")
        save(pd.concat([pd.DataFrame({"propensity_late": p[d.late == 1]}).describe(),
                        pd.DataFrame({"propensity_early": p[d.late == 0]}).describe()], axis=1),
             out / "composition_overlap.csv", index=True)
        save(composition(d, BOOT), out / "composition.csv")
        save(review_pressure(d, w, BOOT), out / "review_pressure.csv")
        save(merge_queue(u), out / "merge_queue.csv")
        save(freeze_response(u), out / "freeze_response.csv")

        RNG = np.random.default_rng(SEED)
        s = soak_outcomes(load(), links)
        save(soak_total_vs_timing(s, links, BOOT), out / "soak_total_vs_timing.csv")
        save(soak_prediction(s), out / "soak_prediction.csv")
        save(soak_by_week(s), out / "soak_by_week.csv")
        save(soak_across_releases(s), out / "soak_across_releases.csv")

    style()
    fig_rush()
    fig_changes()
    fig_soak()


if __name__ == "__main__":
    main()
