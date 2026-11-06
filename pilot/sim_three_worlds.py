"""Three worlds, one pattern: a synthetic illustration for the motivating example.

Each world generates release cycles that look like Kubernetes (about 960 merges per
12-week pre-freeze window, a final-week surge, revert rate about 0.55%) and is
calibrated so the naive final-week revert risk ratio is about 1.6, the value the
pilot measured. Only the mechanism differs:

  pressure     merging in the final week raises every PR's risk (review is thinner)
  composition  features are riskier and are the PRs that get pulled into the final week
  congestion   risk rises with the number of same-day merges in the same component

The script reports which diagnostics tell the worlds apart, at population scale
(many cycles) and at Kubernetes scale (27 cycles), plus the interventional answer
"move this final-week PR to week 7" that release teams actually care about.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(2027)
DAYS = 84            # 12 weeks before freeze
PER_CYCLE = 960
P_FEATURE = 0.25
N_COMPONENTS = 6
PULL = 0.07          # share of PRs pulled into the final 7 days by the deadline
PULL_LAST2 = 0.5     # of those, share landing in the final 48h (pilot: ~half of final-week excess)
BASE = {"bug": 0.0045, "feature": 0.0045}

WORLDS = {
    # name: (feature share among pulled PRs, final-week multiplier, congestion slope, feature/bug risk)
    "pressure":    dict(pull_feat=P_FEATURE, phi=1.6, gamma=0.0,  feat_rr=1.0),
    "composition": dict(pull_feat=0.95,      phi=1.0, gamma=0.0,  feat_rr=3.6),
    "congestion":  dict(pull_feat=P_FEATURE, phi=1.0, gamma=0.7,  feat_rr=1.0),
}


def simulate(world, n_cycles, rng):
    w = WORLDS[world]
    n = n_cycles * PER_CYCLE
    cycle = np.repeat(np.arange(n_cycles), PER_CYCLE)
    pulled = rng.random(n) < PULL
    p_feat = np.where(pulled, w["pull_feat"], (P_FEATURE - PULL * w["pull_feat"]) / (1 - PULL))
    feature = rng.random(n) < p_feat
    pulled_day = np.where(rng.random(n) < PULL_LAST2, rng.integers(0, 2, n), rng.integers(2, 7, n))
    day = np.where(pulled, pulled_day, rng.integers(0, DAYS, n))   # 0 = last day
    comp = rng.integers(0, N_COMPONENTS, n)
    df = pd.DataFrame({"cycle": cycle, "day": day, "comp": comp, "feature": feature})
    df["k"] = df.groupby(["cycle", "day", "comp"]).day.transform("size") - 1   # same-day, same-component merges
    df["late"] = df.day < 7
    p = BASE["bug"] * np.where(df.feature, w["feat_rr"], 1.0)
    p = p * np.where(df.late, w["phi"], 1.0) * (1 + w["gamma"] * df.k)
    df["p"] = p
    df["y"] = rng.random(n) < p
    # Counterfactual: same PR merged on a week-7 day (k drawn from an ordinary day's load).
    k7 = rng.poisson(PER_CYCLE * (1 - PULL) / DAYS / N_COMPONENTS, n)
    df["p_cf"] = BASE["bug"] * np.where(df.feature, w["feat_rr"], 1.0) * (1 + w["gamma"] * k7)
    return df


def rr(df, mask_t, mask_c, col="y"):
    return df.loc[mask_t, col].mean() / df.loc[mask_c, col].mean()


def mh_rr(df, strata, treat="late"):
    """Mantel-Haenszel risk ratio of `treat` vs not, stratified by `strata`."""
    num = den = 0.0
    for _, g in df.groupby(strata):
        t = g[treat]
        n1, n0 = t.sum(), (~t).sum()
        if n1 == 0 or n0 == 0:
            continue
        N = len(g)
        num += g.y[t].sum() * n0 / N
        den += g.y[~t].sum() * n1 / N
    return num / den if den else np.nan


def diagnostics(df, col="y"):
    d = df.assign(y=df[col])
    d["kcat"] = np.minimum(d.k, 12)
    late = d[d.late].copy()
    late["busy"] = late.k > late.groupby(["cycle", "day"]).k.transform("median")
    return {
        "naive RR (final week vs weeks 2-12)": rr(d, d.late, ~d.late),
        "RR within kind (MH)": mh_rr(d, "feature"),
        "RR within kind and same-day load (MH)": mh_rr(d, ["feature", "kcat"]),
        "final week, same day: busy vs quiet component (MH)": mh_rr(late, ["cycle", "day"], "busy"),
    }


def main():
    rows = {}
    for world in WORLDS:
        df = simulate(world, 2000, RNG)
        diag = diagnostics(df, "p")              # population values: use true risks, no sampling noise
        late = df[df.late]
        diag["do(move final-week PR to week 7): risk ratio"] = late.p_cf.mean() / late.p.mean()
        rows[world] = diag
    pop = pd.DataFrame(rows).round(2)

    # Kubernetes-scale power: how often does each diagnostic come out the "right" side of 1.2?
    reps = 300
    hits = {w: {"naive RR > 1.2": 0, "RR within kind > 1.2": 0, "busy vs quiet component > 1.2": 0}
            for w in WORLDS}
    for world in WORLDS:
        for _ in range(reps):
            d = diagnostics(simulate(world, 27, RNG))
            hits[world]["naive RR > 1.2"] += d["naive RR (final week vs weeks 2-12)"] > 1.2
            hits[world]["RR within kind > 1.2"] += d["RR within kind (MH)"] > 1.2
            hits[world]["busy vs quiet component > 1.2"] += d["final week, same day: busy vs quiet component (MH)"] > 1.2
    power = (pd.DataFrame(hits) / reps).round(2)

    out = ["Population values (2,000 synthetic cycles; true risks):", pop.to_string(), "",
           f"Kubernetes scale (27 cycles x {PER_CYCLE} PRs, {reps} replications): "
           "share of replications where the statistic exceeds 1.2", power.to_string()]
    text = "\n".join(out)
    print(text)
    with open(__file__.replace("sim_three_worlds.py", "out/sim_three_worlds.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    with np.errstate(divide="ignore", invalid="ignore"):
        main()
