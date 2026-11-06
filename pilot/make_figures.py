"""Figure 1 for the paper: the rush, the revert gradient, and the v1.31 deadline move.

Reads pilot/out/merges_in_window.csv (from pilot_rush_reverts.py); writes figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "pdf.fonttype": 42,
})


def wilson(x, n, z=1.96):
    p = x / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c, c - h, c + h


def main():
    w = pd.read_csv(HERE / "out" / "merges_in_window.csv", parse_dates=["merged_at"],
                    dtype={"release": str})
    b = w[~w.is_revert & ~w.is_bot].copy()
    n_rel = b.release.nunique()

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.85), gridspec_kw={"width_ratios": [1, 1, 1.25]})

    # (a) merges per release by week before freeze
    ax = axes[0]
    wk = b.groupby("week_before").size() / n_rel
    weeks = np.arange(12, 0, -1)
    vals = wk.reindex(weeks).values
    colors = [ORANGE if x == 1 else BLUE for x in weeks]
    ax.bar(range(12), vals, color=colors, width=0.78, edgecolor="white", linewidth=0.8,
           hatch=["////" if x == 1 else "" for x in weeks])
    ax.set_xticks(range(12), [str(x) for x in weeks])
    ax.set_xlabel("weeks before code freeze")
    ax.set_ylabel("merged PRs per release")
    ax.text(11, vals[-1] + 3, f"{vals[-1]:.0f}", ha="center", va="bottom", color=INK, fontsize=6.5)
    base = vals[:8].mean()
    ax.text(3.5, base + 22, f"weeks 5–12: ~{base:.0f}/week", ha="center", va="bottom", color=INK2, fontsize=6.5)
    ax.set_title(f"(a) The rush ({n_rel} releases, v1.11–v1.37)", loc="left")

    # (b) 30-day revert rate by distance to freeze
    ax = axes[1]
    bins = [("9–12\nwk", b.week_before.between(9, 12)), ("5–8\nwk", b.week_before.between(5, 8)),
            ("3–4\nwk", b.week_before.between(3, 4)), ("2\nwk", b.week_before == 2),
            ("3–7\ndays", (b.week_before == 1) & (b.days_to_freeze >= 2)),
            ("last\n48h", b.days_to_freeze < 2)]
    ref = b[b.week_before >= 2]
    ref_rate = 100 * ref.reverted30.mean()
    ax.axhline(ref_rate, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.text(-0.4, 0.08, f"dashed: weeks 2–12 ({ref_rate:.2f}%)", color=INK2, fontsize=6, va="bottom")
    for i, (lab, m) in enumerate(bins):
        x, n = b.reverted30[m].sum(), m.sum()
        c, lo, hi = (100 * v for v in wilson(x, n))
        col = ORANGE if i >= 4 else BLUE
        ax.errorbar(i, c, yerr=[[c - lo], [hi - c]], fmt="o", ms=4, color=col, ecolor=col,
                    elinewidth=1.2, capsize=0, mec="white", mew=0.8)
    ax.set_xticks(range(len(bins)), [l for l, _ in bins])
    ax.set_ylabel("reverted within 30 days (%)")
    ax.set_ylim(0, 2.0)
    ax.set_title("(b) Revert rate, 95% Wilson CI", loc="left")

    # (c) v1.31: the deadline moved, the spike moved with it
    ax = axes[2]
    lo, hi = pd.Timestamp("2024-06-23", tz="UTC"), pd.Timestamp("2024-07-27", tz="UTC")
    r = b[(b.release == "1.31")]
    daily = r.set_index("merged_at").resample("D").size().reindex(
        pd.date_range(pd.Timestamp("2024-06-24", tz="UTC"), pd.Timestamp("2024-07-23", tz="UTC"), freq="D"), fill_value=0)
    ax.bar(daily.index, daily.values, width=0.8, color=BLUE, edgecolor="white", linewidth=0.5)
    orig, new, ann = (pd.Timestamp(s, tz="UTC") for s in ("2024-07-10", "2024-07-24T02:00", "2024-06-28"))
    ax.axvline(orig, color=INK2, lw=0.9, ls=(0, (3, 2)))
    ax.axvline(new, color=ORANGE, lw=1.4)
    ax.axvline(ann, color=INK2, lw=0.6, ls=":")
    top = daily.max()
    ax.text(orig, top * 1.02, "original\nfreeze", ha="center", va="bottom", fontsize=6, color=INK2)
    ax.text(new, top * 1.02, "actual\nfreeze", ha="center", va="bottom", fontsize=6, color=INK)
    ax.text(ann, top * 1.02, "move\nannounced", ha="center", va="bottom", fontsize=6, color=INK2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, top * 1.35)
    ax.set_xticks(pd.to_datetime(["2024-07-01", "2024-07-10", "2024-07-17", "2024-07-24"]).tz_localize("UTC"),
                  ["Jul 1", "Jul 10", "Jul 17", "Jul 24"])
    ax.set_ylabel("merged PRs per day (UTC)")
    ax.set_title("(c) v1.31: freeze moved two weeks later", loc="left")

    fig.tight_layout(pad=0.4, w_pad=1.0)
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig1_rush.{ext}", dpi=300)
    print("wrote", FIG / "fig1_rush.pdf")


if __name__ == "__main__":
    main()
