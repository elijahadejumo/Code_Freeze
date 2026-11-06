"""Figures for the short paper. Orange marks late (final-week) pull requests throughout.

  fig_rush.pdf     Fig. 1  the rush: (a) PRs per day before freeze, (b) v1.31's moved deadline,
                           (c) the final week across releases
  fig_changes.pdf  Fig. 2  as the freeze approaches: problems after release, reverts, size,
                           features, review, days to release (RQ1 bins)
  fig_soak.pdf     Fig. 3  soak time: (a) how fast problems are found, late vs early,
                           (b) share found after release, observed vs predicted from soak time

Reads results/timing_*.csv (timing_series.py) and the week-bin file written by soak_evidence.py
(check B). The default is the primary outcome: verified reverts plus hand-confirmed fixes.

Usage:
    python analysis/make_paper_figure.py [--gradient results/exploratory_evidence_gradient_h180_human.csv]
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "paper" / "figures"
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


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight")
    fig.savefig((OUT / name).with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {OUT / name}")


def fig_rush():
    daily = pd.read_csv(RES / "timing_rush_daily.csv").query("day_before <= 21")
    v131 = pd.read_csv(RES / "timing_v131_weeks.csv")
    rel = pd.read_csv(RES / "timing_rush_by_release.csv", parse_dates=["freeze"])
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
    cnt = v131.set_index("week_before_actual_freeze").prs
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
    save(fig, "fig_rush.pdf")


def fig_changes():
    t = pd.read_csv(RES / "timing_by_bin.csv")
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
    save(fig, "fig_changes.pdf")


def fig_soak(gradient):
    d = pd.read_csv(RES / "timing_discovery.csv")
    g = pd.read_csv(gradient)
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
    save(fig, "fig_soak.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gradient", type=Path, default=RES / "exploratory_evidence_gradient_h180_human.csv")
    style()
    fig_rush()
    fig_changes()
    fig_soak(ap.parse_args().gradient)
