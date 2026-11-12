# The Code-Freeze Rush in Kubernetes: replication package

Code and data for *The Code-Freeze Rush in Kubernetes* (CauSE 2027). The study covers 25,579  pull requests merged into `kubernetes/kubernetes` that became ready (both `lgtm` and `approved`) within 12 weeks before a code freeze, across 27 releases (v1.11–v1.37).

| | |
|---|---|
| **Treatment** | Ready **late**: in the final 7 days before the freeze. |
| **Comparison** | Ready **early**: 2 to 12 weeks before the freeze. |
| **Problem** | A later pull request that reverts or fixes the original. |
| **Outcome** | A problem found in the 90 days after the release. |
| **Link sets** | **All links**: every revert and fix link. **Checked links**: reverts that undo the original's changes, plus fixes that use cue words, share a file with the original, and were confirmed by hand. |

## Layout

```
analyze.py      reproduces every table and figure in the paper from data/
build.py        rebuilds data/ from GitHub and the git history (optional)
data/           the dataset, outcome links and hand coding
results/        the tables, as CSV
figures/        the paper's Figures 2-4 (PDF and PNG)
```

## Reproducing the paper

Requirements: Python 3.13 and the packages in `requirements.txt`. 

```bash
pip install -r requirements.txt
python analyze.py
```

The run takes about 20 minutes on a laptop, almost all of it in the bootstrap. It writes `results/` and `figures/`, and prints every table. Bootstrap intervals use a fixed seed, so a re-run gives identical numbers.

### Where each result comes from

Files in `all_links/` and `checked_links/` exist for both link sets. Below, `*/` stands for either folder.

| Paper | File |
|---|---|
| **RQ1: the rush** | |
| Fig. 2(a): pull requests per day before the freeze | `results/rush_daily.csv` |
| Fig. 2(b): v1.31's moved deadline | `results/rush_v131_weeks.csv` |
| Fig. 2(c): final week vs an average earlier week (1.8× / 2.5×) | `results/rush_by_release.csv` |
| Share of late pull requests after the move to three releases a year (+5.2 pp) | `*/three_releases_a_year.csv` (`final_week_share`, `post_kep2572`) |
| **RQ1: the impact** | |
| Fig. 3: problems, reverts, size, features, review and days to release, by time before the freeze | `results/by_time_before_freeze.csv` |
| Risk ratios, late vs early, and the placebo deadline | `*/impact.csv` |
| Change in the gap after the move to three releases a year | `*/three_releases_a_year.csv` (`gap_y3_post_release`) |
| v1.31: problems in each deadline week | `*/v131_problems.csv` |
| **Assumptions (Sec. III-C)** | |
| Overlap of the propensity to be late; covariate balance after weighting | `*/composition_overlap.csv`, `*/composition_balance.csv` |
| **RQ2: Table I** | |
| What lands late (composition); the 30-day revert gap after reweighting | `*/composition.csv` |
| Pressure on review: within author, and through review measures | `*/review_pressure.csv` |
| A crowded merge queue (final week, and weeks 2–12) | `*/merge_queue.csv` |
| The freeze response | `*/freeze_response.csv` |
| **RQ2: soak time** | |
| Total vs timing: 180 days from merge, problems in total, before and after release | `*/soak_total_vs_timing.csv` |
| Fig. 4(a): discovery speed, late vs early | `results/discovery_curve.csv` |
| Timing prediction: observed vs predicted share found after release | `checked_links/soak_prediction.csv` |
| Fig. 4(b): the same, by week before the freeze | `checked_links/soak_by_week.csv` |
| Future work: the test across releases with different freeze-to-release times | `*/soak_across_releases.csv` |
| **Hand coding (Sec. III-B)** | |
| Share of fix links confirmed, overall and for late vs early pull requests | `results/link_check_precision.csv` |

## Data

| File | Contents |
|---|---|
| `data/k8s_freeze_calendar.csv` | Code freeze (UTC), thaw and release date for each release, from the release team's schedules in `kubernetes/sig-release`. v1.31 uses its moved freeze (24 July 2024). |
| `data/analysis.csv.gz` | One row per pull request: readiness, merge and release times; size, kind, SIG, milestone, priority and author; review measures; and outcomes (all links). |
| `data/links_reverts.csv` | Revert → original links. `inverse` and `verified` record whether the revert swaps the original's additions and deletions. |
| `data/links_fixes.csv` | Fix → original links. `strict` marks cue words ("regression", "broke", ...), and `file_overlap` marks a shared file. |
| `data/validation/human_census_key.csv` | The hand-coded fix links under neutral ids, in random order. |
| `data/validation/human_census_answers.csv` | The coder's answers (y / n / ?). The coder saw titles, descriptions, labels and files, but not dates or whether a pull request was late. |

The data come from a crawl on 28 October 2026. Problems are counted only within windows that had closed by then.

