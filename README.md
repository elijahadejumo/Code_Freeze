# The Code-Freeze Rush in Kubernetes

Replication package for *The Code-Freeze Rush in Kubernetes*, a full paper submitted to CauSE 2027 (Causal Methods in Software Engineering, co-located with ICSE 2027).

## The study

Kubernetes ships three releases a year on a fixed calendar. Before each release, a **code freeze** stops new work. The weeks between the freeze and the release are used to test and stabilize the code. The deadline triggers a rush, as contributors try to get their code in before it.

We ask whether becoming ready to merge late causes more problems found after release, and through which path.

| | |
|---|---|
| **Units** | All non-bot pull requests merged into `kubernetes/kubernetes` that became ready within 12 weeks before a code freeze: 25,579 pull requests across 27 releases (v1.11–v1.37). |
| **Ready** | When a pull request has both the `lgtm` and `approved` labels. We use this time, not the merge time, because the merge queue can delay a merge past the deadline. |
| **Treatment** | Ready **late**, in the final 7 days before the freeze. |
| **Comparison** | Ready **early**, 2 to 12 weeks before the freeze. |
| **Problem** | A later pull request that reverts the original, or fixes it. |
| **Outcome** | A problem found after release. |

Late readiness can lead to a problem after release along two paths:

```
            Composition (size, kind, SIGs, author)
               │                        │
               ▼                        ▼
         Ready late ──code path──▶ Problems in the code ──▶ Problem found
               │                                              after release
               └────soak path───▶ Soak time ─────────────────▶
```

- **Code path:** late code is worse. It contains more problems, so more are found after release.
- **Soak path:** late code is no worse, but it has less **soak time**, the time between merge and release. A problem it contains has less time to be found before release.
- **Composition** (size, kind, SIGs, author) affects both when a change becomes ready and how many problems it has. It is a confounder, and we adjust for it.

The two paths imply different results. Under the code path, late pull requests have more problems in total. Under the soak path alone, they have as many problems as early ones, but more of them are found after release.

**RQ1.** How does the code-freeze rush unfold?
**RQ2.** Does late code fail more often, or are its failures found later?

## Layout

| Folder | Contents |
|---|---|
| `crawl/` | GitHub GraphQL crawl of pull request metadata and label timelines. |
| `analysis/` | Builds the dataset, validates outcome links, runs the analyses and draws the figures. |
| `pilot/` | The git-only feasibility pilot. `pilot_rush_reverts.py` is still used: it detects and links reverts for `build_dataset.py`. The other scripts are the Rust data-adequacy screen and a synthetic illustration. |
| `data/` | The freeze calendar, the analysis dataset, outcome links and validation coding (see below). |
| `results/` | Every table the paper reports, as CSV, plus run logs. |
| `paper/` | The LaTeX source and figures. |
| `figures/` | The pilot's figure (not used in the paper). |

### Data

| File | Contents |
|---|---|
| `data/k8s_freeze_calendar.csv` | Code freeze (UTC), thaw and release date for each release, taken from the release team's schedules in `kubernetes/sig-release`. v1.31 uses its moved freeze (24 July 2024). |
| `data/analysis.csv.gz` | One row per pull request: readiness, merge and release times; size, kind, SIG, milestone, priority and author labels; review measures; and outcomes. |
| `data/links_reverts.csv`, `data/links_fixes.csv` | Revert → original and fix → original links, before checks. |
| `data/links_*_checked.csv` | The same links with the automatic checks: whether a revert swaps the original's additions and deletions, and whether a fix uses cue words and shares a file with the original. |
| `data/validation/human_census_*.csv` | Hand coding of every fix link the outcomes use. The key links neutral ids to pull requests, and the answers are y / n / ?. The coder did not see dates or whether a pull request was late. |
| `data/raw/` | The raw crawl (about 256 MB). It is not in the repository; rebuild it with `crawl/` (see below). |

## Reproducing the results

Requirements: Python 3.13 with pandas 2.3, numpy 2.3, scipy 1.16, statsmodels 0.14, scikit-learn 1.7 and matplotlib 3.10.

Run single-threaded. With multithreaded BLAS, the bootstrap runs in parallel can take about an hour instead of about ten minutes.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

### From the included dataset (no GitHub token needed)

Each outcome is reported with two link sets: **all links** (`--links all`) and **checked links** (`--links human`). Checked links are verified reverts plus the fixes the hand coding confirmed.

```bash
python analysis/run_rqs.py --links all          # the rush's impact and the four explanations
python analysis/run_rqs.py --links human        #   ... with checked links (results/links_human/)
python analysis/soak_decomposition.py --links human   # 180 days from merge: total vs before/after release
python analysis/soak_evidence.py --links human        # discovery curve, observed vs expected share by week
python analysis/timing_series.py                # the rush and what changes as the freeze nears
python analysis/make_paper_figure.py            # paper/figures/fig_{rush,changes,soak}.pdf
```

`soak_decomposition.py --horizon 90` and `--horizon 365` give the other window lengths (all links only; the hand coding covers links within 180 days of merge).

### Where each part of the paper comes from

The scripts number the analyses RQ1–RQ5 as in the original analysis plan. The paper groups them into two research questions:

| Paper | Script | Results |
|---|---|---|
| RQ1: the rush, the v1.31 moved deadline, the move to three releases a year | `timing_series.py`, `run_rqs.py` | `timing_rush_*.csv`, `timing_v131_weeks.csv`, `rq3_v131.csv`, `rq3_dose*.csv` |
| RQ1: the impact (risk ratios by time before freeze, placebo deadline) | `run_rqs.py` | `rq1_bins.csv`, `rq1_rr.csv` |
| RQ1: what changes as the freeze nears | `timing_series.py` | `timing_by_bin.csv` |
| RQ2: composition | `run_rqs.py` | `rq2_composition.csv`, `rq2_balance.csv`, `rq2_overlap_*.csv` |
| RQ2: pressure on review | `run_rqs.py` | `rq3_pressure.csv` |
| RQ2: a crowded merge queue | `run_rqs.py` | `rq4_congestion.csv` |
| RQ2: the freeze itself | `run_rqs.py` | `rq5_freeze_response.csv` |
| RQ2: soak time | `soak_decomposition.py`, `soak_evidence.py` | `exploratory_soak_*.csv`, `exploratory_evidence_*.csv`, `timing_discovery.csv` |

Checked-link versions of the `rq*` files are in `results/links_human/`. The soak-time analyses and the descriptive series were added after we saw the planned results, so their files are marked `exploratory_` or `timing_`.

### Full rebuild from GitHub

This needs a GitHub token in a `.env` file in the project root (`GITHUB_TOKEN=...`). Never commit that file; `.gitignore` excludes it.

```bash
git clone --filter=tree:0 --bare --single-branch --branch master \
    https://github.com/kubernetes/kubernetes.git k8s.git

python crawl/crawl_prs.py k8s.git              # PR metadata; add --shards N --shard i to split
python crawl/refetch_timelines.py              # full, verified label timelines
python analysis/build_dataset.py k8s.git       # data/analysis.csv.gz and the outcome links
python analysis/auto_validate_links.py         # automatic revert and fix checks
python analysis/prepare_human_census.py OUT.json   # page data for hand-coding the fix links
python analysis/score_human_census.py          # after coding: precision for late vs early
```

GitHub's GraphQL API silently truncates `timelineItems` when many pull requests are queried at once. It gives no error and no `hasNextPage`. `refetch_timelines.py` therefore queries in batches of 10. It checks every pull request by replaying its label events against its current labels, and retries failures one at a time.

A new crawl reflects GitHub as it is on that day. Pull requests deleted since our crawl in October 2026 will be missing, and labels may have changed.
