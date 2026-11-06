# Pressure, Composition, or Congestion? Causal effects of release deadlines (CauSE 2027)

**Venue:** CauSE 2027 (Causal Methods in Software Engineering), co-located with ICSE 2027, Dublin.
Site: https://causality-software-engineering.github.io/cause-workshop-2027/

- **Submission: 13 Nov 2026 (AoE)**. Notification 11 Dec 2026. Camera-ready 29 Jan 2027.
- Full paper 8 pp. Position/short paper 5 pp. Tool paper 5 pp. Double-anonymous review, ICSE 2027 format (check the template and whether references count toward the limit).
- **Track: full paper (8 pp), Kubernetes only** (decided 2026-10-28). Fallback: cut to a 5-pp short paper if the Nov 6 checkpoint in `PLAN.md` is missed.

## Verdict: worth pursuing, with three adjustments

**Why it works**
1. **The phenomenon is real and measurable from git alone.** Across 27 cycles, the final week carries 2× the normal merge volume. The 30-day revert rate in the last 48 hours is twice the baseline (RR 2.02, 95% CI 1.24–3.29). See `figures/fig1_rush.pdf`.
2. **There is a real natural experiment.** In v1.31 the freeze was moved two weeks later. The spike moved with it: 11 merges in the 24 hours before the old date, 36 before the new one.
3. **There is a real motivating case where the counterfactual was almost observed.** #122541 merged 29 minutes before the v1.30 freeze and was reverted. It re-landed in week 1 of v1.31 and stayed (`01_motivating_example.md`).
4. **It fits CauSE well.** The question is counterfactual, pressure is a mediation question, and congestion is *interference between units*, a SUTVA violation. That framing will appeal to this program committee.

**What to change**
1. **Add a fourth story: freeze response.** During freeze, maintainers revert rather than fix forward. #122541 had a known one-line fix and was reverted anyway. So "reverted" is partly *caused by the freeze policy*, independent of code quality, and reviewers will raise it. Fix: use outcomes measured after the freeze ends (post-release bug fixes, patch-release cherry-picks), or compare revert-vs-fix-forward decisions directly.
2. **Reverts alone are too rare.** There are 149 reverts among 25,880 PRs. The simulation shows that at Kubernetes scale the congestion diagnostic gives about 30% false signals with reverts only. Add more frequent outcomes: follow-up fix PRs that reference the original, `kind/regression` and `kind/flake` issues, and post-merge CI failures.
3. **Correct facts in the pitch.** Code freeze falls in week **8–13**, not always week 12. Freeze-to-release runs about 3–7 weeks. In 2019 the freeze-to-thaw window was 8–12 days. The "merge bot needs time" claim needs a citation. A citable substitute from the official release-phases doc: *"Due to the Prow queue, this may mean that approved PRs merge shortly after the deadline."* This also means merge time ≠ readiness time, so use lgtm/approve timestamps for exposure.

## Sources of identifying variation

| Mechanism | Design | Data |
|---|---|---|
| Composition | Stratify or match on kind, size, SIG, KEP-tracked. Bunching estimator for the "pulled-forward" excess mass. | PR labels (GitHub API/GraphQL) |
| Pressure | Author fixed effects plus cycle fixed effects. Mediators: review rounds, time from last push to lgtm, number of reviewers. Dose from policy: code slush (v1.13), freeze-week variation (8–13), 4→3 releases per year (v1.22), v1.31 postponement, AoE deadline (v1.35). | GitHub timeline events, `data/k8s_freeze_calendar.csv` |
| Congestion | Exposure mapping: same-day merges touching the same packages or SIG, holding time-to-freeze fixed. Load shocks outside deadlines, e.g. the unexplained 76-merge burst on 24–27 Jun 2024. | git, Prow/Tide history (availability for older cycles needs checking) |
| Freeze response | Revert vs fix-forward choice conditional on a regression being found; outcomes measured after freeze. | revert PR text, cherry-pick PRs |

Negative controls: placebo deadlines, docs-only and bot PRs.

## Ecosystem selection criteria (draft for the Method section)

An ecosystem qualifies only if it meets all of C1–C3 (the deadline must be a credible exogenous shock), plus enough of C4–C9 to support the designs.

| # | Criterion | Why the causal design needs it |
|---|---|---|
| C1 | Time-based releases with a **hard merge cutoff announced in advance** | The deadline must not move because code is late. Otherwise the code causes the deadline, and causation runs the wrong way. |
| C2 | **≥ 20 repeated deadlines** | Cycle fixed effects and inference clustered by cycle |
| C3 | Cutoff dates recoverable from **primary sources** | Exposure must be measured, not guessed |
| C4 | Timestamps for **readiness** (approval) and **merge** | Separates author/reviewer behaviour from queue delay (congestion) |
| C5 | Change-type signal (feature vs bug labels, blueprint/bug trailers) | Composition controls |
| C6 | Linkable quality outcomes **with enough events** (reverts/backouts, `Fixes:` links, regression labels) | Power. Reverts alone are too rare (see the simulation). |
| C7 | Stable contributor identities across cycles | Within-author comparisons |
| C8 | Documented changes to deadline policy | Dose variation |
| C9 | Observable merge queue / CI | Congestion exposure |

Exclusion: releases driven by features, deadlines that slip when work isn't ready (e.g. Ethereum network upgrades), and repos without PR-level history.

### Decision rule for adding Rust (fixed 2026-10-28, before any Rust data was examined)

Rust is included if and only if it passes these **data-adequacy** checks. The direction of its results plays no part.
1. Beta-cut dates recoverable from primary sources (version-bump commits or release notes) for ≥ 20 cycles.
2. ≥ 80% of revert PRs linkable to the PR they revert.
3. ≥ 100 linked reverts inside the 6-week pre-cut windows, so the gradient's confidence interval is informative.

Pre-stated reading of the outcomes:
- (a) Weak rush and weak gradient: consistent with a high cost of missing driving the effect in Kubernetes.
- (b) Rush plus gradient: the effect generalises beyond Kubernetes.
- (c) Gradient without a rush: points away from pressure, toward scrutiny or freeze response at branch cut.

If Rust fails the checks, the paper says so in one sentence and stays Kubernetes-only.

**Outcome (2026-10-28, `pilot/rust_adequacy.py`, no Rust results computed):**
- [1] PASS: 59 beta cuts (1.13–1.99) consistent with RELEASES.md.
- [2] **FAIL: 277/430 revert PRs linked (64.4%)**.
- [3] PASS: 194 linked reverts inside windows.

Why [2] failed, across the 153 unlinked reverts:
- 73 have only a descriptive title ("Revert the `LazyConst` PR").
- 67 quote a title that doesn't exactly match the original.
- 12 cite a bare PR number the git-only linker missed.
- 1 is from another repo.

The failure comes from git-only linking, which can't see PR bodies, not from missing data. The Kubernetes linker uses the same git-only method and reaches 85% (622/732).
**Decision (2026-10-28): accept the failure. Rust is excluded and the paper is Kubernetes-only.** The paper reports that Rust was screened and excluded on revert linkability.

## Folder

- `01_motivating_example.md`: real vignette, scale evidence, three-worlds table, paragraph ready for the paper
- `figures/fig1_rush.{pdf,png}`: (a) rush, (b) revert gradient, (c) v1.31 deadline move
- `data/k8s_freeze_calendar.csv`: freeze, thaw and release dates for v1.11–v1.38, from `kubernetes/sig-release`
- `pilot/pilot_rush_reverts.py`: rush and revert pilot (git only, no token needed); outputs in `pilot/out/`
- `pilot/sim_three_worlds.py`: synthetic observational-equivalence demo
- `pilot/make_figures.py`: builds Figure 1
- `crawl/crawl_prs.py`: GraphQL crawl of labels, files, reviews and lgtm/approve timeline for the 32,980 PRs merged since 2018. Writes `data/raw/prs_graphql.jsonl` and resumes if stopped.
- `crawl/refetch_timelines.py`: re-fetches every PR's label timeline in small batches with pagination and verifies it against the PR's current labels. Needed because batched GraphQL queries silently truncate timelines (see PLAN.md, Deviations).
- `crawl/refetch_timeline_tail.py`: superseded by `refetch_timelines.py`; its output is used only for PRs the verified re-fetch lacks.
- `analysis/build_dataset.py`: builds the units, readiness exposure, Y1–Y3, covariates, mediators and congestion exposures. Writes `data/analysis.csv.gz` and the link files.
- `analysis/run_rqs.py`: runs RQ1–RQ5 as specified in `PLAN.md` and writes `results/`.
- `analysis/sample_validation.py`: draws the 50 + 50 link samples for manual precision coding (`data/validation/`).
- `.env`: holds `GITHUB_TOKEN`. Never commit it.

Full run, once the crawl finishes:
```
python crawl/refetch_timelines.py        # or 4 workers: --shards 4 --shard 0..3
python analysis/build_dataset.py k8s.git
python analysis/sample_validation.py
python analysis/run_rqs.py --boot 400
```

Reproduce:
```
git clone --filter=tree:0 --bare --single-branch --branch master https://github.com/kubernetes/kubernetes.git k8s.git
python pilot/pilot_rush_reverts.py k8s.git && python pilot/sim_three_worlds.py && python pilot/make_figures.py
```

## Pilot limitations (state these honestly)

- Reverts are linked by title, branch name and "This reverts commit" heuristics: 622 of 732 revert PRs are linked. Coverage of manual reverts is unvalidated.
- Freeze times before v1.22 are approximated from "EOD PST".
- The within-author comparison is descriptive and has no controls for kind or size yet.
- The next step is a GitHub token for labels and review events. GraphQL over about 26k PRs is a 1–2 day crawl.
