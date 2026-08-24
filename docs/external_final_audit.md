# Mamba Playground — Final External Audit

> **Date:** 2026-08-25 · **Scope:** full repository (code, experiments, saved reports, docs)
> at commit `9bc1b84`, plus independent verification runs.
> **Outcome:** after corrections, **none of the three gates pass**. The project closes as a
> documented negative result. No SSM integration into agent-pool, SimpleAO, or TSE.

---

## 1. Summary

The playground did its job: it asked a precise question, built explicit gates, and produced
an answer. The audit's contribution was catching that two of the three "passed/failed"
verdicts were distorted by how the baselines were treated — the same class of error the
project had already caught and fixed once in Phase 2 (the Exp 03 threshold k-sweep), but
did not apply consistently everywhere.

| Gate | Pre-audit verdict | Post-audit verdict | What changed |
|------|-------------------|--------------------|--------------|
| **Primary** — Exp 04 routing: SSM > round-robin by >10% | NOT MET (+3.4pp) | **NOT MET** (+5.6pp / +4.8pp across seeds) | Cost scale + seed hygiene corrected the numbers; seed replication exposed that shift behaviour is unstable |
| **Secondary** — Exp 03 anomaly: SSM ≥3 steps earlier | PASSED (3.7 steps) | **NOT MET** (1.1 steps) | Lag compared at matched false-positive rates |
| **TSE** — Exp 05 velocity: SSM > baseline by >10pp | PASSED (+49.3pp) | **NOT MET** (+5.2pp) | Baseline threshold tuned on the validation set |

The consistent picture: **the learned SSM never clears its integration gate on any task.**
Where it wins, the margin is small (routing +5pp, velocity +5pp); off-distribution its
behaviour is not just worse but *unpredictable* — the router swings from best strategy to
worst strategy depending on its training seed (§4). Corrected static/tuned baselines are
sufficient everywhere.

---

## 2. Audit method

1. Read every file: `core/ssm.py`, `data/generators.py`, all five experiment scripts,
   all saved reports, all docs.
2. Reproduced committed numbers independently (numpy only): Exp 05 baseline exactly
   (0.5075), Exp 04 static baselines exactly (round-robin 254.84ms / 0.054 / $0.0010,
   least-busy, cost-aware, sticky — byte-identical metrics). The repo is genuinely
   reproducible from seeds.
3. Ran targeted verification computations (slope statistics per class, threshold sweeps
   on the validation split, FP-rate/lag tradeoffs).
4. Fixed the code, re-ran experiments 03/04/05 end-to-end on CPU (torch 2.13), re-ran
   the smoke test suite (18/18 pass).

---

## 3. Findings

### F1 — CRITICAL: Exp 05's headline result was an artifact of an untuned baseline

The `MovingAverageClassifier` ran with `slope_threshold=0.08`, but the generator's
rising/declining trends have a true slope of 0.8/11 ≈ **0.0736 ± 0.0043** per week. The
threshold sat ~1.5σ *above* the signal, so the baseline systematically classified clean
monotonic trends as noise:

| Class | True slope (measured, test set) | MA @ thr=0.08 | MA @ thr=0.06 (val-tuned) |
|-------|--------------------------------|---------------|---------------------------|
| rising | +0.0736 ± 0.0043 | 0.105 | **0.990** |
| peaking | +0.0083 ± 0.0117 (tent) | 0.982 | 0.991 |
| declining | −0.0722 ± 0.0041 | 0.033 | **1.000** |
| noise | +0.0010 ± 0.0226 | 0.856 | 0.789 |
| **overall** | | **0.5075** | **0.9475** |

Tuning the threshold on the same validation split the SSM used — exactly the procedure the
project added to Exp 03 in Phase 2 — moves the baseline from 50.75% to **94.75%**. The SSM's
advantage collapses from **+49.3pp to +5.2pp**, below the 10pp gate.

Aggravating circumstances:

- The project's own Lessons Learned table said *"A single untuned configuration is not a
  fair baseline — always tune on val set."* That lesson was applied to Exp 03 but not Exp 05.
- The analysis doc's explanation of the result was wrong on two counts: it attributed the
  baseline's failure to a "short 3-week window" (the code's first decision branch is a
  regression over **all 12 weeks**), and it illustrated the point with a saturating trend
  (`0.1 … 0.85 0.88 0.90 0.91`) that the generator never produces (rising is linear
  0.1→0.9). The real failure was the threshold, not window myopia.
- The SSM's 100% accuracy is a ceiling result on perfectly separable synthetic data — the
  same condition the project itself dismissed in Exp 02 as "not a useful benchmark."

### F2 — MAJOR: Exp 03's gate verdict compared detectors at unequal operating points

Committed comparison (fixed score threshold 0.5 for both models):

| Model | Lag | FP rate |
|-------|-----|---------|
| SSM | 3.0 steps | **39%** |
| Tuned threshold (k=0.5) | 6.65 steps | **17.7%** |

The SSM "wins by 3.7 steps" while firing on more than twice as many normal sequences —
a more sensitive detector detects earlier by construction. Matching the SSM's operating
point to the baseline's FP rate (SSM threshold 0.80 → FP 11.0% vs baseline 13.0%) gives:

- SSM lag at matched FP: **5.75 steps** → advantage over baseline: **1.1 steps** — below
  the ≥3-step gate. **The secondary gate does not survive an honest comparison.**

Additional problem: the distribution-shift results were reported without their FP rates.
On the shift set (onsets 68–80%, outside the training range) the SSM's FP rate is **100%**
and the threshold baseline's is **52.5%** — the SSM fires before onset in *every* shift
sequence. The doc's claim "shift lag 0.00 — generalises well" was an artifact of an
always-firing detector. (Mechanism: the model learned that anomalies begin by 20–65% of
the sequence; on late-onset sequences it false-alarms while waiting.)

### F3 — MAJOR: Exp 04's evaluation metric had an inert cost term; train/eval seeds overlapped

1. `composite_quality()` normalised cost by `MAX_COST_USD = 1.0` while per-request costs
   are ~$0.001 — every strategy scored ≈0.9998 on the cost term, so the advertised 20%
   cost weight contributed nothing to the verdict. Meanwhile the SSM's *training target*
   (`step_quality`, scale 0.015) was cost-sensitive. The model was optimised for one
   objective and judged by another.
2. The training dataset was built from `default_rng(SEED)` — the same stream the
   evaluation episodes draw from. The first 200 training episodes were exactly the 200
   in-distribution evaluation episodes. Mild leakage; the (clean, seed 43) distribution-shift
   eval already hinted at it: there SSM ≈ round-robin (0.741 vs 0.730).
3. Dead code: the timeout path (`latency > 25000` → 30000ms) was unreachable — degraded
   latency peaks at ~1200ms in these pools; `timeout_rate` was always 0.
4. `all_api_pool()` was defined and tested but never used in any experiment. Notably, in
   an all-paid pool `cost-aware` degenerates by construction to `least-busy` (no free
   workers to prefer), and in the default pool the two produced identical metrics — so
   `cost-aware` was never actually tested as a distinct policy.

> Post-fix note: correcting F3 changed the SSM's numbers in both directions — the
> in-distribution margin rose (+3.4 → +5.6pp) while the specific v2 shift figures became
> invalid (see §4). Leakage corrupts in whichever direction the noise points, which is
> exactly why it must be audited rather than assumed benign.

### F4 — MINOR: documentation drift and overstated claims

- The integration-decision diagram carried stale v1 numbers ("Exp 03 PASSED 0.25 steps",
  "Exp 04 NOT MET +1%") contradicting the tables below them (3.65 steps, +3.4%).
- "We ran four experiments" / "all four experiments" in several places; five exist.
- `integration_plan.md` promised 3,000–8,000 seq/s where Exp 01 measured ~1,008 seq/s.
- "Actionable today: switch agent-pool default strategy to sticky" violated the project's
  own rule that synthetic-data gates only justify Phase 2 validation, not production
  changes — and sticky had won only one of the two tested environments (in-distribution
  it lost to least-busy and SSM).
- The Exp 05 narrative misdescribed the baseline's code (see F1).

---

## 4. Fixes applied

**Code (`experiments/05_tse_velocity.py`)** — added `tune_slope_threshold()` sweeping the
baseline's threshold on the validation set (same split the SSM uses); the report now
contains both the tuned result and the fixed-0.08 v1 reference; the verdict gate uses the
tuned baseline.

**Code (`experiments/03_anomaly_detect.py`)** — added `sweep_detection_thresholds()` and
`pick_fp_matched()`; the report now contains the SSM's threshold/FP/lag sweep and the
FP-matched operating point; the gate verdict is computed at the matched FP rate (raw-0.5
numbers kept for reference).

**Code (`experiments/04_routing_sim.py`)** — `MAX_COST_USD` aligned to the per-call scale
(0.015) so eval and training target weigh cost identically; training episodes moved to a
separate seed stream (`SEED + 1000`); added a third evaluation condition (all-API pool —
cost-regime shift); deltas reported for all three conditions.

**Code (`data/generators.py`)** — removed the unreachable timeout path.

**Docs** — rewrote `experiment_results_analysis.md` around the corrected numbers (all
FP rates now shown, Exp 05 explanation corrected, stale diagram numbers fixed);
`integration_plan.md` marked NOT APPROVED with the corrected throughput table and TSE
verdict; `README.md` / `CLAUDE.md` / `docs/README.md` updated with the closure status and
the Exp 05 gate.

**Verification** — smoke tests: 18/18 pass after changes. Experiments 03, 04, 05 re-run
end-to-end; reports regenerated.

### Re-run results (Exp 05, Exp 03)

**Exp 05 — TSE velocity (final):**

| Model | Accuracy | rising | peaking | declining | noise |
|-------|----------|--------|---------|-----------|-------|
| SSMClassifier | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Slope baseline, **val-tuned** (thr=0.06) | **0.948** | 0.990 | 0.991 | 1.000 | 0.789 |
| Slope baseline, fixed 0.08 (v1, reference) | 0.507 | 0.105 | 0.982 | 0.033 | 0.856 |

Delta vs tuned baseline: **+5.2pp — below the 10pp gate. NOT MET.** The SSM's remaining
edge is concentrated in the noise class (+21pp): the tuned threshold over-fires
rising/declining on noisy sequences. On synthetic data *designed* to favour shape
recognition, a linear-slope classifier is within 5 points of the SSM.

**Exp 03 — anomaly detection (final):**

| Model | AUC | Lag @ own FP | FP rate | Lag @ matched FP |
|-------|-----|--------------|---------|------------------|
| SSMAnomalyDetector | 0.978 | 2.99 | 37.3% | **5.75** (thr=0.80, FP=11.0%) |
| Tuned threshold (k=0.5) | 0.963 | 6.84 | 13.0% | 6.84 (reference) |

FP-matched advantage: **1.1 steps — below the ≥3-step gate. NOT MET.**
Shift set: SSM AUC 0.996 at **FP=100%**; threshold AUC 0.928 at FP=52.5% — neither
detector is production-usable off-distribution without threshold re-calibration.

### Re-run results (Exp 04)

*(numbers from the corrected re-run — see `results/04_routing_report.json`)*

**In-distribution (default pool):**

| Strategy | Quality | Latency | Error | Cost/req |
|----------|---------|---------|-------|----------|
| **SSMQualityPredictor** | **0.876** | **181ms** | 5.0% | $0.0000 |
| least-busy | 0.860 | 222ms | 3.4% | $0.0000 |
| cost-aware | 0.860 | 222ms | 3.4% | $0.0000 |
| sticky | 0.829 | 244ms | 3.5% | $0.0013 |
| round-robin | 0.820 | 255ms | 5.4% | $0.0010 |

Delta vs round-robin: **+5.6pp — below the 10pp gate. NOT MET.**

**Distribution shift (heavy degradation pool, 3/5 workers degrade):**

| Strategy | Quality (seed 42) | Seed-43 replication |
|----------|-------------------|---------------------|
| **SSMQualityPredictor** | **0.877** | **0.705 — worst strategy** (19.3% errors) |
| sticky | 0.798 | 0.797 |
| least-busy | 0.760 | 0.761 |
| cost-aware | 0.748 | 0.748 |
| round-robin | 0.717 | 0.717 |

**Cost-regime shift (all-API pool, added by this audit):** seed 42: SSM 0.741 (best, just
above least-busy/cost-aware 0.738); seed 43: SSM **0.558 — parks on `gpt-4`, the slowest,
most expensive worker**, far below round-robin (0.685). `cost-aware` ≡ `least-busy` here
*by construction* (no free worker to prefer). Static strategies reproduce identically
across seeds — same policies, same physics; the learned router does not.

**The most consequential correction — and what the replication added.** The v2 report's
"SSM drops −14.4% on shift, sticky is most robust, switch agent-pool to sticky" was
computed with seed leakage (training episodes = eval episodes), so its specific numbers
were invalid. The corrected seed-42 run at first *looked* like a clean reversal (SSM best
everywhere, +16pp on the degradation shift — it parks on `ollama-0`, the fastest free
non-degrading worker: 181ms / 5% err / $0). A full replication with training seed 43
(report kept at `results/_replication_seed43/04_routing_report.json`) showed that this
was a lucky draw: the same architecture, data size and hyperparameters parks on a
*degrading* worker in the heavy pool and on the worst worker in the all-API pool.

Final reading: the in-distribution gain is real but consistently small (+5.6pp / +4.8pp,
below the 10pp gate), and the out-of-distribution behaviour is a **training-seed coin
flip between best strategy and worst strategy**. For a production router, unpredictable
fragility is disqualifying regardless of the in-dist margin — which closes Exp 04 more
firmly than either v2 or the first corrected run did.

---

## 5. Final gate verdicts

| Gate | Condition | Result | Verdict |
|------|-----------|--------|---------|
| Primary (Exp 04) | SSM > round-robin by >10% composite quality | +5.6pp / +4.8pp (two seeds); shift unstable best↔worst | **NOT MET** |
| Secondary (Exp 03) | SSM ≥3 steps earlier at matched FP | +1.1 steps | **NOT MET** |
| TSE (Exp 05) | SSM > tuned baseline by >10pp | +5.2pp | **NOT MET** |

**Decision: project closed. No Mamba/SSM component is integrated into any production
system. The negative result is the deliverable.**

---

## 6. Opinion: is Mamba a fit for the Trend Signal Engine?

**No — not as TSE is architected today.** Four reasons, in order of weight:

1. **Sequence length kills the architectural case.** Mamba exists to make *long* sequences
   cheap (linear-time scan, compressed state over thousands of steps). TSE's velocity
   question is 12 weekly points. At L=12 a GRU, a gradient-boosted tree on slope features,
   or — as this audit verified — a tuned linear-slope classifier sits within ~5pp of the
   SSM *on synthetic data deliberately shaped to favour the SSM*. The one experiment that
   directly tested Mamba's alternative (LSTM, Exp 02) showed parity at 20× less CPU.
2. **The bar is now measured, and it's high.** 94.8% accuracy from a stateless classifier
   with two interpretable parameters. The SSM must beat that by >10pp *on real TSE data*
   to justify training infrastructure, checkpoint management, and loss of interpretability.
   Nothing in this playground suggests it can.
3. **The "persist state instead of reprocessing" story doesn't pay at this scale.**
   Reprocessing 12–52 weekly aggregates is microseconds of work; there is no compute
   problem for a compressed state to solve. State persistence becomes interesting when
   history is huge or arrival is streaming — neither describes weekly TSE runs.
4. **Data volume and explainability.** A 17k-parameter learned model needs labelled
   velocity examples TSE likely doesn't have, and "why was this trend marked declining?"
   is answerable from a slope classifier and not from an SSM hidden state.

**When to reconsider (explicit triggers):** any of these changes the calculus —
(a) TSE moves to streaming/incremental ingestion (daily or hourly, many signals,
sequences in the hundreds-plus); (b) real labelled history shows shape-driven failures of
the tuned slope classifier (saturation, multi-phase rise-decline, seasonality); (c) the
routing/monitoring tasks get real traces and re-open Phase 2. If a reconsideration ever
happens, the comparison set should include GRU/TCN — Mamba's edge over them only appears
at sequence lengths this project never tested.

For today: implement velocity as the tuned piecewise-slope classifier (one file, no
training, interpretable), or keep `DECAY_WINDOW_DAYS`. Do not integrate the SSM.

---

## 7. What remains valuable from this project

- **The negative result itself** — "we tested it properly and static/tuned-simple won"
  is worth more than a hunch; it should stop any future re-litigation without new evidence.
- **The routing simulator** (`data/generators.py`: `WorkerState`, `simulate_routing_episode`)
  — a cheap offline harness for evaluating *any* agent-pool routing policy before it ships.
- **`core/ssm.py`** — a clean, dependency-light reference implementation of the Mamba
  selective scan, useful for teaching or prototyping; the smoke tests cover its shapes.
- **The Lessons Learned table** — every row of it now earned the hard way, including the
  new one from this audit: *tune every baseline on the same val split the model uses,
  and compare detectors at matched operating points, or the gates will lie to you.*
- **The process** — gates defined before results, external review, phase-2 corrections,
  and now a final audit that flipped two verdicts. This is what a small research project
  should look like.

---

## 8. Closure

All five experiments complete and corrected. All reports regenerated from the fixed code.
All gates failed; integration plan marked not approved; this audit is the final document.

**Reopen conditions:** real event traces from agent-pool or real weekly history from TSE,
plus at least one trigger from §6. Until then — closed.
