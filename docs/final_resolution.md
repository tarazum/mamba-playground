# Mamba Playground — Final Resolution

> **The research question:** Should Mamba-style State Space Models (SSMs) be integrated into
> agent-pool, SimpleAgentsOrchestrator, or Trend Signal Engine?
>
> **The answer:** No — not in v1. Build the products first with simple heuristics.
> SSMs are not required, not production-ready, and not validated on real data.
> Revisit only if specific, measurable failure conditions are triggered after launch.

---

## What We Did

Five synthetic experiments tested whether a CPU-compatible Mamba-style SSM could outperform
static heuristics across three use cases:

| Exp | Question | Result |
|-----|----------|--------|
| 01 | Environment ready? | ✅ CPU feasible (1,008 seq/s at seq=32) |
| 02 | Classify pool health states | ✅ Both SSM and LSTM hit 100% — data too separable to rank |
| 03 | Detect anomaly onset earlier | ✅ Gate PASSED (+3.7 steps vs tuned threshold) — but 39% FP, hard variant fragile |
| 04 | Route requests better than static strategies | ❌ Gate NOT MET (+3.4%, threshold was >10%) |
| 05 | Classify trend velocity | ✅ Gate PASSED (+49.3% easy) — but hard variant collapses to random chance |

Two rounds of independent review and two experimental design improvements (Phase 2) were applied.
The results held.

---

## Final Decision Per Product

### agent-pool

#### Routing

**Decision: No SSM. Use sticky routing.**

| What was tested | Result |
|----------------|--------|
| SSMQualityPredictor vs round-robin | +3.4% in-distribution (gate requires >10%) |
| SSMQualityPredictor on distribution shift | −14.4% (fragile, overfit to training pattern) |
| Sticky routing on distribution shift | −2.6% (most robust of all strategies tested) |

The SSM does not improve routing meaningfully on synthetic data and degrades significantly
when worker behaviour shifts. Sticky routing — stay with the current worker until it errors,
then rotate — is the most robust strategy tested across both pool configurations.

**Status: ✅ Implemented.** `agent_pool/router.py` default changed to `sticky`.

---

#### Anomaly Detection / Health Loop

**Decision: No SSM for v1. Use simple threshold on error rate + latency.**

| What was tested | Result |
|----------------|--------|
| SSMAnomalyDetector vs tuned threshold (k=0.5) | SSM detects 3.7 steps earlier |
| SSM false-positive rate | 39% on normal sequences |
| SSM on distribution-shift set (onset outside training range) | FP rate = 100% (always-on) |

The SSM detects anomalies earlier on clean synthetic data, but fires on 100% of out-of-range
sequences before any anomaly occurs. A model that is always-on under distribution shift is not
safe for a production health loop. The tuned threshold (k=0.5) detects everything at 3.85
steps lag with 52.5% shift FP rate — also too high for v1.

**For v1:** implement a simple rule in `_health_loop()`:
```
if worker.error_rate_last_N > 0.15 OR latency_spike > 2x_baseline:
    flag for immediate health check
```
This costs nothing to implement, has predictable behaviour, and can be tuned with real data.

**Revisit condition:** if the simple threshold misses >20% of real incidents after launch,
revisit SSMAnomalyDetector on recorded traces (infrastructure is ready in mamba-playground).

---

### SimpleAgentsOrchestrator

#### Guard System (Pipeline Stall Detection)

**Decision: No SSM. Existing guard system is sufficient for v1.**

SimpleAO already has a working guard system with liveness monitoring and auto-recovery.
The SSM anomaly detector was proposed as an upgrade — but:

- It depends on agent-pool anomaly detection being validated first (it isn't)
- The 39% FP rate would produce noisy alerts in a pipeline guard context
- SimpleAO's artifact-based state (output file = stage complete) already provides a natural
  stall signal without any learned model

**For v1:** use the existing guard as-is. No changes required.

**Revisit condition:** same as agent-pool anomaly — after real operational data exists and the
simple guard demonstrably misses stalls.

---

### Trend Signal Engine

#### Trend Velocity Tracking

**Decision: No SSM. Keep DECAY_WINDOW_DAYS heuristic for v1.**

| What was tested | Result |
|----------------|--------|
| SSMClassifier vs MovingAverage — easy synthetic data | SSM 100% vs MA 50.7% (+49.3%) |
| SSMClassifier vs MovingAverage — hard variant (noise×4, 25% missing weeks, slope×0.4) | SSM 26.0% vs MA 27.6% — both near random chance (25%) |

The 100% easy accuracy is a separability artifact: the synthetic easy dataset has clean,
unambiguous trends. Under realistic conditions (higher noise, missing data, weaker signals)
both models collapse to random guessing. Neither model is ready.

The current `DECAY_WINDOW_DAYS` heuristic is good enough for v1. It is transparent,
tunable, and does not require training data that does not yet exist.

**Revisit condition:** after ≥10 weekly TSE runs are accumulated in the `scored_trends` DB table,
run Exp 05 on real sequences. If the moving average demonstrably misclassifies velocity
on >30% of manually-labelled trends, revisit SSMVelocityTracker.
(Storage is already writing the right data — no extra instrumentation needed.)

---

## What the Playground Did Right

The playground answered the research question before any SSM code was written into production.
The answer is "not yet" — which is worth knowing. Specific things it prevented:

- Routing: prevented wiring `SSMQualityPredictor` into agent-pool (it degrades −14.4% under realistic conditions)
- Anomaly: prevented deploying a 100% FP rate model into a health loop
- Velocity: prevented replacing a working heuristic with a model that collapses on noisy data

The research question is now closed for v1. Reopen it only when trigger conditions are met.

---

## The Actual Build Order

```
1. Finish agent-pool
   - sticky routing ✅ done
   - simple threshold health loop (not SSM)
   - event logging (passive data collection for future Phase 3)
   - clean invocation API for SimpleAO

2. Integrate agent-pool into SimpleAO
   - SimpleAO uses pool for all LLM calls
   - existing guard handles anomaly detection for v1

3. Pilot TSE with SimpleAO + agent-pool
   - TSE pipeline stages mapped to SimpleAO stages
   - DECAY_WINDOW_DAYS for velocity (sufficient for v1)
   - scored_trends DB already accumulating weekly history

4. Operate and collect data (passive — no extra work)
   - agent-pool logs routing events and errors automatically
   - TSE writes scored_trends on every run automatically

5. Only if trigger conditions are met: Phase 3 SSM validation
   - Re-run Exp 03 on real agent-pool traces
   - Re-run Exp 05 on real TSE sequences
   - mamba-playground infrastructure is ready and waiting
```

---

## Trigger Conditions Summary

These are the specific, measurable signals that would justify reopening the SSM question:

| Product | Component | Trigger |
|---------|-----------|---------|
| agent-pool | Health loop | Simple threshold misses >20% of real error incidents |
| agent-pool | Routing | Sticky routing produces >10% worse quality than the best alternative on ≥1,000 real requests |
| SimpleAO | Guard | Existing guard misses pipeline stalls that human review would have caught, >2x per week |
| TSE | Velocity | DECAY_WINDOW_DAYS causes wrong velocity classification on >30% of manually-labelled trend histories |

If none of these conditions are triggered, the current heuristics are working and SSMs add no value.

---

## Archived Research

All experimental code, results, and analysis remain in `mamba-playground` for future reference:

| File | Contents |
|------|----------|
| `docs/experiment_results_analysis.md` | Full results with Phase 2 improvements and hard variant data |
| `docs/integration_plan.md` | Code-level integration guide (for when/if trigger conditions are met) |
| `docs/real_trace_schema.md` | Phase 3 data schema contract — field specs and collection points |
| `results/03_anomaly_report.json` | SSM anomaly detector: AUC, lag, FP rate, operating points |
| `results/04_routing_report.json` | Routing quality comparison across all strategies |
| `results/05_tse_velocity_report.json` | Velocity classification: easy and hard variant results |
| `core/ssm.py` | CPU-compatible SSM implementation — ready to copy when needed |

The SSM implementation (`core/ssm.py`) requires only PyTorch. No changes are needed to use it
in a future integration — copy the file, load a checkpoint, call `.forward()`.
