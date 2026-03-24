# Mamba Playground — Integration Plan

> How experiment results map to production decisions in agent-pool, SimpleAO, and TSE.

## Gate Conditions and Phase 2 Outcomes

| Gate | Condition | Phase 2 Result | Decision |
|------|-----------|---------------|----------|
| **Primary** (Exp 04) | SSM routing beats round-robin by >10% avg quality | **NOT MET** (+3.4%) | Switch default to **sticky** routing; no SSM router integration |
| **Secondary** (Exp 03) | SSM detects stuck workers >3 events before tuned threshold | **PASSED** (+3.7 steps) | Phase 3: validate on real agent-pool event traces |
| **TSE** (Exp 05) | SSM beats moving average by >10% on velocity classification | **PASSED** (+49.3%) | Phase 3: validate on real TSE weekly history |

**Do not integrate before running all five experiments AND validating on real traces.**

---

## Project Mapping

### agent-pool (`C:\tarazu\projects\agent-pool`)

The highest-value integration target. Agent pool is a routing and health system — exactly what SSMs are designed for.

**Exp 04 gate NOT MET — immediate action:**

Switch the default routing strategy in agent-pool from `cost-aware` to `sticky`.
- Sticky is the most robust strategy tested: only -2.6% on distribution shift vs -14.4% for SSMQualityPredictor and -13.0% for cost-aware
- Requires no training, no model checkpoint, no new dependencies
- Change one config value in `agent_pool/router.py`

**If Exp 04 gate is re-tested with real traces and passes:**

```
agent_pool/
├── router.py          ← add RoutingStrategy.SSM_LEARNED
│   └── SSMRoutingStrategy: loads trained SSMQualityPredictor checkpoint
│                        maintains rolling event_history buffer (seq_len=32)
│                        predicts quality per worker, routes to argmax
└── workers/
    └── base.py        ← emit WorkerEvent on complete/error (feeds SSM history)
```

Integration steps:
1. Copy `core/ssm.py` into `agent_pool/ssm/` (or install as package)
2. Add `train_quality_predictor.py`: trains SSMQualityPredictor on recorded pool events using masked MSE
3. Modify `Router.select()`: add `SSM_LEARNED` case calling `ssm_predictor.route(history)`
4. Add event history buffer to `WorkerPool`: deque of last 32 WorkerEvent features
5. Serialize trained model checkpoint to `config/ssm_quality_predictor.pt`
6. Load in `build_pool()` when strategy = `ssm_learned`

**If Exp 03 gate passes:**

```
agent_pool/
└── pool.py
    └── _health_loop()  ← integrate SSMAnomalyDetector
        per worker: maintain sliding window of events
        if anomaly_score > 0.7: flag worker for health check immediately
        (currently: only checks ERROR workers every 30s)
```

This tightens the health check loop from 30s polling to event-driven detection, matching the "detect stuck workers >3 events before timeout" gate.

---

### SimpleAgentsOrchestrator (`C:\tarazu\projects\SimpleAgentsOrchestrator`)

Integration focus: **pipeline stall prediction** using `SSMAnomalyDetector`.

**If Exp 03 gate passes:**

SimpleAO runs multi-step pipelines via `sao.yaml`. Each step emits artifacts and status events. An SSM monitoring layer can watch step latency and artifact metadata to predict stalls before hard timeouts fire.

```
SimpleAO pipeline event features (map to POOL_FEATURE_DIM=6):
  [0] step_latency_norm    — normalized latency vs. historical mean
  [1] artifact_size_delta  — change in output artifact size
  [2] retry_count_norm     — retries normalized to [0, 1]
  [3] downstream_pending   — fraction of downstream steps blocked
  [4] llm_token_rate       — tokens/s this step (drop = model overload)
  [5] error_flag           — 1 if last LLM call returned error, else 0
```

Implementation:
1. Add `SimpleAO/guard/ssm_guard.py` implementing `Guard` interface
2. On each step completion: update feature window, call `SSMAnomalyDetector.forward()`
3. If `anomaly_score > threshold`: emit `WARN` event to Guard system (existing mechanism)
4. Guard can then: pause pipeline, alert, or trigger remediation step

The SimpleAO Guard system already handles the action layer — SSM only needs to provide the signal.

---

### Trend Signal Engine (`C:\tarazu\projects\trend-signal-engine`)

Integration focus: **trend velocity tracking** — maintaining state across weekly runs.

Current TSE problem: each weekly run reprocesses all signals from scratch. Expensive, loses temporal context.

**Exp 05 gate PASSED (+49.3%) — validated on synthetic weekly sequences:**

```
TSE Stage 2.5 (between Normalize and Embed):
  SSMVelocityTracker
    input:  normalized signal stream (weekly batch, treated as sequence)
    output: per-signal velocity vector (rising / stable / falling)
    state:  compressed SSM hidden state persisted between weekly runs
```

Implementation:
1. After clustering (Stage 3), extract cluster centroid sequences over time
2. Feed weekly centroid deltas into SSMVelocityTracker
3. Augment scored_trends with `velocity_score` (rising = reward, falling = penalize)
4. Persist SSM hidden state to SQLite (compressed bytes column) between runs

This replaces the current `DECAY_WINDOW_DAYS` heuristic with a learned velocity model.

**Signal feature vector for TSE (map to POOL_FEATURE_DIM=6):**

```python
[0] frequency_norm     — normalized occurrence frequency this week
[1] cluster_density    — density of the cluster this signal belongs to
[2] diversity_score    — cross-source diversity (TSE Stage 4 output)
[3] specificity_score  — phrase specificity (TSE Stage 4 output)
[4] validation_score   — external validation score (TSE Stage 6 output)
[5] week_delta_norm    — normalized change vs. previous week
```

---

## Decision Framework

Phase 2 results (synthetic data) — use this matrix for Phase 3 real-data decisions:

| Exp 04 Result | Exp 03 Result | Exp 05 Result | Recommended Action |
|---------------|---------------|---------------|-------------------|
| SSM wins (>10%) | SSM early detection | SSM wins | Full integration: router + health loop + velocity |
| SSM comparable | SSM early detection | SSM wins | **Current outcome**: switch to sticky routing; validate health loop + velocity on real traces |
| SSM wins (>10%) | Baseline sufficient | SSM wins | Router + velocity only; keep static health checks |
| SSM comparable | Baseline sufficient | SSM wins | Velocity tracker only; static routing + health checks win |
| Any | Any | Comparable | Skip TSE velocity integration; DECAY_WINDOW_DAYS sufficient |

**Current Phase 2 row**: SSM comparable routing + SSM early detection + SSM velocity wins → switch to sticky routing; proceed to Phase 3 for health loop and velocity.

---

## Performance Expectations (CPU baseline)

From Exp 01 speed test benchmarks:

| Config | Expected throughput |
|--------|---------------------|
| seq=32, d=32, batch=32 | ~3,000–8,000 seq/s |
| seq=64, d=32, batch=32 | ~1,500–4,000 seq/s |
| seq=128, d=64, batch=16 | ~500–1,500 seq/s |

For production routing (agent-pool), seq=32 is sufficient. Inference at batch=1 is the bottleneck for online use — expect ~10–50ms/request on CPU, <1ms on GPU.

For TSE velocity tracking (batch processing), seq=52 (weekly), batch=256 is fine on CPU.

---

## Files to Copy to Production

When integration is approved:

| File | Destination | Notes |
|------|-------------|-------|
| `core/ssm.py` | `agent_pool/ssm/ssm.py` | No modification needed |
| `core/ssm.py` | `SimpleAO/guard/ssm_core.py` | Same file, different path |
| Trained `ssm_quality_predictor.pt` | `agent_pool/config/` | Generated by train script (if Exp 04 passes on real data) |
| Trained `ssm_anomaly.pt` | `agent_pool/config/` | Generated by train script (after Phase 3 validation) |

The SSM implementation has zero external dependencies beyond PyTorch — safe to copy.

---

## Mamba-ssm GPU Upgrade Path

When the Blackwell machine (RTX 5070) is available for production:

1. Confirm Exp 04 results are worth the GPU infrastructure cost
2. Build mamba-ssm from source with `TORCH_CUDA_ARCH_LIST="12.0"`
3. Replace `SelectiveSSMBlock` with `mamba_ssm.Mamba` (same interface)
4. Expected speedup: 10–50x on sequences of length 64+
5. Enables real-time routing decisions even for high-throughput pools (>1000 req/s)

The interface is identical — swapping CPU↔GPU backend requires changing one import.
