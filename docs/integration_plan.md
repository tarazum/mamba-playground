# Mamba Playground — Integration Plan

> How experiment results map to production decisions in agent-pool, SimpleAO, and TSE.
>
> **STATUS: NOT APPROVED (2026-08).** All gates failed after the final audit corrected
> baselines and comparison methodology (see `external_final_audit.md`). This plan is
> kept for reference in case a future phase passes the gates on real trace data.

## Gate Conditions (from CLAUDE.md)

| Gate | Condition | Decision |
|------|-----------|----------|
| **Primary** (Exp 04) | SSM routing beats round-robin by >10% avg quality | Integrate SSMRouter into agent-pool |
| **Primary** (Exp 04) | Within 5% of round-robin | Keep static strategies; SSM not worth overhead |
| **Secondary** (Exp 03) | SSM detects anomalies ≥3 steps earlier than the tuned threshold at a matched FP rate | Integrate SSMAnomalyDetector into agent-pool health loop + SimpleAO Guard |
| **Secondary** (Exp 03) | Detection lag comparable to threshold rules | Static rules sufficient |
| **TSE** (Exp 05) | SSM beats the tuned slope baseline by >10% | Integrate SSMVelocityTracker into TSE |
| **TSE** (Exp 05) | Within 10% of tuned baseline | Tuned slope classifier sufficient |

**Do not integrate before running all five experiments.**

**Final outcome: none of the gates passed → no integration.**

---

## Project Mapping

### agent-pool (`C:\tarazu\projects\agent-pool`)

The highest-value integration target. Agent pool is a routing and health system — exactly what SSMs are designed for.

**If Exp 04 gate passes:**

```
agent_pool/
├── router.py          ← add RoutingStrategy.SSM_LEARNED
│   └── SSMRoutingStrategy: loads trained SSMRouter checkpoint
│                        maintains rolling event_history buffer (seq_len=32)
│                        predicts worker_idx from history on each request
└── workers/
    └── base.py        ← emit WorkerEvent on complete/error (feeds SSM history)
```

Integration steps:
1. Copy `core/ssm.py` into `agent_pool/ssm/` (or install as package)
2. Add `train_router.py` script: trains SSMRouter offline on recorded pool events
3. Modify `Router.select()`: add `SSM_LEARNED` case calling `ssm_router.route(history)`
4. Add event history buffer to `WorkerPool`: deque of last 32 WorkerEvent features
5. Serialize trained model checkpoint to `config/ssm_router.pt`
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

**STATUS: rejected by Exp 05 (final audit).** The tuned slope-threshold classifier
reaches 94.8% on the synthetic velocity task vs 100% for the SSM — a +5.2pp margin,
below the 10% gate. With weekly cadence and 12-step sequences, the SSM's advantages
(long-horizon state compression, linear-time scan) never engage. If TSE ever moves to
high-frequency streaming ingestion over long horizons, re-evaluate; otherwise a tuned
piecewise-slope classifier (one file, no training) is the sufficient tool.

Original sketch, kept for reference if the gate is ever re-run and passed:

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

After running all experiments, use this matrix:

| Exp 04 Result | Exp 03 Result | Recommended Action |
|---------------|---------------|-------------------|
| SSM wins (>10%) | SSM early detection | Full integration: router + health loop + velocity |
| SSM comparable | SSM early detection | Health loop + velocity only; keep static routing |
| SSM wins (>10%) | Baseline sufficient | Router only; keep static health checks |
| SSM comparable | Baseline sufficient | No integration — static strategies win |

---

## Performance Expectations (CPU baseline)

From Exp 01 speed test benchmarks (measured):

| Config | Measured throughput |
|--------|---------------------|
| seq=32, d=32, batch=32 | ~1,000 seq/s |
| seq=64, d=32, batch=32 | ~570 seq/s |
| seq=128, d=64, batch=16 | ~155 seq/s |

For production routing (agent-pool), seq=32 is sufficient. Inference at batch=1 is the bottleneck for online use — expect ~10–50ms/request on CPU, <1ms on GPU.

For TSE velocity tracking (batch processing), seq=52 (weekly), batch=256 is fine on CPU.

---

## Files to Copy to Production

When integration is approved:

| File | Destination | Notes |
|------|-------------|-------|
| `core/ssm.py` | `agent_pool/ssm/ssm.py` | No modification needed |
| `core/ssm.py` | `SimpleAO/guard/ssm_core.py` | Same file, different path |
| Trained `ssm_router.pt` | `agent_pool/config/` | Generated by train script |
| Trained `ssm_anomaly.pt` | `agent_pool/config/` | Generated by train script |

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
