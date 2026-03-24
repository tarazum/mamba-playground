# Real Trace Schema — Phase 3 Validation

> This document defines the expected event fields for each Phase 3 validation target.
> No real traces are committed here. This schema is the contract that data collection must satisfy
> before re-running Exp 03 and Exp 05 on real data.

---

## Agent-Pool Events (Exp 03 — Anomaly Detection, Exp 04 — Routing)

Maps to `POOL_FEATURE_DIM = 6` in `data/generators.py`.

| Index | Field | Type | Range | Description |
|-------|-------|------|-------|-------------|
| 0 | `worker_type` | float | 0–1 | Normalised worker class (CLI=0, OLLAMA=0.5, API=1.0) |
| 1 | `request_type` | float | 0–1 | Normalised request category (CODE=0, ANALYSIS=0.33, CREATIVE=0.67, FAST=1.0) |
| 2 | `latency_norm` | float | 0–1 | Request latency normalised by `max_latency_ms` (default 2000ms) |
| 3 | `success` | float | 0 or 1 | 1 = completed without error |
| 4 | `cost_norm` | float | 0–1 | Cost normalised by `max_cost_usd` (default $0.10) |
| 5 | `retry_count_norm` | float | 0–1 | Retries normalised by max retries (default 3) |

**Collection point:** `agent_pool/pool.py` — one row per request completion event.

**Minimum for Phase 3:**
- ≥5,000 events per worker
- At least one anomaly episode (degrading or stuck worker) with known onset time
- Known onset step index per sequence for detection-lag evaluation

**Sequence construction:** sliding window of last `seq_len=64` events per worker, sampled every N completions.

---

## Pipeline Events (SimpleAO Guard — future)

Maps to `POOL_FEATURE_DIM = 6` in `data/generators.py` (reused schema).

| Index | Field | Type | Range | Description |
|-------|-------|------|-------|-------------|
| 0 | `step_latency_norm` | float | 0–1 | Step latency normalised by historical mean |
| 1 | `artifact_size_delta` | float | –1–1 | Change in output artifact size (bytes) normalised |
| 2 | `retry_count_norm` | float | 0–1 | Step retries normalised by max |
| 3 | `downstream_pending` | float | 0–1 | Fraction of downstream steps currently blocked |
| 4 | `llm_token_rate` | float | 0–1 | Tokens/s this step normalised by baseline rate |
| 5 | `error_flag` | float | 0 or 1 | 1 if last LLM call returned an error |

**Collection point:** `SimpleAgentsOrchestrator` — one row per pipeline step completion.

---

## TSE Weekly Signals (Exp 05 — Velocity Tracking)

Maps to `(n_weeks, 3)` tensor used by `generate_trend_velocity_dataset`.

| Index | Field | Type | Range | Description |
|-------|-------|------|-------|-------------|
| 0 | `cluster_size_norm` | float | 0–1 | Normalised cluster member count this week |
| 1 | `diversity_norm` | float | 0–1 | Cross-source diversity score (TSE scorer output) |
| 2 | `total_score_norm` | float | 0–1 | Normalised weighted total score (TSE scorer output) |

**Collection point:** `trend-signal-engine/storage/store.py` — `scored_trends` table already persists per-run data.
Query: `SELECT label, diversity_score, total_score, run_id, created_at FROM scored_trends ORDER BY label, created_at`.

**Minimum for Phase 3:**
- ≥10 weekly runs stored in the DB (covers the 12-week sequence length)
- ≥50 distinct trend labels tracked across all runs
- Ground-truth velocity labels are not available from TSE — use human-labelled subset OR
  define proxy labels from observed multi-week direction (rising: last score > first score by >20%)

**Sequence construction:** for each trend label, collect the weekly `[cluster_size_norm, diversity_norm, total_score_norm]`
vector across all runs in chronological order. Pad missing weeks with the previous week's value or 0.

---

## Logging Checklist for Data Collection

- [ ] agent-pool: log `worker_id`, `request_id`, `worker_type`, `request_type`, `latency_ms`, `success`, `cost_usd`, `retries`, `timestamp`
- [ ] agent-pool: tag sequences with `anomaly_onset_step` when a worker degrades or gets stuck (for lag evaluation)
- [ ] TSE: confirm `scored_trends` table is being populated on every weekly run (storage/store.py batch insert is in place)
- [ ] TSE: add `run_date` column if not already present (needed for chronological sequence assembly)
