# Mamba Playground — Claude Code Context

## Purpose

A research playground to validate whether Mamba-3 (State Space Model) is worth
integrating into our production systems: agent-pool, SimpleAgentsOrchestrator, and
Trend Signal Engine.

Key question per project:
- agent-pool: can a learned SSM router outperform static strategies (round-robin, least-busy)?
- SimpleAO: can SSM-based monitoring predict pipeline stalls before they happen?
- TSE: can an SSM maintain trend velocity state across weekly runs more efficiently than reprocessing?

## Core Insight Being Tested

> Transformer = reasoning engine (keep for prompts, planning, code)
> Mamba/SSM = state engine (use for routing, monitoring, anomaly detection)

These are complementary, not competing.

## Project Structure

```
mamba-playground/
├── CLAUDE.md
├── README.md
├── requirements.txt          ← CPU deps (works on any machine)
├── requirements-gpu.txt      ← GPU deps (mamba-ssm, requires CUDA + Blackwell notes)
├── data/
│   └── generators.py         ← synthetic event stream generators (no external data needed)
├── core/
│   └── ssm.py                ← minimal CPU-compatible SSM (PyTorch, no CUDA kernels)
├── experiments/
│   ├── 01_setup_check.py     ← environment detection, speed test, GPU notes
│   ├── 02_event_classify.py  ← classify event streams: normal / degrading / stuck
│   ├── 03_anomaly_detect.py  ← detect anomaly onset in streaming agent events
│   ├── 04_routing_sim.py     ← compare SSM routing vs round-robin / least-busy
│   └── 05_tse_velocity.py    ← classify trend velocity: rising / peaking / declining / noise
├── docs/
│   ├── README.md
│   ├── experiment_results_analysis.md  ← full results with diagrams and integration decisions
│   └── integration_plan.md             ← code-level integration guide per project
└── results/                  ← experiment outputs (JSON + plots)
```

## Running Experiments

```bash
# 1. Install CPU deps (works everywhere)
pip install -r requirements.txt

# 2. Check your environment first
python experiments/01_setup_check.py

# 3. Run experiments in order
python experiments/02_event_classify.py
python experiments/03_anomaly_detect.py
python experiments/04_routing_sim.py
python experiments/05_tse_velocity.py
```

## GPU Setup (RTX 5070 / Blackwell)

See `requirements-gpu.txt` and the GPU section of `experiments/01_setup_check.py`
for Blackwell-specific (sm_120) build instructions.

## What "Success" Looks Like

Experiment 04 (routing simulation) is the primary gate:
- If SSM routing beats round-robin by >10% on composite quality → candidate for agent-pool router
- If SSM routing is within 5% of round-robin → static strategies are good enough

Experiment 03 (anomaly detection) is the secondary gate:
- If SSM detects stuck workers >3 events before timeout → candidate for agent-pool health loop + SimpleAO Guard
- If detection lag is similar to a simple threshold → static rules are good enough

Experiment 05 (TSE velocity) is the TSE gate:
- If SSM beats moving-average baseline by >10% on velocity classification → candidate for TSE velocity tracker
- If comparable → DECAY_WINDOW_DAYS heuristic is sufficient

**Gates passing on synthetic data means: proceed to Phase 3 (real trace validation), not production integration.**

Do not integrate Mamba into production systems before running all five experiments AND validating on real event traces.
