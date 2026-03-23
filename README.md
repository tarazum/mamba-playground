# Mamba Playground

Research playground to validate whether Mamba-style State Space Models (SSMs) are worth
integrating into production systems: **agent-pool**, **SimpleAgentsOrchestrator**, and
**Trend Signal Engine**.

## Core Question

> Transformer = reasoning engine (keep for prompts, planning, code)
> Mamba/SSM = state engine (use for routing, monitoring, anomaly detection)

Can a learned SSM routing/monitoring layer meaningfully outperform static heuristics?

## Project Structure

```
mamba-playground/
├── CLAUDE.md                  ← project context for Claude Code
├── requirements.txt           ← CPU deps (works on any machine)
├── requirements-gpu.txt       ← GPU deps (mamba-ssm + Blackwell build notes)
├── data/
│   └── generators.py         ← synthetic event stream generators
├── core/
│   └── ssm.py                ← CPU-compatible SSM (pure PyTorch, no CUDA kernels)
├── experiments/
│   ├── 01_setup_check.py     ← environment detection + speed benchmarks
│   ├── 02_event_classify.py  ← classify event streams: normal / degrading / stuck
│   ├── 03_anomaly_detect.py  ← detect anomaly onset (secondary gate)
│   └── 04_routing_sim.py     ← SSM routing vs round-robin / least-busy (PRIMARY gate)
├── docs/
│   └── integration_plan.md   ← how results map to production decisions
└── results/                  ← experiment outputs (JSON)
```

## Quick Start

```bash
# 1. Install CPU deps (works on Windows/Mac/Linux, no GPU required)
pip install -r requirements.txt

# 2. Check your environment
python experiments/01_setup_check.py

# 3. Run experiments in order
python experiments/02_event_classify.py
python experiments/03_anomaly_detect.py
python experiments/04_routing_sim.py
python experiments/05_tse_velocity.py
```

## GPU Setup (RTX 5070 / Blackwell sm_120)

Pre-compiled mamba-ssm wheels do not include sm_120 support. Build from source:

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
pip install causal-conv1d --no-binary causal-conv1d
pip install mamba-ssm --no-binary mamba-ssm
```

Requires: CUDA 12.6+, PyTorch 2.6+, gcc/nvcc in PATH.
See `experiments/01_setup_check.py` for automated detection and instructions.

## Success Criteria

| Experiment | Gate | Decision |
|------------|------|----------|
| Exp 04 (routing) | SSM beats round-robin by >10% | Integrate SSMRouter into agent-pool |
| Exp 04 (routing) | Within 5% of round-robin | Static strategies sufficient |
| Exp 03 (anomaly) | Detects stuck workers >3 events early | Integrate into health loop + SimpleAO Guard |
| Exp 03 (anomaly) | Comparable to threshold rules | Static rules sufficient |

See `docs/integration_plan.md` for detailed production integration instructions per project.

## Architecture

All experiments use a CPU-compatible MinimalSSM (`core/ssm.py`) that reproduces the
Mamba selective scan in pure PyTorch — no CUDA kernels, no mamba-ssm dependency.
On GPU, replace `SelectiveSSMBlock` with `mamba_ssm.Mamba` for 10–50x speedup.
The interface is identical.
