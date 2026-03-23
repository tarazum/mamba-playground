# Mamba Playground Review

## Executive Summary

This repository is a useful research playground, not yet strong evidence for production integration.

The project goal is clear and worthwhile: test whether a Mamba-style State Space Model can add value as a state-tracking layer for routing, anomaly detection, and trend velocity tasks across `agent-pool`, `SimpleAgentsOrchestrator`, and `Trend Signal Engine`.

The strongest part of the project is the framing. It isolates a concrete architectural hypothesis:

- Transformers remain the reasoning layer.
- SSMs are evaluated as sequence-state engines.

That is a defensible product hypothesis. The weakness is that the current experiments mostly validate that the synthetic generators are learnable, not that the model will outperform robust production heuristics on real workloads.

## What The Project Does Well

- The repository is small, understandable, and easy to run.
- The CPU fallback in `core/ssm.py` makes the idea testable without CUDA.
- The project uses explicit gates instead of vague "AI is better" claims.
- Experiment outputs are saved to JSON, which is good for reproducibility.
- The docs already connect results to downstream product decisions instead of stopping at model metrics.

## Assessment Of The Goal

The stated goal is:

> determine whether Mamba-style SSMs are worth integrating into production systems

That is the right question, but the current setup only partially answers it.

Current confidence by area:

- `agent-pool` routing: low confidence
- `agent-pool` anomaly detection: medium confidence for synthetic data, low confidence for production
- `TSE` velocity tracking: medium confidence for the synthetic task, still unproven for real trend data

## Key Findings

### 1. The routing experiment cannot prove that the SSM beats the heuristic

This is the biggest issue in the repository.

In [`experiments/04_routing_sim.py`](../experiments/04_routing_sim.py), the router is explicitly trained by imitation learning to copy the `cost-aware` strategy:

- `build_routing_dataset()` says it will "mimic the cost-aware oracle"
- labels come directly from `simulate_routing_episode(... strategy="cost-aware")`

That means the learned model's ceiling is the heuristic it copies. It cannot demonstrate that the SSM is better than the heuristic because the training target already defines "correct" behavior.

Implication:

- The current result supports "the SSM can reproduce the heuristic"
- It does not support "the SSM can discover a better routing policy"

This makes the project's core routing question only partially answered.

### 2. `least-busy` is not a meaningful routing baseline in the current simulator

In [`data/generators.py`](../data/generators.py), `least-busy` selects `argmin(call_counts)`.

There is no modeled concurrency, active queue depth, or worker occupancy. `call_counts` is only a cumulative count. Under this setup, `least-busy` collapses into balanced assignment and therefore becomes effectively equivalent to round-robin. The saved results confirm identical metrics for both strategies in [`results/04_routing_report.json`](../results/04_routing_report.json).

Implication:

- one of the key baselines is not testing a distinct policy
- routing conclusions are weaker than they appear

### 3. The synthetic data is too easy to support strong integration claims

The repository already hints at this in Experiment 02, and the code confirms it.

Examples:

- Exp 02 reaches 100% for both SSM and LSTM in [`results/02_classify_report.json`](../results/02_classify_report.json)
- Exp 05 reaches 100% for the SSM in [`results/05_tse_velocity_report.json`](../results/05_tse_velocity_report.json)
- anomaly onset is fixed at exactly 50% of the sequence in [`experiments/03_anomaly_detect.py`](../experiments/03_anomaly_detect.py)
- routing evaluation uses fixed worker definitions and one degradation pattern in [`data/generators.py`](../data/generators.py)

These are good toy tasks, but they are not hard enough to justify statements like "integrate into production" without an intermediate validation stage on recorded real traces.

Implication:

- the playground is useful for feasibility
- it is not sufficient for production-readiness decisions

### 4. The anomaly result is promising but the baseline comparison is weak

Experiment 03 is the most interesting result in the repo, but it is still overstated.

Strengths:

- per-timestep scoring matches the problem well
- the SSM clearly learns the synthetic anomaly pattern

Weaknesses:

- the threshold baseline is a single untuned configuration
- anomaly onset is fixed
- normal and anomalous halves are concatenated from separately generated sequences
- the SSM false-positive rate is 15.4%, which is not trivial

So the practical conclusion should be:

- "worth validating on real event streams"

not:

- "ready to integrate into agent-pool and SimpleAO"

### 5. The routing quality metric does not match the full business objective

`composite_quality()` in [`experiments/04_routing_sim.py`](../experiments/04_routing_sim.py) only uses latency and error rate.

But the experiment itself also tracks cost, and the narrative emphasizes cost-aware routing. If routing decisions are supposed to trade off latency, reliability, and spend, then cost should be part of the optimized objective, not just displayed alongside it.

Implication:

- the experiment currently optimizes one objective while discussing another
- routing conclusions may shift if cost is weighted explicitly

### 6. Some documentation and repository references are inconsistent

There are a few maintainability issues:

- [`CLAUDE.md`](../CLAUDE.md) references `core/model.py`, but that file does not exist
- [`CLAUDE.md`](../CLAUDE.md) says "all four experiments", while the repo contains five
- [`README.md`](../README.md) quick-start stops at Experiment 04 even though Experiment 05 is present and discussed elsewhere

These are small, but they reduce trust in the repo's source of truth.

### 7. `Infinity` is written into a JSON report

[`results/03_anomaly_report.json`](../results/03_anomaly_report.json) contains `Infinity` for `mean_lag_steps`.

Python's serializer allows that, but it is not valid strict JSON and may break downstream tooling.

Implication:

- results are less portable than they look
- a string or nullable field would be safer

## Result-By-Result Feedback

### Exp 01

Useful setup validation. Good operational check. No major issues.

### Exp 02

Good smoke test for model correctness. Not useful as a ranking benchmark because the generator is too separable.

### Exp 03

Most promising experiment in the repo. Worth continuing, but only after strengthening the baseline and testing variable anomaly timing and real traces.

### Exp 04

Methodologically flawed for the main research question. It is an imitation benchmark, not a policy-learning benchmark.

### Exp 05

Interesting and directionally plausible. The task matches SSM strengths better than routing does. Still needs validation on actual TSE historical data before any integration decision.

## Recommendations

### Priority 1: Fix the experimental design

1. Replace imitation-only routing with an objective-driven evaluation.
   The SSM should optimize reward directly, or at least predict future latency/error/cost and choose actions from those predictions.

2. Make baselines real competitors.
   Add queue-aware least-busy, cost-latency weighted routing, sticky routing, and a small contextual bandit baseline.

3. Use variable anomaly onset and multiple failure modes.
   Randomize onset position, severity, and duration in Exp 03.

4. Add train/test distribution shift.
   Evaluate on worker pools, latencies, and degradation profiles not seen during training.

### Priority 2: Introduce real data before any production recommendation

1. Capture event traces from `agent-pool`.
   Even a few thousand routing and health events would be more valuable than more synthetic polish.

2. Capture historical TSE weekly sequences.
   The current TSE result is plausible, but the real test is whether long-horizon trajectory shape exists in production data strongly enough to beat hand-tuned decay heuristics.

3. Add replay-style evaluation.
   Use recorded traces to compare "what the heuristic did" vs "what the model would have done" under a fixed event history.

### Priority 3: Tighten engineering quality

1. Make reports strict JSON-safe.
   Replace `Infinity` with `null` plus a `missed_all: true` flag.

2. Clean up doc drift.
   Align `README.md`, `CLAUDE.md`, and `docs/README.md` around the same experiment list and current files.

3. Separate research claims from production decisions.
   Phrase current outputs as "playground findings" or "candidate directions", not integration approvals.

4. Add one minimal test layer.
   At least smoke-test dataset shapes, model forward passes, and report generation.

## Recommended Decision Right Now

Based on the current repository alone:

- Do not use this repo as sufficient evidence to integrate an SSM router into `agent-pool`.
- Do not use this repo as sufficient evidence to integrate the anomaly detector into production yet.
- Use this repo as justification for a second phase based on recorded traces.
- The TSE direction is the most worth pursuing next, because the task structure genuinely favors sequence-state models.

## Bottom Line

This is a solid prototype repository with a clear research thesis and good implementation discipline for a small playground.

Its main limitation is not code quality. It is experimental validity.

If the next step is "collect real traces and rerun stronger baselines", this project has done its job well.
If the next step is "ship the current SSM components into production", the evidence here is not yet strong enough.
