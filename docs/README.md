# Mamba Playground — Documentation

## Documents

| Document | Purpose |
|----------|---------|
| [experiment_results_analysis.md](experiment_results_analysis.md) | Full results analysis with diagrams, lessons learned, and integration decisions |
| [integration_plan.md](integration_plan.md) | Code-level integration guide: what to copy where, with snippets |
| [project_review_feedback.md](project_review_feedback.md) | Independent review of the project goal, experiment quality, current results, and recommended next steps |

## Quick Reference

### Experiment Order

```
01_setup_check.py    → verify environment, Blackwell GPU notes, CPU speed test
02_event_classify.py → SSM vs LSTM on pool event classification (sanity check)
03_anomaly_detect.py → SSM vs threshold baseline, detection lag (secondary gate)
04_routing_sim.py    → SSM vs static routing strategies (PRIMARY gate)
05_tse_velocity.py   → SSM vs moving average for trend velocity tracking (TSE gate)
```

### Gate Decisions (Phase 2 Results)

See [experiment_results_analysis.md](experiment_results_analysis.md) for full analysis and [integration_plan.md](integration_plan.md) for integration guide.

| Exp | Gate | Outcome | Action |
|-----|------|---------|--------|
| Exp 04 routing | SSM >10% over round-robin | **NOT MET** (+3.4%) | Switch default to **sticky** routing; no SSM integration |
| Exp 03 anomaly | SSM >3 steps earlier than tuned threshold | **PASSED** (+3.7 steps) | Phase 3: validate on real agent-pool event traces |
| Exp 05 TSE velocity | SSM >10% over moving average | **PASSED** (+49.3%) | Phase 3: validate on real TSE weekly history |
