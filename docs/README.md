# Mamba Playground — Documentation

## Documents

| Document | Purpose |
|----------|---------|
| [experiment_results_analysis.md](experiment_results_analysis.md) | Full results analysis with diagrams, lessons learned, and integration decisions |
| [integration_plan.md](integration_plan.md) | Code-level integration guide: what to copy where, with snippets |

## Quick Reference

### Experiment Order

```
01_setup_check.py    → verify environment, Blackwell GPU notes, CPU speed test
02_event_classify.py → SSM vs LSTM on pool event classification (sanity check)
03_anomaly_detect.py → SSM vs threshold baseline, detection lag (secondary gate)
04_routing_sim.py    → SSM vs static routing strategies (PRIMARY gate)
05_tse_velocity.py   → SSM vs moving average for trend velocity tracking (TSE gate)
```

### Gate Decisions

See [integration_plan.md](integration_plan.md) for the full decision matrix.

Short version:
- Exp 04 SSM > round-robin by >10% → integrate router
- Exp 03 SSM detects >3 steps earlier → integrate health loop
- Otherwise → static strategies win, no integration needed
