# Mamba Playground — Documentation

## Documents

| Document | Purpose |
|----------|---------|
| [integration_plan.md](integration_plan.md) | How experiment results map to agent-pool, SimpleAO, and TSE production decisions |

## Quick Reference

### Experiment Order

```
01_setup_check.py   → verify environment, Blackwell GPU notes, CPU speed test
02_event_classify.py → SSM vs LSTM on pool event classification
03_anomaly_detect.py → SSM vs threshold baseline on anomaly detection (secondary gate)
04_routing_sim.py   → SSM vs static routing strategies (PRIMARY gate)
```

### Gate Decisions

See [integration_plan.md](integration_plan.md) for the full decision matrix.

Short version:
- Exp 04 SSM > round-robin by >10% → integrate router
- Exp 03 SSM detects >3 steps earlier → integrate health loop
- Otherwise → static strategies win, no integration needed
