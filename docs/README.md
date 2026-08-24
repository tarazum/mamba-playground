# Mamba Playground — Documentation

## Documents

| Document | Purpose |
|----------|---------|
| [external_final_audit.md](external_final_audit.md) | Final external audit: findings, verified numbers, fixes applied, closure decision |
| [experiment_results_analysis.md](experiment_results_analysis.md) | Full results analysis with diagrams, lessons learned, and integration decisions |
| [integration_plan.md](integration_plan.md) | Code-level integration guide — **not approved**; kept for reference if gates ever pass |
| [project_review_feedback.md](project_review_feedback.md) | Independent review of the project goal, experiment quality, current results, and recommended next steps |

## Quick Reference

### Experiment Order

```
01_setup_check.py    → verify environment, Blackwell GPU notes, CPU speed test
02_event_classify.py → SSM vs LSTM on pool event classification (sanity check)
03_anomaly_detect.py → SSM vs tuned threshold baseline, FP-matched lag (secondary gate)
04_routing_sim.py    → SSM vs static routing strategies (PRIMARY gate)
05_tse_velocity.py   → SSM vs tuned slope baseline for trend velocity (TSE gate)
```

### Gate Decisions

**Final outcome (project closed): all gates failed on corrected methodology — no integration.**

- Exp 04: SSM +5.6pp/+4.8pp over round-robin across seeds — below the 10% integrate bar; shift behaviour unstable across training seeds
- Exp 03: SSM only ~1 step earlier at matched FP rate (gate ≥3) → static rules sufficient
- Exp 05: SSM +5.2pp over tuned slope baseline (gate >10pp) → tuned classifier sufficient
