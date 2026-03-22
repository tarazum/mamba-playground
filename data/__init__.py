"""
data/ — Synthetic event stream generators for mamba-playground experiments.

No external datasets required — all sequences are procedurally generated
from realistic models of agent pool behavior, pipeline events, and trend signals.

Main exports:
    generate_pool_dataset       — Exp 02: pool event classification
    build_anomaly_dataset       — Exp 03: anomaly onset detection
    simulate_routing_episode    — Exp 04: routing simulation
    generate_trend_velocity_dataset — TSE velocity tracking
"""

from data.generators import (
    EventLabel,
    PipelineLabel,
    POOL_FEATURE_DIM,
    PIPELINE_FEATURE_DIM,
    generate_pool_sequence,
    generate_pool_dataset,
    generate_pipeline_sequence,
    generate_pipeline_dataset,
    generate_trend_velocity_dataset,
    simulate_routing_episode,
)

__all__ = [
    "EventLabel",
    "PipelineLabel",
    "POOL_FEATURE_DIM",
    "PIPELINE_FEATURE_DIM",
    "generate_pool_sequence",
    "generate_pool_dataset",
    "generate_pipeline_sequence",
    "generate_pipeline_dataset",
    "generate_trend_velocity_dataset",
    "simulate_routing_episode",
]
