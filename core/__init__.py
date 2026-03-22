"""
core/ — Mamba-style SSM implementations (CPU-compatible, pure PyTorch).

Main exports:
    SSMClassifier       — sequence classification
    SSMAnomalyDetector  — per-timestep anomaly scoring
    SSMRouter           — worker routing from event history
    StackedSSM          — raw stacked SSM blocks (for custom heads)
    SelectiveSSMBlock   — single Mamba-style block
"""

from core.ssm import (
    SelectiveSSMBlock,
    StackedSSM,
    SSMClassifier,
    SSMAnomalyDetector,
    SSMRouter,
)

__all__ = [
    "SelectiveSSMBlock",
    "StackedSSM",
    "SSMClassifier",
    "SSMAnomalyDetector",
    "SSMRouter",
]
