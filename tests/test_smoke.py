"""
Smoke tests — verify shapes, forward passes, and report generation.

Run: python -m pytest tests/test_smoke.py -v
     (or: python tests/test_smoke.py)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pytest

from data.generators import (
    generate_pool_sequence,
    generate_pool_dataset,
    generate_trend_velocity_dataset,
    simulate_routing_episode,
    default_worker_pool,
    heavy_degradation_pool,
    all_api_pool,
    EventLabel,
    POOL_FEATURE_DIM,
)
from core.ssm import (
    SelectiveSSMBlock,
    StackedSSM,
    SSMClassifier,
    SSMAnomalyDetector,
    SSMRouter,
    SSMQualityPredictor,
)


# ── generator shapes ──────────────────────────────────────────────────────────

class TestGeneratorShapes:
    def test_pool_sequence_shape(self):
        seq = generate_pool_sequence(32, EventLabel.NORMAL)
        assert seq.shape == (32, POOL_FEATURE_DIM), seq.shape

    def test_pool_sequence_all_labels(self):
        for label in EventLabel:
            seq = generate_pool_sequence(16, label)
            assert seq.shape == (16, POOL_FEATURE_DIM)
            assert np.isfinite(seq).all(), f"Non-finite values for {label}"

    def test_pool_dataset_shapes(self):
        X, y = generate_pool_dataset(60, seq_len=32, seed=0)
        assert X.shape == (60, 32, POOL_FEATURE_DIM)
        assert y.shape == (60,)
        assert set(y.tolist()) <= {0, 1, 2}

    def test_trend_velocity_shapes(self):
        X, y = generate_trend_velocity_dataset(40, n_weeks=12, seed=0)
        assert X.shape == (40, 12, 3)
        assert y.shape == (40,)
        assert set(y.tolist()) <= {0, 1, 2, 3}

    def test_routing_episode_keys(self):
        m = simulate_routing_episode(n_requests=20, strategy="round-robin", seed=0)
        required = {"avg_latency_ms", "error_rate", "total_cost_usd",
                    "event_history", "decisions", "step_quality"}
        assert required <= set(m.keys()), f"Missing keys: {required - set(m.keys())}"
        assert len(m["event_history"]) == 20
        assert len(m["decisions"])     == 20
        assert len(m["step_quality"])  == 20

    def test_routing_episode_no_infinity(self):
        for strategy in ["round-robin", "least-busy", "cost-aware", "sticky"]:
            m = simulate_routing_episode(n_requests=30, strategy=strategy, seed=1)
            assert np.isfinite(m["avg_latency_ms"])
            assert np.isfinite(m["error_rate"])

    def test_least_busy_differs_from_round_robin(self):
        """Queue-depth least-busy should produce a different distribution than round-robin."""
        rr = simulate_routing_episode(n_requests=200, strategy="round-robin",   seed=42)
        lb = simulate_routing_episode(n_requests=200, strategy="least-busy",    seed=42)
        # call distributions must differ (least-busy routes away from overloaded workers)
        assert rr["call_distribution"] != lb["call_distribution"], \
            "least-busy is identical to round-robin — queue-depth fix not working"

    def test_worker_pool_factories(self):
        for fn in [default_worker_pool, heavy_degradation_pool, all_api_pool]:
            pool = fn()
            assert len(pool) == 5
            m = simulate_routing_episode(n_requests=10, workers=pool, seed=0)
            assert "avg_latency_ms" in m


# ── model forward passes ──────────────────────────────────────────────────────

BATCH, SEQ, DIM = 4, 16, POOL_FEATURE_DIM


class TestModelForwardPasses:
    def _x(self):
        return torch.randn(BATCH, SEQ, DIM)

    def test_selective_ssm_block(self):
        block = SelectiveSSMBlock(d_model=DIM)
        out = block(self._x())
        assert out.shape == (BATCH, SEQ, DIM)
        assert torch.isfinite(out).all()

    def test_stacked_ssm(self):
        model = StackedSSM(d_model=DIM, n_layers=2)
        out = model(self._x())
        assert out.shape == (BATCH, SEQ, DIM)

    def test_ssm_classifier(self):
        model = SSMClassifier(input_dim=DIM, d_model=16, n_classes=3)
        out = model(self._x())
        assert out.shape == (BATCH, 3)
        assert torch.isfinite(out).all()

    def test_ssm_anomaly_detector(self):
        model = SSMAnomalyDetector(input_dim=DIM, d_model=16)
        out = model(self._x())
        assert out.shape == (BATCH, SEQ)
        assert (out >= 0).all() and (out <= 1).all(), "Scores must be in [0, 1]"

    def test_ssm_router(self):
        model = SSMRouter(event_dim=DIM, d_model=16, n_workers=4)
        out = model(self._x())
        assert out.shape == (BATCH, 4)

    def test_ssm_router_route(self):
        model = SSMRouter(event_dim=DIM, d_model=16, n_workers=4)
        x = torch.randn(1, SEQ, DIM)
        idx = model.route(x)
        assert 0 <= idx < 4

    def test_ssm_quality_predictor(self):
        model = SSMQualityPredictor(event_dim=DIM, d_model=16, n_workers=5)
        out = model(self._x())
        assert out.shape == (BATCH, 5)
        assert (out >= 0).all() and (out <= 1).all(), "Quality scores must be in [0, 1]"

    def test_ssm_quality_predictor_route(self):
        model = SSMQualityPredictor(event_dim=DIM, d_model=16, n_workers=5)
        x = torch.randn(1, SEQ, DIM)
        idx = model.route(x)
        assert 0 <= idx < 5

    def test_ssm_quality_predictor_used_in_simulation(self):
        """SSMQualityPredictor should integrate with simulate_routing_episode."""
        model = SSMQualityPredictor(event_dim=DIM, d_model=16, n_workers=5)
        model.eval()
        m = simulate_routing_episode(n_requests=30, strategy="ssm", ssm_model=model, seed=0)
        assert "avg_latency_ms" in m
        assert len(m["decisions"]) == 30


# ── JSON report safety ────────────────────────────────────────────────────────

class TestReportSafety:
    def test_no_infinity_in_detection_lag(self):
        """evaluate_detection_lag must return None, not float('inf'), when all missed."""
        import json
        sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
        # Import via importlib to avoid running main()
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "exp03",
            Path(__file__).parent.parent / "experiments" / "03_anomaly_detect.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        scores = np.zeros((10, 64))  # all zeros — nothing will cross threshold 0.5
        onsets = np.full(10, 32, dtype=np.int32)
        result = mod.evaluate_detection_lag(scores, threshold=0.5, onsets=onsets)

        assert result["mean_lag_steps"] is None, "Expected None, not inf"
        assert result["missed_all"] is True

        # Must be JSON-serialisable without error
        serialised = json.dumps(result)
        parsed = json.loads(serialised)
        assert parsed["mean_lag_steps"] is None


if __name__ == "__main__":
    # Run without pytest
    import traceback
    suites = [TestGeneratorShapes, TestModelForwardPasses, TestReportSafety]
    passed, failed = 0, 0
    for suite_cls in suites:
        suite = suite_cls()
        for name in [m for m in dir(suite_cls) if m.startswith("test_")]:
            try:
                getattr(suite, name)()
                print(f"  ✓ {suite_cls.__name__}.{name}")
                passed += 1
            except Exception as e:
                print(f"  ✗ {suite_cls.__name__}.{name}: {e}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
