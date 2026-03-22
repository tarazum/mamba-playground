"""
Synthetic event stream generators for Mamba playground experiments.

No external data required — everything is generated procedurally.
All generators produce numpy arrays that can be converted to torch tensors.

Three event types:
    AgentPoolEvent  — worker request/response cycles (for agent-pool routing experiments)
    PipelineEvent   — stage execution events (for SimpleAO monitoring experiments)
    TrendSignal     — signal cluster measurements across runs (for TSE velocity experiments)
"""

import numpy as np
from dataclasses import dataclass
from enum import IntEnum


# ── event schemas ──────────────────────────────────────────────────────────────

class WorkerType(IntEnum):
    CLI = 0
    OLLAMA = 1
    API = 2


class RequestType(IntEnum):
    CODE = 0
    ANALYSIS = 1
    CREATIVE = 2
    FAST = 3


class EventLabel(IntEnum):
    NORMAL = 0
    DEGRADING = 1
    STUCK = 2


class PipelineLabel(IntEnum):
    NORMAL = 0
    LOOPING = 1
    STALLING = 2


# Feature indices for AgentPoolEvent vectors
# [worker_type, request_type, latency_norm, success, cost_norm, retry_count_norm]
POOL_FEATURE_DIM = 6

# Feature indices for PipelineEvent vectors
# [stage_id, status, duration_norm, retry_count_norm, artifact_size_norm]
PIPELINE_FEATURE_DIM = 5


# ── agent pool event generator ─────────────────────────────────────────────────

def generate_pool_sequence(
    n_events: int = 64,
    pattern: EventLabel = EventLabel.NORMAL,
    n_workers: int = 5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a sequence of agent pool events as a float32 array of shape (n_events, POOL_FEATURE_DIM).

    Normal:    stable latency, high success rate
    Degrading: latency increases linearly, error rate rises gradually
    Stuck:     one worker stops responding (latency spikes to max, then timeout)
    """
    if rng is None:
        rng = np.random.default_rng()

    events = np.zeros((n_events, POOL_FEATURE_DIM), dtype=np.float32)

    for i in range(n_events):
        t = i / n_events  # normalized time [0, 1]

        worker_type = rng.integers(0, 3)
        request_type = rng.integers(0, 4)

        if pattern == EventLabel.NORMAL:
            base_latency = 200 + (worker_type * 50)
            latency = rng.normal(base_latency, 30)
            success = 1.0 if rng.random() > 0.05 else 0.0
            cost = 0.0 if worker_type < 2 else rng.uniform(0.001, 0.01)
            retries = 0.0

        elif pattern == EventLabel.DEGRADING:
            # latency grows linearly; error rate rises from 5% to 40%
            base_latency = 200 + (worker_type * 50)
            degradation = t * 3.0
            latency = rng.normal(base_latency * (1 + degradation), 50 + t * 100)
            error_prob = 0.05 + t * 0.35
            success = 1.0 if rng.random() > error_prob else 0.0
            cost = 0.0 if worker_type < 2 else rng.uniform(0.001, 0.02)
            retries = min(t * 2.0, 1.0)

        elif pattern == EventLabel.STUCK:
            # first 40% normal, then one worker starts hanging
            if t < 0.4:
                latency = rng.normal(200, 30)
                success = 1.0 if rng.random() > 0.05 else 0.0
                retries = 0.0
            elif t < 0.6:
                # worker starting to slow down
                latency = rng.normal(800, 200)
                success = 1.0 if rng.random() > 0.3 else 0.0
                retries = rng.uniform(0.3, 0.7)
            else:
                # stuck: timeout responses
                latency = 30000.0 + rng.normal(0, 100)
                success = 0.0
                retries = 1.0
            cost = 0.0 if worker_type < 2 else rng.uniform(0.001, 0.01)

        # normalize features to [0, 1]
        events[i, 0] = worker_type / 2.0
        events[i, 1] = request_type / 3.0
        events[i, 2] = min(latency / 30000.0, 1.0)
        events[i, 3] = success
        events[i, 4] = min(cost / 0.02, 1.0)
        events[i, 5] = retries

    return events


def generate_pool_dataset(
    n_samples: int = 1000,
    seq_len: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (X, y) where:
        X: float32 (n_samples, seq_len, POOL_FEATURE_DIM)
        y: int64   (n_samples,)  — 0=normal, 1=degrading, 2=stuck
    """
    rng = np.random.default_rng(seed)
    labels = [EventLabel.NORMAL, EventLabel.DEGRADING, EventLabel.STUCK]

    X_list, y_list = [], []
    per_class = n_samples // len(labels)

    for label in labels:
        for _ in range(per_class):
            seq = generate_pool_sequence(seq_len, pattern=label, rng=rng)
            X_list.append(seq)
            y_list.append(int(label))

    # shuffle
    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ── pipeline event generator ───────────────────────────────────────────────────

PIPELINE_STAGES = ["brief", "draft", "review", "integrate", "check", "accept"]


def generate_pipeline_sequence(
    n_events: int = 48,
    pattern: PipelineLabel = PipelineLabel.NORMAL,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a sequence of pipeline stage events for SimpleAO monitoring.

    Normal:   stages progress in order, short durations, rare retries
    Looping:  one stage repeats 3-8 times before advancing
    Stalling: stage duration grows each attempt, never completes
    """
    if rng is None:
        rng = np.random.default_rng()

    events = np.zeros((n_events, PIPELINE_FEATURE_DIM), dtype=np.float32)
    n_stages = len(PIPELINE_STAGES)

    if pattern == PipelineLabel.NORMAL:
        stage_idx = 0
        for i in range(n_events):
            stage_idx = min(stage_idx + (1 if rng.random() > 0.2 else 0), n_stages - 1)
            events[i, 0] = stage_idx / (n_stages - 1)
            events[i, 1] = 1.0  # success
            events[i, 2] = rng.uniform(0.1, 0.4)  # duration (normalized)
            events[i, 3] = 0.0  # retry count
            events[i, 4] = rng.uniform(0.1, 0.6)  # artifact size

    elif pattern == PipelineLabel.LOOPING:
        loop_stage = rng.integers(1, n_stages - 1)
        loop_count = rng.integers(5, 10)
        stage_idx = 0
        loop_iter = 0
        for i in range(n_events):
            if stage_idx == loop_stage and loop_iter < loop_count:
                loop_iter += 1
                retry_norm = min(loop_iter / 10.0, 1.0)
            else:
                if loop_iter >= loop_count:
                    stage_idx = min(stage_idx + 1, n_stages - 1)
                    loop_iter = 0
                else:
                    stage_idx = min(stage_idx + (1 if rng.random() > 0.3 else 0), loop_stage)
                retry_norm = 0.0
            events[i, 0] = stage_idx / (n_stages - 1)
            events[i, 1] = 0.5 if loop_iter > 0 else 1.0
            events[i, 2] = rng.uniform(0.1, 0.3)
            events[i, 3] = retry_norm
            events[i, 4] = rng.uniform(0.0, 0.3) if loop_iter > 0 else rng.uniform(0.1, 0.6)

    elif pattern == PipelineLabel.STALLING:
        stall_stage = rng.integers(1, n_stages - 1)
        stage_idx = 0
        stall_depth = 0
        for i in range(n_events):
            t = i / n_events
            if stage_idx < stall_stage:
                stage_idx = min(stage_idx + (1 if rng.random() > 0.3 else 0), stall_stage)
                duration = rng.uniform(0.1, 0.3)
                status = 1.0
                retry_norm = 0.0
            else:
                stall_depth += 1
                duration = min(0.1 + stall_depth * 0.08, 1.0)  # growing
                status = 0.2  # failing
                retry_norm = min(stall_depth / 10.0, 1.0)
            events[i, 0] = stage_idx / (n_stages - 1)
            events[i, 1] = status
            events[i, 2] = duration
            events[i, 3] = retry_norm
            events[i, 4] = rng.uniform(0.0, 0.2)

    return events


def generate_pipeline_dataset(
    n_samples: int = 900,
    seq_len: int = 48,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = [PipelineLabel.NORMAL, PipelineLabel.LOOPING, PipelineLabel.STALLING]

    X_list, y_list = [], []
    per_class = n_samples // len(labels)

    for label in labels:
        for _ in range(per_class):
            seq = generate_pipeline_sequence(seq_len, pattern=label, rng=rng)
            X_list.append(seq)
            y_list.append(int(label))

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ── trend signal generator (TSE velocity tracking) ────────────────────────────

def generate_trend_velocity_dataset(
    n_trends: int = 200,
    n_weeks: int = 12,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate TSE running weekly. Each trend has a cluster_size over 12 weeks.

    Labels:
        0 = rising    (cluster size increasing)
        1 = peaking   (cluster size cresting then declining)
        2 = declining (cluster size decreasing)
        3 = noise     (random, no clear trend)

    Returns:
        X: float32 (n_trends, n_weeks, 3)  — [cluster_size_norm, diversity_norm, score_norm]
        y: int64   (n_trends,)
    """
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    patterns = {
        0: "rising",
        1: "peaking",
        2: "declining",
        3: "noise",
    }

    per_class = n_trends // len(patterns)

    for label in range(len(patterns)):
        for _ in range(per_class):
            weeks = np.zeros((n_weeks, 3), dtype=np.float32)
            t = np.linspace(0, 1, n_weeks)

            if label == 0:  # rising
                base = 0.1 + t * 0.8
                noise = rng.normal(0, 0.05, n_weeks)
            elif label == 1:  # peaking
                peak = rng.uniform(0.4, 0.7)
                base = np.where(t < peak, t / peak, 1.0 - (t - peak) / (1.0 - peak))
                noise = rng.normal(0, 0.05, n_weeks)
            elif label == 2:  # declining
                base = 0.9 - t * 0.8
                noise = rng.normal(0, 0.05, n_weeks)
            else:  # noise
                base = rng.uniform(0.1, 0.9, n_weeks)
                noise = rng.normal(0, 0.1, n_weeks)

            cluster_size = np.clip(base + noise, 0.0, 1.0)
            diversity = np.clip(cluster_size * rng.uniform(0.7, 1.0) + rng.normal(0, 0.05, n_weeks), 0, 1)
            score = np.clip((cluster_size * 0.6 + diversity * 0.4) + rng.normal(0, 0.03, n_weeks), 0, 1)

            weeks[:, 0] = cluster_size
            weeks[:, 1] = diversity
            weeks[:, 2] = score

            X_list.append(weeks)
            y_list.append(label)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ── routing simulation ─────────────────────────────────────────────────────────

@dataclass
class WorkerState:
    worker_id: str
    worker_type: WorkerType
    base_latency: float       # ms
    error_rate: float         # 0.0 - 1.0
    cost_per_call: float      # USD
    is_degrading: bool = False
    degradation_start: int = 0


def simulate_routing_episode(
    n_requests: int = 200,
    workers: list[WorkerState] | None = None,
    strategy: str = "round-robin",
    ssm_model=None,
    seed: int = 42,
) -> dict:
    """
    Simulate routing N requests through a pool of workers using the given strategy.

    Strategies: "round-robin", "least-busy", "cost-aware", "ssm" (requires ssm_model)

    Returns a dict with metrics: avg_latency, total_cost, error_rate, timeout_rate
    """
    rng = np.random.default_rng(seed)

    if workers is None:
        workers = [
            WorkerState("claude-0", WorkerType.CLI, 220, 0.03, 0.0),
            WorkerState("claude-1", WorkerType.CLI, 240, 0.03, 0.0),
            WorkerState("ollama-0", WorkerType.OLLAMA, 180, 0.05, 0.0),
            WorkerState("openai-0", WorkerType.API, 300, 0.02, 0.005),
            # one worker will degrade at request 100
            WorkerState("claude-2", WorkerType.CLI, 220, 0.03, 0.0, is_degrading=True, degradation_start=100),
        ]

    rr_index = 0
    call_counts = [0] * len(workers)
    history: list[np.ndarray] = []

    latencies, costs, errors, timeouts = [], [], [], []

    for req_i in range(n_requests):
        # compute current worker states
        worker_latencies = []
        worker_errors = []
        for wi, w in enumerate(workers):
            t = max(0, req_i - w.degradation_start) if w.is_degrading and req_i >= w.degradation_start else 0
            deg = t / 100.0 if w.is_degrading else 0.0
            effective_latency = w.base_latency * (1 + deg * 2)
            effective_error = min(w.error_rate + deg * 0.4, 0.9)
            worker_latencies.append(effective_latency)
            worker_errors.append(effective_error)

        # select worker
        if strategy == "round-robin":
            wi = rr_index % len(workers)
            rr_index += 1

        elif strategy == "least-busy":
            wi = int(np.argmin(call_counts))

        elif strategy == "cost-aware":
            # prefer zero-cost workers
            zero_cost = [i for i, w in enumerate(workers) if w.cost_per_call == 0.0]
            if zero_cost:
                wi = min(zero_cost, key=lambda i: call_counts[i])
            else:
                wi = int(np.argmin(call_counts))

        elif strategy == "ssm" and ssm_model is not None:
            # use SSM model to score workers (returns index of best worker)
            if len(history) >= 4:
                import torch
                seq = np.stack(history[-16:])  # last 16 events
                seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    wi = ssm_model.route(seq_t, len(workers))
            else:
                wi = rr_index % len(workers)
                rr_index += 1
        else:
            wi = rr_index % len(workers)
            rr_index += 1

        call_counts[wi] += 1
        w = workers[wi]
        latency = worker_latencies[wi]
        error_prob = worker_errors[wi]

        # simulate call outcome
        is_error = rng.random() < error_prob
        is_timeout = latency > 25000
        actual_latency = rng.normal(latency, latency * 0.1)
        if is_timeout:
            actual_latency = 30000.0

        latencies.append(actual_latency)
        costs.append(w.cost_per_call)
        errors.append(float(is_error))
        timeouts.append(float(is_timeout))

        # record event for SSM history
        event = np.array([
            wi / len(workers),
            actual_latency / 30000.0,
            float(is_error),
            w.cost_per_call / 0.01,
            call_counts[wi] / n_requests,
            0.0,
        ], dtype=np.float32)
        history.append(event)

    return {
        "strategy": strategy,
        "avg_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "total_cost_usd": float(np.sum(costs)),
        "error_rate": float(np.mean(errors)),
        "timeout_rate": float(np.mean(timeouts)),
        "call_distribution": call_counts,
    }
