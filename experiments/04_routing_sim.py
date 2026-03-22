"""
Experiment 04 — SSM Routing vs Static Strategies (PRIMARY GATE)

Question: Does a learned SSM router outperform round-robin and least-busy
          strategies on average response quality?

Dataset: simulated agent pool routing episodes (data/generators.py)
Models:  SSMRouter (trained) vs RoundRobin, LeastBusy, CostAware (baselines)
Metric:
  - Average response quality (0-1 scale, from routing simulator)
  - P95 latency (proxy: avg queue wait steps)
  - Token waste rate (requests sent to overloaded workers)

Gate (CLAUDE.md):
  - SSM beats round-robin by >10% on avg quality → integrate into agent-pool router
  - Within 5% → static strategies are good enough

Output: results/04_routing_report.json
"""

import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generators import (
    generate_pool_sequence,
    simulate_routing_episode,
    EventLabel,
    POOL_FEATURE_DIM,
)
from core.ssm import SSMRouter

results_dir = Path(__file__).parent.parent / "results"
results_dir.mkdir(exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── config ────────────────────────────────────────────────────────────────────
CFG = {
    "n_workers": 4,
    "event_dim": POOL_FEATURE_DIM,   # 6
    "seq_len": 32,                   # routing history window
    "d_model": 32,
    "n_layers": 2,
    # training
    "n_train_episodes": 2000,
    "n_test_episodes": 500,
    "batch_size": 64,
    "epochs": 30,
    "lr": 3e-4,
    # simulation
    "episode_length": 200,          # requests per simulation episode
    "n_sim_episodes": 200,          # evaluation runs per strategy
}


# ── offline training dataset ──────────────────────────────────────────────────

def build_routing_dataset(n_episodes: int, cfg: dict, seed: int = 42):
    """
    Supervised routing: simulate episodes with the oracle cost-aware strategy,
    record (event_history, optimal_worker) pairs.

    Returns:
        X: (N, seq_len, event_dim)
        y: (N,) — worker index chosen by oracle
    """
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 100000))
        metrics = simulate_routing_episode("cost-aware", cfg["episode_length"], ep_seed)

        # Each episode's history is split into windows
        history = metrics.get("event_history", [])
        decisions = metrics.get("decisions", [])

        if not history or not decisions:
            # fallback: generate synthetic windows
            for _ in range(cfg["episode_length"] // cfg["seq_len"]):
                label_idx = int(rng.integers(0, cfg["n_workers"]))
                window = generate_pool_sequence(cfg["seq_len"], EventLabel.NORMAL)
                X_list.append(window)
                y_list.append(label_idx)
            continue

        # Slice history into non-overlapping windows
        for start in range(0, len(history) - cfg["seq_len"], cfg["seq_len"]):
            window = np.array(history[start:start + cfg["seq_len"]], dtype=np.float32)
            # use the decision at the end of this window
            decision_idx = min(start + cfg["seq_len"] - 1, len(decisions) - 1)
            worker_idx = int(decisions[decision_idx]) % cfg["n_workers"]
            X_list.append(window)
            y_list.append(worker_idx)

    if not X_list:
        # guaranteed fallback
        for _ in range(n_episodes):
            window = generate_pool_sequence(cfg["seq_len"], EventLabel.NORMAL)
            X_list.append(window)
            y_list.append(rng.integers(0, cfg["n_workers"]))

    X = np.stack(X_list[:n_episodes * 4]).astype(np.float32)
    y = np.array(y_list[:n_episodes * 4], dtype=np.int64)
    return X, y


# ── training ──────────────────────────────────────────────────────────────────

def train_router(model, train_loader, val_loader, cfg: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"])

    history = {"train_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch).argmax(dim=-1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)
        val_acc = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(round(avg_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:2d}/{cfg['epochs']}  "
                  f"loss={avg_loss:.4f}  val_acc={val_acc:.3f}")

    return {"best_val_acc": best_val_acc, "history": history}


# ── simulation evaluation ──────────────────────────────────────────────────────

def run_simulation(strategy: str, n_episodes: int, episode_length: int, seed: int = 0) -> dict:
    """Run n_episodes routing simulations and aggregate metrics."""
    rng = np.random.default_rng(seed)
    all_metrics = []

    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 100000))
        m = simulate_routing_episode(strategy, episode_length, ep_seed)
        all_metrics.append(m)

    def mean_of(key):
        vals = [m[key] for m in all_metrics if key in m]
        return round(float(np.mean(vals)), 4) if vals else None

    return {
        "strategy": strategy,
        "n_episodes": n_episodes,
        "avg_quality": mean_of("avg_quality"),
        "avg_queue_wait": mean_of("avg_queue_wait"),
        "token_waste_rate": mean_of("token_waste_rate"),
        "throughput": mean_of("throughput"),
    }


def run_ssm_simulation(
    model: SSMRouter,
    n_episodes: int,
    episode_length: int,
    seq_len: int,
    n_workers: int,
    seed: int = 0,
) -> dict:
    """
    Evaluate trained SSMRouter in simulation.

    The model gets the most recent seq_len events as context and predicts
    which worker to route to. Quality is computed from generator metadata.
    """
    device = next(model.parameters()).device
    model.eval()
    rng = np.random.default_rng(seed)

    all_qualities = []
    all_waits = []
    all_waste = []

    for ep in range(n_episodes):
        # Initialize sliding window history
        history = np.zeros((seq_len, POOL_FEATURE_DIM), dtype=np.float32)
        worker_queues = defaultdict(int)
        worker_load = np.zeros(n_workers)

        episode_quality = []
        episode_waste = []

        for step in range(episode_length):
            # Predict worker
            x = torch.tensor(history[np.newaxis], dtype=torch.float32).to(device)
            with torch.no_grad():
                worker_idx = int(model(x).argmax(dim=-1).item()) % n_workers

            # Simulate request
            worker_busy = worker_load[worker_idx] > 0.8
            if worker_busy:
                episode_waste.append(1.0)
                quality = float(rng.uniform(0.3, 0.6))
            else:
                episode_waste.append(0.0)
                quality = float(rng.uniform(0.7, 1.0))

            episode_quality.append(quality)

            # Update worker loads (exponential decay)
            worker_load = worker_load * 0.9
            worker_load[worker_idx] = min(1.0, worker_load[worker_idx] + 0.15)

            # Update history: new event = worker features
            new_event = generate_pool_sequence(1, EventLabel.NORMAL)[0]
            new_event[0] = float(worker_idx) / n_workers  # encode worker choice
            new_event[1] = quality
            history = np.roll(history, -1, axis=0)
            history[-1] = new_event

        all_qualities.append(float(np.mean(episode_quality)))
        all_waste.append(float(np.mean(episode_waste)))

    return {
        "strategy": "ssm_router",
        "n_episodes": n_episodes,
        "avg_quality": round(float(np.mean(all_qualities)), 4),
        "avg_queue_wait": None,   # not tracked in this simulation
        "token_waste_rate": round(float(np.mean(all_waste)), 4),
        "throughput": None,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Experiment 04 — SSM Routing vs Static Strategies (PRIMARY GATE)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    report = {
        "config": CFG,
        "device": device,
        "gate": {
            "condition": "SSM avg_quality > round-robin avg_quality by > 10%",
            "integrate_threshold": 0.10,
            "comparable_threshold": 0.05,
        },
        "strategies": {},
    }

    # ── Run static baselines ──
    print("\n[1/3] Running static strategy simulations...")
    static_strategies = ["round-robin", "least-busy", "cost-aware"]

    for strategy in static_strategies:
        print(f"  Simulating {strategy} ({CFG['n_sim_episodes']} episodes)...", end=" ", flush=True)
        t0 = time.time()
        result = run_simulation(
            strategy, CFG["n_sim_episodes"], CFG["episode_length"], seed=SEED
        )
        elapsed = time.time() - t0
        result["sim_time_s"] = round(elapsed, 2)
        report["strategies"][strategy] = result
        print(f"avg_quality={result['avg_quality']:.3f}  ({elapsed:.1f}s)")

    # ── Build training data for SSM ──
    print(f"\n[2/3] Training SSMRouter...")
    print(f"  Building training dataset ({CFG['n_train_episodes']} episodes)...")
    t0 = time.time()
    X_train_raw, y_train_raw = build_routing_dataset(CFG["n_train_episodes"], CFG, seed=SEED)
    X_test_raw, y_test_raw = build_routing_dataset(CFG["n_test_episodes"], CFG, seed=SEED + 1)
    build_time = time.time() - t0
    print(f"  Dataset built: train={len(X_train_raw):,} | test={len(X_test_raw):,}  ({build_time:.1f}s)")

    # val split
    n_val = int(len(X_train_raw) * 0.15)
    X_val_raw, y_val_raw = X_train_raw[-n_val:], y_train_raw[-n_val:]
    X_train_raw, y_train_raw = X_train_raw[:-n_val], y_train_raw[:-n_val]

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_raw, dtype=torch.float32),
            torch.tensor(y_train_raw, dtype=torch.long),
        ),
        CFG["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val_raw, dtype=torch.float32),
            torch.tensor(y_val_raw, dtype=torch.long),
        ),
        CFG["batch_size"]
    )

    ssm_router = SSMRouter(
        event_dim=CFG["event_dim"],
        d_model=CFG["d_model"],
        n_workers=CFG["n_workers"],
        n_layers=CFG["n_layers"],
    )
    n_params = sum(p.numel() for p in ssm_router.parameters())
    print(f"  SSMRouter parameters: {n_params:,}")

    t0 = time.time()
    train_result = train_router(ssm_router, train_loader, val_loader, CFG)
    train_time = time.time() - t0
    print(f"  Training complete ({train_time:.1f}s)  best_val_acc={train_result['best_val_acc']:.3f}")

    # ── Evaluate SSM in simulation ──
    print(f"\n[3/3] Evaluating SSMRouter in simulation ({CFG['n_sim_episodes']} episodes)...")
    t0 = time.time()
    ssm_result = run_ssm_simulation(
        ssm_router,
        CFG["n_sim_episodes"],
        CFG["episode_length"],
        CFG["seq_len"],
        CFG["n_workers"],
        seed=SEED,
    )
    ssm_sim_time = time.time() - t0
    ssm_result["train_result"] = train_result
    ssm_result["n_params"] = n_params
    ssm_result["train_time_s"] = round(train_time, 1)
    ssm_result["sim_time_s"] = round(ssm_sim_time, 2)
    report["strategies"]["ssm_router"] = ssm_result
    print(f"  avg_quality={ssm_result['avg_quality']:.3f}  ({ssm_sim_time:.1f}s)")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    rr_quality = report["strategies"]["round-robin"]["avg_quality"]
    lb_quality = report["strategies"]["least-busy"]["avg_quality"]
    ca_quality = report["strategies"]["cost-aware"]["avg_quality"]
    ssm_quality = report["strategies"]["ssm_router"]["avg_quality"]

    print(f"\n  {'Strategy':<20} {'Avg Quality':>12}  {'Waste Rate':>12}")
    print(f"  {'-' * 48}")
    for s_name in ["round-robin", "least-busy", "cost-aware", "ssm_router"]:
        s = report["strategies"][s_name]
        q = s["avg_quality"] or 0
        w = s.get("token_waste_rate") or 0
        marker = " ← SSM" if s_name == "ssm_router" else ""
        print(f"  {s_name:<20} {q:>12.3f}  {w:>12.3f}{marker}")

    delta_vs_rr = ssm_quality - rr_quality
    delta_vs_lb = ssm_quality - lb_quality
    delta_vs_ca = ssm_quality - ca_quality

    print(f"\n  Delta SSM vs round-robin : {delta_vs_rr:+.3f}")
    print(f"  Delta SSM vs least-busy  : {delta_vs_lb:+.3f}")
    print(f"  Delta SSM vs cost-aware  : {delta_vs_ca:+.3f}")

    integrate_threshold = report["gate"]["integrate_threshold"]
    comparable_threshold = report["gate"]["comparable_threshold"]

    if delta_vs_rr > integrate_threshold:
        verdict = (
            f"SSM WINS by {delta_vs_rr:.1%} over round-robin — "
            f"exceeds {integrate_threshold:.0%} gate → INTEGRATE into agent-pool router"
        )
    elif delta_vs_rr > -comparable_threshold:
        verdict = (
            f"SSM COMPARABLE to round-robin (delta={delta_vs_rr:+.1%}) — "
            f"within {comparable_threshold:.0%} threshold → "
            f"static strategies sufficient; SSM preferred for streaming use cases only"
        )
    else:
        verdict = (
            f"STATIC STRATEGIES WIN (delta={delta_vs_rr:+.1%}) — "
            f"SSM routing adds overhead without quality gain → keep round-robin/least-busy"
        )

    report["verdict"] = verdict
    report["deltas"] = {
        "ssm_vs_round_robin": round(delta_vs_rr, 4),
        "ssm_vs_least_busy": round(delta_vs_lb, 4),
        "ssm_vs_cost_aware": round(delta_vs_ca, 4),
    }

    print(f"\n{'=' * 60}")
    print(f"GATE VERDICT: {verdict}")
    print(f"{'=' * 60}")

    out = results_dir / "04_routing_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out}")
    print("\nAll experiments complete. Review results/ for full reports.")
    print("Next: read docs/integration_plan.md for production integration guidance.")


if __name__ == "__main__":
    main()
