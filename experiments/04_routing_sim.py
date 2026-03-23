"""
Experiment 04 — SSM Routing vs Static Strategies (PRIMARY GATE)

Question: Does a learned SSM router outperform round-robin and least-busy
          strategies on routing quality?

Dataset: simulated agent pool routing episodes (data/generators.py)
Models:  SSMRouter (trained) vs RoundRobin, LeastBusy, CostAware (baselines)
Metric:
  - avg_latency_ms: lower is better
  - error_rate: lower is better
  - total_cost_usd: lower is better
  - composite quality = (1 - error_rate) * (1 - latency_norm)

Gate (CLAUDE.md):
  - SSM beats round-robin by >10% on composite quality → integrate into agent-pool router
  - Within 5% of round-robin → static strategies are good enough

Output: results/04_routing_report.json
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generators import simulate_routing_episode, POOL_FEATURE_DIM
from core.ssm import SSMRouter

results_dir = Path(__file__).parent.parent / "results"
results_dir.mkdir(exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── config ────────────────────────────────────────────────────────────────────
CFG = {
    "n_workers": 5,              # matches generator default worker pool
    "event_dim": POOL_FEATURE_DIM,
    "seq_len": 16,               # routing history window
    "d_model": 32,
    "n_layers": 2,
    # training
    "n_train_episodes": 500,
    "batch_size": 64,
    "epochs": 20,
    "lr": 3e-4,
    # simulation
    "episode_length": 200,
    "n_sim_episodes": 200,
}

# Latency reference: max expected avg latency (ms) for normalization
# Based on generator worker configs (worst case degraded worker ~660ms avg)
MAX_LATENCY_MS = 700.0

# Cost reference: max expected avg cost per episode for normalization
# API worker costs $0.005/call × 200 requests = $1.00 worst case
MAX_COST_USD = 1.0

# Quality weights: must sum to 1.0
QUALITY_WEIGHTS = {"latency": 0.40, "reliability": 0.40, "cost": 0.20}


# ── composite quality ──────────────────────────────────────────────────────────

def composite_quality(avg_latency_ms: float, error_rate: float, avg_cost_usd: float = 0.0) -> float:
    """
    Single quality score in [0, 1] — higher is better.
    Weights: latency 40%, reliability 40%, cost 20%.
    Cost is included because cost-aware routing is a stated objective.
    """
    latency_score = 1.0 - min(avg_latency_ms / MAX_LATENCY_MS, 1.0)
    reliability_score = 1.0 - error_rate
    cost_score = 1.0 - min(avg_cost_usd / MAX_COST_USD, 1.0)
    return (
        QUALITY_WEIGHTS["latency"] * latency_score
        + QUALITY_WEIGHTS["reliability"] * reliability_score
        + QUALITY_WEIGHTS["cost"] * cost_score
    )


# ── training dataset ──────────────────────────────────────────────────────────

def build_routing_dataset(n_episodes: int, cfg: dict, seed: int = 42):
    """
    Imitation learning: train SSMRouter to mimic the cost-aware oracle.
    Each episode generates (event_history_window, chosen_worker) pairs.
    """
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 100000))
        m = simulate_routing_episode(
            n_requests=cfg["episode_length"],
            strategy="cost-aware",
            seed=ep_seed,
        )
        history = m["event_history"]
        decisions = m["decisions"]

        # Slice history into overlapping windows
        for t in range(cfg["seq_len"], len(history)):
            window = np.array(history[t - cfg["seq_len"]:t], dtype=np.float32)
            label = decisions[t] % cfg["n_workers"]
            X_list.append(window)
            y_list.append(label)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
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

def run_simulation(strategy: str, n_episodes: int, episode_length: int,
                   seed: int = 0, ssm_model=None) -> dict:
    """Run n_episodes routing simulations and aggregate metrics."""
    rng = np.random.default_rng(seed)
    all_latencies, all_errors, all_costs = [], [], []

    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 100000))
        m = simulate_routing_episode(
            n_requests=episode_length,
            strategy=strategy,
            ssm_model=ssm_model,
            seed=ep_seed,
        )
        all_latencies.append(m["avg_latency_ms"])
        all_errors.append(m["error_rate"])
        all_costs.append(m["total_cost_usd"])

    avg_lat = float(np.mean(all_latencies))
    avg_err = float(np.mean(all_errors))
    avg_cost = float(np.mean(all_costs))

    return {
        "strategy": strategy,
        "n_episodes": n_episodes,
        "avg_latency_ms": round(avg_lat, 2),
        "p95_latency_ms": round(float(np.percentile(all_latencies, 95)), 2),
        "error_rate": round(avg_err, 4),
        "avg_cost_usd": round(avg_cost, 4),
        "composite_quality": round(composite_quality(avg_lat, avg_err, avg_cost), 4),
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
            "condition": "SSM composite_quality > round-robin by > 10%",
            "integrate_threshold": 0.10,
            "comparable_threshold": 0.05,
        },
        "strategies": {},
    }

    # ── Static baselines ──
    print("\n[1/3] Running static strategy simulations...")
    for strategy in ["round-robin", "least-busy", "cost-aware"]:
        print(f"  {strategy} ({CFG['n_sim_episodes']} episodes)...", end=" ", flush=True)
        t0 = time.time()
        result = run_simulation(
            strategy, CFG["n_sim_episodes"], CFG["episode_length"], seed=SEED
        )
        elapsed = time.time() - t0
        result["sim_time_s"] = round(elapsed, 2)
        report["strategies"][strategy] = result
        print(f"quality={result['composite_quality']:.3f}  "
              f"latency={result['avg_latency_ms']:.0f}ms  "
              f"err={result['error_rate']:.3f}  ({elapsed:.1f}s)")

    # ── Train SSMRouter ──
    print(f"\n[2/3] Training SSMRouter (imitation of cost-aware oracle)...")
    print(f"  Building dataset from {CFG['n_train_episodes']} episodes...", end=" ", flush=True)
    t0 = time.time()
    X_all, y_all = build_routing_dataset(CFG["n_train_episodes"], CFG, seed=SEED)
    build_time = time.time() - t0
    print(f"{len(X_all):,} windows  ({build_time:.1f}s)")

    n_val = int(len(X_all) * 0.15)
    X_val, y_val = X_all[-n_val:], y_all[-n_val:]
    X_tr, y_tr = X_all[:-n_val], y_all[:-n_val]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        CFG["batch_size"], shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        CFG["batch_size"],
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
    print(f"  Training done ({train_time:.1f}s)  best_val_acc={train_result['best_val_acc']:.3f}")

    # ── Evaluate SSM in simulation ──
    print(f"\n[3/3] Evaluating SSMRouter in simulation ({CFG['n_sim_episodes']} episodes)...")
    ssm_router.eval()
    t0 = time.time()
    ssm_result = run_simulation(
        "ssm", CFG["n_sim_episodes"], CFG["episode_length"],
        seed=SEED, ssm_model=ssm_router,
    )
    sim_time = time.time() - t0
    ssm_result["train_result"] = train_result
    ssm_result["n_params"] = n_params
    ssm_result["train_time_s"] = round(train_time, 1)
    ssm_result["sim_time_s"] = round(sim_time, 2)
    report["strategies"]["ssm_router"] = ssm_result
    print(f"  quality={ssm_result['composite_quality']:.3f}  "
          f"latency={ssm_result['avg_latency_ms']:.0f}ms  "
          f"err={ssm_result['error_rate']:.3f}  ({sim_time:.1f}s)")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    rr_q = report["strategies"]["round-robin"]["composite_quality"]
    lb_q = report["strategies"]["least-busy"]["composite_quality"]
    ca_q = report["strategies"]["cost-aware"]["composite_quality"]
    ssm_q = ssm_result["composite_quality"]

    print(f"\n  {'Strategy':<20} {'Quality':>8}  {'Latency(ms)':>12}  {'Err':>6}  {'Cost($)':>8}")
    print(f"  {'-' * 60}")
    for s_name, label in [
        ("round-robin", ""), ("least-busy", ""), ("cost-aware", ""),
        ("ssm_router", " ← SSM"),
    ]:
        s = report["strategies"][s_name]
        print(f"  {s_name:<20} {s['composite_quality']:>8.3f}  "
              f"{s['avg_latency_ms']:>12.1f}  {s['error_rate']:>6.3f}  "
              f"{s.get('avg_cost_usd', 0):>8.4f}{label}")

    delta_vs_rr = ssm_q - rr_q
    delta_vs_lb = ssm_q - lb_q
    delta_vs_ca = ssm_q - ca_q

    print(f"\n  Delta SSM vs round-robin : {delta_vs_rr:+.3f}")
    print(f"  Delta SSM vs least-busy  : {delta_vs_lb:+.3f}")
    print(f"  Delta SSM vs cost-aware  : {delta_vs_ca:+.3f}")

    report["deltas"] = {
        "ssm_vs_round_robin": round(delta_vs_rr, 4),
        "ssm_vs_least_busy": round(delta_vs_lb, 4),
        "ssm_vs_cost_aware": round(delta_vs_ca, 4),
    }

    integrate_t = report["gate"]["integrate_threshold"]
    comparable_t = report["gate"]["comparable_threshold"]

    if delta_vs_rr > integrate_t:
        verdict = (
            f"SSM WINS by {delta_vs_rr:.1%} over round-robin — "
            f"exceeds {integrate_t:.0%} gate → INTEGRATE into agent-pool router"
        )
    elif delta_vs_rr > -comparable_t:
        verdict = (
            f"SSM COMPARABLE to round-robin (delta={delta_vs_rr:+.1%}) — "
            f"within {comparable_t:.0%} → static strategies sufficient; "
            f"SSM preferred for streaming/stateful use cases"
        )
    else:
        verdict = (
            f"STATIC STRATEGIES WIN (delta={delta_vs_rr:+.1%}) — "
            f"SSM routing adds overhead without quality gain → keep round-robin/least-busy"
        )

    report["verdict"] = verdict
    print(f"\n{'=' * 60}")
    print(f"GATE VERDICT: {verdict}")
    print(f"{'=' * 60}")

    out = results_dir / "04_routing_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out}")
    print("\nAll experiments complete. See docs/integration_plan.md for next steps.")


if __name__ == "__main__":
    main()
