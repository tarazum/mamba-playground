"""
Experiment 04 — SSM Routing vs Static Strategies (PRIMARY GATE)

Improvements over v1:
  - Prediction-based routing: SSMQualityPredictor trains to predict worker quality
    from observed outcomes (random routing data), not to copy an oracle's labels.
  - Queue-depth least-busy: tracks in-flight requests, not cumulative counts.
  - Sticky routing baseline: added as a real competitor.
  - Distribution shift: evaluate all strategies on a heavy-degradation pool not
    seen during training.

Gate (CLAUDE.md):
  - SSM composite quality beats round-robin by >10% → candidate for agent-pool router
  - Within 5% → static strategies are good enough

Output: results/04_routing_report.json
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generators import (
    simulate_routing_episode,
    default_worker_pool,
    heavy_degradation_pool,
    POOL_FEATURE_DIM,
)
from core.ssm import SSMQualityPredictor

results_dir = Path(__file__).parent.parent / "results"
results_dir.mkdir(exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

CFG = {
    "n_workers": 5,
    "event_dim": POOL_FEATURE_DIM,
    "seq_len": 16,
    "d_model": 32,
    "n_layers": 2,
    "n_train_episodes": 500,
    "batch_size": 64,
    "epochs": 20,
    "lr": 3e-4,
    "episode_length": 200,
    "n_sim_episodes": 200,
}

MAX_LATENCY_MS = 700.0
MAX_COST_USD   = 1.0

QUALITY_WEIGHTS = {"latency": 0.40, "reliability": 0.40, "cost": 0.20}


def composite_quality(avg_latency_ms: float, error_rate: float, avg_cost_usd: float = 0.0) -> float:
    lat_score  = 1.0 - min(avg_latency_ms / MAX_LATENCY_MS, 1.0)
    rel_score  = 1.0 - error_rate
    cost_score = 1.0 - min(avg_cost_usd / MAX_COST_USD, 1.0)
    return (
        QUALITY_WEIGHTS["latency"]      * lat_score
        + QUALITY_WEIGHTS["reliability"] * rel_score
        + QUALITY_WEIGHTS["cost"]        * cost_score
    )


# ── training data (quality prediction, not imitation) ─────────────────────────

def build_quality_prediction_dataset(n_episodes: int, cfg: dict, seed: int = 42):
    """
    Collect (history_window, worker_idx, observed_quality) tuples from
    round-robin episodes.  Round-robin provides roughly balanced coverage of
    all workers without requiring counterfactual rollouts.

    Returns:
        X:        (N, seq_len, event_dim)
        targets:  (N, n_workers)  — observed quality at chosen worker;
                                    -1.0 everywhere else (masked in loss)
    """
    rng = np.random.default_rng(seed)
    X_list, target_list = [], []

    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 100_000))
        m = simulate_routing_episode(
            n_requests=cfg["episode_length"],
            strategy="round-robin",   # unbiased observation across workers
            seed=ep_seed,
        )
        history   = m["event_history"]
        decisions = m["decisions"]
        step_qual = m["step_quality"]

        for t in range(cfg["seq_len"], len(history)):
            window = np.array(history[t - cfg["seq_len"]:t], dtype=np.float32)
            wi     = decisions[t] % cfg["n_workers"]
            q      = float(step_qual[t])

            target = np.full(cfg["n_workers"], -1.0, dtype=np.float32)
            target[wi] = q

            X_list.append(window)
            target_list.append(target)

    return np.stack(X_list), np.stack(target_list)


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE only on observed entries (target >= 0)."""
    mask = target >= 0
    if not mask.any():
        return pred.sum() * 0.0
    return F.mse_loss(pred[mask], target[mask])


# ── training ──────────────────────────────────────────────────────────────────

def train_quality_predictor(model, train_loader, val_loader, cfg: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])

    history   = {"train_loss": [], "val_loss": []}
    best_vloss = float("inf")

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        for X_b, t_b in train_loader:
            X_b, t_b = X_b.to(device), t_b.to(device)
            opt.zero_grad()
            loss = masked_mse(model(X_b), t_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, t_b in val_loader:
                X_b, t_b = X_b.to(device), t_b.to(device)
                val_loss += masked_mse(model(X_b), t_b).item()
        avg_train = total_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        history["train_loss"].append(round(avg_train, 5))
        history["val_loss"].append(round(avg_val, 5))
        if avg_val < best_vloss:
            best_vloss = avg_val
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:2d}/{cfg['epochs']}  "
                  f"train_mse={avg_train:.5f}  val_mse={avg_val:.5f}")

    return {"best_val_loss": round(best_vloss, 5), "history": history}


# ── simulation ─────────────────────────────────────────────────────────────────

def run_simulation(
    strategy: str,
    n_episodes: int,
    episode_length: int,
    seed: int = 0,
    ssm_model=None,
    workers_fn=None,
) -> dict:
    rng = np.random.default_rng(seed)
    all_lat, all_err, all_cost = [], [], []

    for ep in range(n_episodes):
        ep_seed = int(rng.integers(0, 100_000))
        workers = workers_fn() if workers_fn else None
        m = simulate_routing_episode(
            n_requests=episode_length,
            workers=workers,
            strategy=strategy,
            ssm_model=ssm_model,
            seed=ep_seed,
        )
        all_lat.append(m["avg_latency_ms"])
        all_err.append(m["error_rate"])
        all_cost.append(m["avg_cost_usd"])

    avg_lat  = float(np.mean(all_lat))
    avg_err  = float(np.mean(all_err))
    avg_cost = float(np.mean(all_cost))
    return {
        "strategy":          strategy,
        "n_episodes":        n_episodes,
        "avg_latency_ms":    round(avg_lat,  2),
        "p95_latency_ms":    round(float(np.percentile(all_lat, 95)), 2),
        "error_rate":        round(avg_err,  4),
        "avg_cost_usd":      round(avg_cost, 4),
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
        "in_distribution": {},
        "distribution_shift": {},
    }

    # ── Static baselines (in-distribution) ──
    print("\n[1/4] Static strategy simulations (default pool)...")
    strategies = ["round-robin", "least-busy", "cost-aware", "sticky"]
    for strategy in strategies:
        print(f"  {strategy} ({CFG['n_sim_episodes']} eps)...", end=" ", flush=True)
        t0 = time.time()
        result = run_simulation(
            strategy, CFG["n_sim_episodes"], CFG["episode_length"],
            seed=SEED, workers_fn=default_worker_pool,
        )
        result["sim_time_s"] = round(time.time() - t0, 2)
        report["in_distribution"][strategy] = result
        print(f"quality={result['composite_quality']:.3f}  "
              f"lat={result['avg_latency_ms']:.0f}ms  "
              f"err={result['error_rate']:.3f}  "
              f"cost=${result['avg_cost_usd']:.4f}")

    # ── Train SSMQualityPredictor ──
    print(f"\n[2/4] Training SSMQualityPredictor (quality prediction on random routing data)...")
    print(f"  Building dataset from {CFG['n_train_episodes']} round-robin episodes...",
          end=" ", flush=True)
    t0 = time.time()
    X_all, T_all = build_quality_prediction_dataset(CFG["n_train_episodes"], CFG, seed=SEED)
    build_time = time.time() - t0
    print(f"{len(X_all):,} windows  ({build_time:.1f}s)")

    n_val  = int(len(X_all) * 0.15)
    X_val, T_val  = X_all[-n_val:], T_all[-n_val:]
    X_tr,  T_tr   = X_all[:-n_val], T_all[:-n_val]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(T_tr)),
        CFG["batch_size"], shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(T_val)),
        CFG["batch_size"],
    )

    predictor = SSMQualityPredictor(
        event_dim=CFG["event_dim"],
        d_model=CFG["d_model"],
        n_workers=CFG["n_workers"],
        n_layers=CFG["n_layers"],
    )
    n_params = sum(p.numel() for p in predictor.parameters())
    print(f"  SSMQualityPredictor parameters: {n_params:,}")

    t0 = time.time()
    train_result = train_quality_predictor(predictor, train_loader, val_loader, CFG)
    train_time = time.time() - t0
    print(f"  Training done ({train_time:.1f}s)  best_val_mse={train_result['best_val_loss']:.5f}")

    # ── Evaluate SSM in-distribution ──
    print(f"\n[3/4] Evaluating SSMQualityPredictor in simulation (default pool)...")
    predictor.eval()
    t0 = time.time()
    ssm_result = run_simulation(
        "ssm", CFG["n_sim_episodes"], CFG["episode_length"],
        seed=SEED, ssm_model=predictor, workers_fn=default_worker_pool,
    )
    ssm_result["train_result"] = train_result
    ssm_result["n_params"]     = n_params
    ssm_result["train_time_s"] = round(train_time, 1)
    ssm_result["sim_time_s"]   = round(time.time() - t0, 2)
    report["in_distribution"]["ssm_predictor"] = ssm_result
    print(f"  quality={ssm_result['composite_quality']:.3f}  "
          f"lat={ssm_result['avg_latency_ms']:.0f}ms  "
          f"err={ssm_result['error_rate']:.3f}")

    # ── Distribution shift: heavy degradation pool ──
    print(f"\n[4/4] Distribution shift — heavy degradation pool "
          f"(3/5 workers degrade, not seen during training)...")
    for strategy in strategies + ["ssm_predictor"]:
        s_name = "ssm" if strategy == "ssm_predictor" else strategy
        m_name = strategy
        ssm_m  = predictor if strategy == "ssm_predictor" else None
        print(f"  {m_name} ({CFG['n_sim_episodes']} eps)...", end=" ", flush=True)
        t0 = time.time()
        result = run_simulation(
            s_name, CFG["n_sim_episodes"], CFG["episode_length"],
            seed=SEED + 1, ssm_model=ssm_m, workers_fn=heavy_degradation_pool,
        )
        result["sim_time_s"] = round(time.time() - t0, 2)
        report["distribution_shift"][m_name] = result
        print(f"quality={result['composite_quality']:.3f}  "
              f"lat={result['avg_latency_ms']:.0f}ms  "
              f"err={result['error_rate']:.3f}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\n  In-distribution (default pool):")
    print(f"  {'Strategy':<20} {'Quality':>8}  {'Latency':>10}  {'Error':>7}  {'Cost':>8}")
    print(f"  {'-' * 58}")
    for name in strategies + ["ssm_predictor"]:
        s = report["in_distribution"][name]
        marker = " ← SSM" if name == "ssm_predictor" else ""
        print(f"  {name:<20} {s['composite_quality']:>8.3f}  "
              f"{s['avg_latency_ms']:>10.1f}  {s['error_rate']:>7.3f}  "
              f"{s['avg_cost_usd']:>8.4f}{marker}")

    print(f"\n  Distribution shift (heavy degradation pool):")
    print(f"  {'Strategy':<20} {'Quality':>8}  {'Latency':>10}  {'Error':>7}")
    print(f"  {'-' * 50}")
    for name in strategies + ["ssm_predictor"]:
        s = report["distribution_shift"][name]
        marker = " ← SSM" if name == "ssm_predictor" else ""
        print(f"  {name:<20} {s['composite_quality']:>8.3f}  "
              f"{s['avg_latency_ms']:>10.1f}  {s['error_rate']:>7.3f}{marker}")

    rr_q   = report["in_distribution"]["round-robin"]["composite_quality"]
    ssm_q  = report["in_distribution"]["ssm_predictor"]["composite_quality"]
    delta_rr = ssm_q - rr_q

    rr_shift_q  = report["distribution_shift"]["round-robin"]["composite_quality"]
    ssm_shift_q = report["distribution_shift"]["ssm_predictor"]["composite_quality"]
    delta_shift  = ssm_shift_q - rr_shift_q

    report["deltas"] = {
        "ssm_vs_round_robin_in_dist":  round(delta_rr,    4),
        "ssm_vs_round_robin_shift":    round(delta_shift,  4),
    }

    integrate_t  = report["gate"]["integrate_threshold"]
    comparable_t = report["gate"]["comparable_threshold"]

    if delta_rr > integrate_t:
        verdict = (
            f"SSM WINS by {delta_rr:.1%} over round-robin (in-dist) — "
            f"exceeds {integrate_t:.0%} gate; validate on real traces before integration"
        )
    elif delta_rr > -comparable_t:
        verdict = (
            f"SSM COMPARABLE to round-robin (delta={delta_rr:+.1%}) — "
            f"within {comparable_t:.0%}; static strategies sufficient; "
            f"SSM preferred only where temporal state tracking adds independent value"
        )
    else:
        verdict = (
            f"STATIC STRATEGIES WIN (delta={delta_rr:+.1%}) — "
            f"SSM routing adds overhead without quality gain"
        )

    report["verdict"] = verdict
    print(f"\n  Delta SSM vs round-robin (in-dist):  {delta_rr:+.3f}")
    print(f"  Delta SSM vs round-robin (shift):    {delta_shift:+.3f}")
    print(f"\n{'=' * 60}")
    print(f"GATE VERDICT: {verdict}")
    print(f"{'=' * 60}")

    out = results_dir / "04_routing_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out}")
    print("\nAll experiments complete. See docs/experiment_results_analysis.md")


if __name__ == "__main__":
    main()
