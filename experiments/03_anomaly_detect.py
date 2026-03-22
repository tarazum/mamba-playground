"""
Experiment 03 — Anomaly Detection in Agent Event Streams

Question: Can a Mamba-style SSM detect anomaly onset (stuck/degrading worker)
          earlier in the event stream than a simple threshold baseline?

Dataset: synthetic agent pool events (data/generators.py)
Models:  SSMAnomalyDetector vs. ThresholdDetector (rolling mean baseline)
Metric:
  - Detection lag: how many events AFTER anomaly onset does each method detect it?
  - False positive rate: anomalies flagged on normal sequences
  - AUC-ROC across detection thresholds

Output: results/03_anomaly_report.json
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

from data.generators import generate_pool_sequence, EventLabel, POOL_FEATURE_DIM
from core.ssm import SSMAnomalyDetector

results_dir = Path(__file__).parent.parent / "results"
results_dir.mkdir(exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── config ────────────────────────────────────────────────────────────────────
CFG = {
    "n_samples": 1200,
    "seq_len": 64,
    "input_dim": POOL_FEATURE_DIM,   # 6
    "d_model": 32,
    "n_layers": 2,
    "batch_size": 32,
    "epochs": 25,
    "lr": 1e-3,
    # anomaly onset: we inject anomaly at the midpoint of each sequence
    "anomaly_onset_frac": 0.5,       # anomaly begins at seq_len * 0.5
    "detection_threshold": 0.5,      # score > this = anomaly flagged
}

ANOMALY_ONSET = int(CFG["seq_len"] * CFG["anomaly_onset_frac"])


# ── dataset construction ───────────────────────────────────────────────────────

def build_anomaly_dataset(n_samples: int, seq_len: int, seed: int = 42):
    """
    Build a per-timestep anomaly detection dataset.

    Each sequence is split:
      - first half: normal events
      - second half: either stuck or degrading events (anomaly)

    Labels: 0 = normal, 1 = anomaly (per timestep)

    Returns:
        X: (n_samples, seq_len, POOL_FEATURE_DIM)
        y: (n_samples, seq_len)   — binary, anomaly starts at ANOMALY_ONSET
        onset: (n_samples,) actual onset step (all = ANOMALY_ONSET here)
    """
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    anomaly_types = [EventLabel.DEGRADING, EventLabel.STUCK]

    for i in range(n_samples):
        # Normal prefix
        normal_len = ANOMALY_ONSET
        normal_part = generate_pool_sequence(normal_len, EventLabel.NORMAL)

        # Anomaly suffix
        atype = anomaly_types[i % len(anomaly_types)]
        anomaly_len = seq_len - normal_len
        anomaly_part = generate_pool_sequence(anomaly_len, atype)

        seq = np.concatenate([normal_part, anomaly_part], axis=0)
        labels = np.zeros(seq_len, dtype=np.float32)
        labels[normal_len:] = 1.0

        X_list.append(seq)
        y_list.append(labels)

    X = np.stack(X_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)
    return X, y


# ── threshold baseline ─────────────────────────────────────────────────────────

class ThresholdDetector:
    """
    Rolling-window baseline: flag anomaly when the rolling mean of any feature
    deviates by > k sigma from the sequence mean up to that point.

    Returns per-timestep anomaly scores in [0, 1].
    """

    def __init__(self, window: int = 8, k: float = 2.0):
        self.window = window
        self.k = k

    def score_sequence(self, x: np.ndarray) -> np.ndarray:
        """
        x: (seq_len, feature_dim)
        returns: (seq_len,) scores
        """
        L, D = x.shape
        scores = np.zeros(L, dtype=np.float32)

        for t in range(self.window, L):
            window_data = x[max(0, t - self.window):t]
            prefix_data = x[:t]

            mu = prefix_data.mean(axis=0)
            sigma = prefix_data.std(axis=0) + 1e-6

            # deviation of current window mean from prefix mean
            window_mu = window_data.mean(axis=0)
            z_scores = np.abs((window_mu - mu) / sigma)

            # max z-score across features → scalar anomaly score
            scores[t] = min(float(z_scores.max()) / (self.k * 3), 1.0)

        return scores

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """X: (B, L, D) → (B, L) scores"""
        return np.stack([self.score_sequence(X[i]) for i in range(len(X))])


# ── training ──────────────────────────────────────────────────────────────────

def train_anomaly_model(model, train_loader, val_loader, cfg: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"])

    history = {"train_loss": [], "val_auc": []}
    best_val_auc = 0.0

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            scores = model(X_batch)          # (B, L)
            loss = criterion(scores, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # validation AUC (simplified: threshold sweep)
        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                s = model(X_batch).cpu().numpy().flatten()
                l = y_batch.numpy().flatten()
                all_scores.extend(s.tolist())
                all_labels.extend(l.tolist())

        val_auc = compute_auc(np.array(all_labels), np.array(all_scores))
        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(round(avg_loss, 4))
        history["val_auc"].append(round(val_auc, 4))

        if val_auc > best_val_auc:
            best_val_auc = val_auc

        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:2d}/{cfg['epochs']}  "
                  f"loss={avg_loss:.4f}  val_auc={val_auc:.3f}")

    return {"best_val_auc": best_val_auc, "history": history}


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUC-ROC via trapezoidal rule (no sklearn dependency)."""
    thresholds = np.linspace(0, 1, 51)
    tprs, fprs = [], []
    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        tpr = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        tprs.append(tpr)
        fprs.append(fpr)
    tprs = np.array(tprs)
    fprs = np.array(fprs)
    order = np.argsort(fprs)
    return float(np.trapz(tprs[order], fprs[order]))


def evaluate_detection_lag(
    scores: np.ndarray,
    threshold: float,
    onset: int,
) -> dict:
    """
    scores: (n_samples, seq_len)
    Returns mean detection lag (steps after onset until first flag),
    missed detection rate, and false positive rate.
    """
    n = len(scores)
    lags = []
    missed = 0
    false_positives = 0

    for i in range(n):
        # False positive check: any flag before onset?
        if (scores[i, :onset] >= threshold).any():
            false_positives += 1

        # Detection lag: first flag at or after onset
        flags_after = np.where(scores[i, onset:] >= threshold)[0]
        if len(flags_after) == 0:
            missed += 1
        else:
            lags.append(int(flags_after[0]))

    return {
        "mean_lag_steps": round(float(np.mean(lags)) if lags else float("inf"), 2),
        "missed_rate": round(missed / n, 3),
        "false_positive_rate": round(false_positives / n, 3),
        "detected_n": len(lags),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Experiment 03 — Anomaly Detection in Agent Event Streams")
    print("=" * 60)

    print(f"\nGenerating {CFG['n_samples']} sequences (seq_len={CFG['seq_len']})...")
    print(f"Anomaly onset at step {ANOMALY_ONSET} (50% of sequence)")
    X, y = build_anomaly_dataset(CFG["n_samples"], CFG["seq_len"], seed=SEED)

    # Split
    n_train = int(len(X) * 0.8)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    n_val = int(n_train * 0.15)
    X_val, y_val = X_train[-n_val:], y_train[-n_val:]
    X_train, y_train = X_train[:-n_val], y_train[:-n_val]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), CFG["batch_size"], shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), CFG["batch_size"])
    test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), CFG["batch_size"])

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    report = {
        "config": CFG,
        "device": device,
        "anomaly_onset_step": ANOMALY_ONSET,
        "models": {},
    }

    # ── Train SSMAnomalyDetector ──
    print("--- SSMAnomalyDetector (Mamba-style) ---")
    ssm_model = SSMAnomalyDetector(
        input_dim=CFG["input_dim"],
        d_model=CFG["d_model"],
        n_layers=CFG["n_layers"],
    )
    n_params = sum(p.numel() for p in ssm_model.parameters())
    print(f"Parameters: {n_params:,}")
    t0 = time.time()
    ssm_train = train_anomaly_model(ssm_model, train_loader, val_loader, CFG)
    ssm_time = time.time() - t0

    # Collect test scores
    ssm_model.eval()
    ssm_scores = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            s = ssm_model(X_batch.to(next(ssm_model.parameters()).device)).cpu().numpy()
            ssm_scores.append(s)
    ssm_scores = np.concatenate(ssm_scores, axis=0)

    ssm_auc = compute_auc(y_test.flatten(), ssm_scores.flatten())
    ssm_lag = evaluate_detection_lag(ssm_scores, CFG["detection_threshold"], ANOMALY_ONSET)
    print(f"Test AUC: {ssm_auc:.3f}  Mean lag: {ssm_lag['mean_lag_steps']} steps  ({ssm_time:.1f}s)")
    report["models"]["ssm"] = {
        **ssm_train,
        "test_auc": round(ssm_auc, 4),
        "detection_lag": ssm_lag,
        "params": n_params,
        "train_time_s": round(ssm_time, 1),
    }

    # ── Threshold baseline ──
    print("\n--- ThresholdDetector (rolling z-score baseline) ---")
    threshold_detector = ThresholdDetector(window=8, k=2.0)
    t0 = time.time()
    thresh_scores = threshold_detector.score_batch(X_test)
    thresh_time = time.time() - t0

    thresh_auc = compute_auc(y_test.flatten(), thresh_scores.flatten())
    thresh_lag = evaluate_detection_lag(thresh_scores, CFG["detection_threshold"], ANOMALY_ONSET)
    print(f"Test AUC: {thresh_auc:.3f}  Mean lag: {thresh_lag['mean_lag_steps']} steps  ({thresh_time:.3f}s)")
    report["models"]["threshold"] = {
        "test_auc": round(thresh_auc, 4),
        "detection_lag": thresh_lag,
        "window": 8,
        "k_sigma": 2.0,
        "train_time_s": round(thresh_time, 3),
    }

    # ── Summary ──
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    ssm_auc_v = report["models"]["ssm"]["test_auc"]
    thr_auc_v = report["models"]["threshold"]["test_auc"]
    ssm_lag_v = report["models"]["ssm"]["detection_lag"]["mean_lag_steps"]
    thr_lag_v = report["models"]["threshold"]["detection_lag"]["mean_lag_steps"]

    print(f"  {'Model':<25} {'AUC':>6}  {'Mean lag (steps)':>18}  {'Miss rate':>10}  {'FP rate':>8}")
    print(f"  {'-'*70}")
    print(f"  {'SSMAnomalyDetector':<25} {ssm_auc_v:>6.3f}  {ssm_lag_v:>18.2f}  "
          f"{report['models']['ssm']['detection_lag']['missed_rate']:>10.3f}  "
          f"{report['models']['ssm']['detection_lag']['false_positive_rate']:>8.3f}")
    print(f"  {'ThresholdDetector':<25} {thr_auc_v:>6.3f}  {thr_lag_v:>18.2f}  "
          f"{report['models']['threshold']['detection_lag']['missed_rate']:>10.3f}  "
          f"{report['models']['threshold']['detection_lag']['false_positive_rate']:>8.3f}")

    lag_improvement = thr_lag_v - ssm_lag_v  # positive = SSM detects earlier

    gate_threshold = 3  # secondary gate: SSM must detect 3+ steps earlier (CLAUDE.md)
    if lag_improvement >= gate_threshold:
        verdict = (f"SSM detects {lag_improvement:.1f} steps earlier — "
                   f"meets gate (≥{gate_threshold} steps) → integrate into agent-pool health loop")
    elif lag_improvement > 0:
        verdict = (f"SSM detects {lag_improvement:.1f} steps earlier — "
                   f"below gate ({gate_threshold} steps) — marginal improvement, consider static rules")
    else:
        verdict = (f"Threshold baseline matches or beats SSM (lag delta={lag_improvement:.1f}) — "
                   f"static rules are sufficient for anomaly detection")

    report["verdict"] = verdict
    print(f"\nGate: detect stuck workers >{gate_threshold} events before timeout")
    print(f"Verdict: {verdict}")

    out = results_dir / "03_anomaly_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out}")
    print("\nNext: python experiments/04_routing_sim.py")


if __name__ == "__main__":
    main()
