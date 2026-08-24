"""
Experiment 05 — TSE Trend Velocity Tracking

Question: Can a Mamba-style SSM classify trend velocity (rising / peaking /
          declining / noise) from 12-week sequences better than a simple
          slope-threshold baseline?

Motivation: TSE currently has no memory between weekly runs. An SSM that
            classifies velocity from the full 12-week trajectory could replace
            the current DECAY_WINDOW_DAYS heuristic with a learned model, and
            its compressed hidden state can be persisted between runs (small
            binary blob) instead of reprocessing all history.

Improvements over v1 (final audit fix):
  - The baseline's slope_threshold is now tuned on the validation set, exactly
    like the k-sweep added to Exp 03 in Phase 2. v1 hard-coded 0.08, which sits
    ~1.5σ above the generator's true rising/declining slope (0.8/11 ≈ 0.073),
    so the baseline systematically misclassified monotonic trends as noise and
    inflated the SSM delta from ~5pp to ~49pp.

Dataset: synthetic trend velocity sequences (data/generators.py)
Models:  SSMClassifier vs MovingAverageClassifier (tuned) vs fixed-0.08 reference
Metric:  4-class accuracy (rising / peaking / declining / noise)
         + per-class accuracy to see where each model struggles

Output: results/05_tse_velocity_report.json
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

from data.generators import generate_trend_velocity_dataset
from core.ssm import SSMClassifier

results_dir = Path(__file__).parent.parent / "results"
results_dir.mkdir(exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

CLASS_NAMES = ["rising", "peaking", "declining", "noise"]

# ── config ────────────────────────────────────────────────────────────────────
CFG = {
    "n_trends": 2000,
    "n_weeks": 12,
    "input_dim": 3,      # [cluster_size_norm, diversity_norm, score_norm]
    "n_classes": 4,
    "d_model": 32,
    "n_layers": 2,
    "batch_size": 32,
    "epochs": 40,
    "lr": 1e-3,
    "train_ratio": 0.8,
}


# ── moving average baseline ────────────────────────────────────────────────────

class MovingAverageClassifier:
    """
    Classify trend velocity from regression slopes.

    Decision order (uses cluster_size, column 0):
      1. overall slope (linear fit over ALL weeks) > +thr  → rising
      2. overall slope < -thr                              → declining
      3. early slope > +thr AND recent slope < -thr         → peaking
      4. otherwise                                          → noise
    """

    def __init__(self, window: int = 3, slope_threshold: float = 0.08):
        self.window = window
        self.slope_threshold = slope_threshold

    def classify_sequence(self, x: np.ndarray) -> int:
        """
        x: (n_weeks, 3) — uses cluster_size (column 0) as the primary signal
        returns: class index
        """
        signal = x[:, 0]  # cluster_size_norm

        # Overall slope: linear regression across all weeks
        t = np.arange(len(signal), dtype=float)
        overall_slope = float(np.polyfit(t, signal, 1)[0])

        # Recent slope: last `window` weeks
        recent = signal[-self.window:]
        recent_t = np.arange(len(recent), dtype=float)
        recent_slope = float(np.polyfit(recent_t, recent, 1)[0])

        # Early slope: first `window` weeks
        early = signal[:self.window]
        early_t = np.arange(len(early), dtype=float)
        early_slope = float(np.polyfit(early_t, early, 1)[0])

        # Classify
        if overall_slope > self.slope_threshold:
            return 0  # rising
        elif overall_slope < -self.slope_threshold:
            return 2  # declining
        elif early_slope > self.slope_threshold and recent_slope < -self.slope_threshold:
            return 1  # peaking (rose then fell)
        else:
            return 3  # noise

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """X: (N, n_weeks, 3) → (N,) predictions"""
        return np.array([self.classify_sequence(X[i]) for i in range(len(X))])


def tune_slope_threshold(
    X_val: np.ndarray,
    y_val: np.ndarray,
    window: int = 3,
    candidates: list[float] | None = None,
) -> tuple[float, float]:
    """
    Sweep slope_threshold on the validation set, return (best_thr, best_val_acc).

    Candidates are ordered descending so that on ties the more conservative
    (larger) threshold wins.
    """
    if candidates is None:
        candidates = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.045,
                      0.04, 0.03, 0.02, 0.015, 0.01, 0.005]
    best_thr, best_acc = candidates[0], -1.0
    for thr in candidates:
        preds = MovingAverageClassifier(window, thr).predict_batch(X_val)
        acc = float((preds == y_val).mean())
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return best_thr, best_acc


# ── training ──────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, cfg: dict) -> dict:
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

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:2d}/{cfg['epochs']}  "
                  f"loss={avg_loss:.4f}  val_acc={val_acc:.3f}")

    return {"best_val_acc": best_val_acc, "history": history}


def evaluate(model_or_fn, test_loader_or_X, y_test=None, is_sklearn=False):
    """Unified evaluator for SSM model and MA baseline."""
    if is_sklearn:
        # MA baseline: model_or_fn is the classifier, test_loader_or_X is X_test np array
        preds = model_or_fn.predict_batch(test_loader_or_X)
        labels = y_test
    else:
        device = next(model_or_fn.parameters()).device
        model_or_fn.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader_or_X:
                X_batch = X_batch.to(device)
                p = model_or_fn(X_batch).argmax(dim=-1).cpu().numpy()
                all_preds.extend(p.tolist())
                all_labels.extend(y_batch.tolist())
        preds = np.array(all_preds)
        labels = np.array(all_labels)

    accuracy = float((preds == labels).mean())
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = labels == i
        if mask.sum() > 0:
            per_class[name] = round(float((preds[mask] == labels[mask]).mean()), 3)

    return {"accuracy": round(accuracy, 4), "per_class": per_class}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Experiment 05 — TSE Trend Velocity Tracking")
    print("=" * 60)

    print(f"\nGenerating {CFG['n_trends']} trend sequences ({CFG['n_weeks']} weeks each)...")
    X, y = generate_trend_velocity_dataset(CFG["n_trends"], CFG["n_weeks"], seed=SEED)
    print(f"  Classes: {', '.join(f'{n}={int((y==i).sum())}' for i, n in enumerate(CLASS_NAMES))}")

    # Split
    n_train = int(len(X) * CFG["train_ratio"])
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    n_val = int(n_train * 0.15)
    X_val, y_val = X_train[-n_val:], y_train[-n_val:]
    X_train_fit, y_train_fit = X_train[:-n_val], y_train[:-n_val]

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_fit, dtype=torch.float32),
            torch.tensor(y_train_fit, dtype=torch.long),
        ),
        CFG["batch_size"], shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        ),
        CFG["batch_size"],
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        ),
        CFG["batch_size"],
    )

    print(f"Train: {len(X_train_fit)} | Val: {len(X_val)} | Test: {len(X_test)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    report = {"config": CFG, "device": device, "models": {}}

    # ── Moving Average baseline (threshold tuned on val) ──
    print("--- MovingAverageClassifier: tune slope_threshold on val set ---")
    best_thr, best_val_acc = tune_slope_threshold(X_val, y_val)
    print(f"  Best slope_threshold={best_thr} (val acc={best_val_acc:.4f})")

    ma_untuned = MovingAverageClassifier(window=3, slope_threshold=0.08)
    ma_untuned_eval = evaluate(ma_untuned, X_test, y_test, is_sklearn=True)
    print(f"\n--- MovingAverageClassifier (fixed thr=0.08, v1 reference) ---")
    print(f"Test accuracy: {ma_untuned_eval['accuracy']:.3f}")
    report["models"]["moving_average_fixed_0.08"] = {
        **ma_untuned_eval, "window": 3, "slope_threshold": 0.08,
        "note": "v1 configuration — not tuned, kept for reference",
    }

    ma = MovingAverageClassifier(window=3, slope_threshold=best_thr)
    t0 = time.time()
    ma_eval = evaluate(ma, X_test, y_test, is_sklearn=True)
    ma_time = time.time() - t0
    print(f"\n--- MovingAverageClassifier (tuned thr={best_thr}) ---")
    print(f"Test accuracy: {ma_eval['accuracy']:.3f}  ({ma_time:.3f}s)")
    print("Per-class:")
    for cls, acc in ma_eval["per_class"].items():
        print(f"  {cls:10s}: {acc:.3f}")
    report["models"]["moving_average_tuned"] = {
        **ma_eval, "window": 3, "slope_threshold": best_thr, "val_acc": round(best_val_acc, 4),
    }

    # ── SSMClassifier ──
    print("\n--- SSMClassifier (Mamba-style, 12-week sequences) ---")
    ssm_model = SSMClassifier(
        input_dim=CFG["input_dim"],
        d_model=CFG["d_model"],
        n_classes=CFG["n_classes"],
        n_layers=CFG["n_layers"],
    )
    n_params = sum(p.numel() for p in ssm_model.parameters())
    print(f"Parameters: {n_params:,}")
    t0 = time.time()
    ssm_train = train_model(ssm_model, train_loader, val_loader, CFG)
    ssm_time = time.time() - t0
    ssm_eval = evaluate(ssm_model, test_loader)
    print(f"Test accuracy: {ssm_eval['accuracy']:.3f}  ({ssm_time:.1f}s)")
    print("Per-class:")
    for cls, acc in ssm_eval["per_class"].items():
        print(f"  {cls:10s}: {acc:.3f}")
    report["models"]["ssm"] = {
        **ssm_train, **ssm_eval,
        "params": n_params,
        "train_time_s": round(ssm_time, 1),
    }

    # ── Summary ──
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    ssm_acc = report["models"]["ssm"]["accuracy"]
    ma_acc = report["models"]["moving_average_tuned"]["accuracy"]
    ma_untuned_acc = report["models"]["moving_average_fixed_0.08"]["accuracy"]
    delta = ssm_acc - ma_acc

    print(f"\n  {'Model':<34} {'Accuracy':>10}")
    print(f"  {'-' * 46}")
    print(f"  {'SSMClassifier':<34} {ssm_acc:>10.3f}")
    print(f"  {'MovingAverage (tuned thr=' + str(best_thr) + ')':<34} {ma_acc:>10.3f}")
    print(f"  {'MovingAverage (fixed thr=0.08, v1)':<34} {ma_untuned_acc:>10.3f}")
    print(f"\n  Delta (SSM - tuned MA): {delta:+.3f}   [gate: > 0.10]")

    print(f"\n  Per-class comparison (tuned MA):")
    print(f"  {'Class':<12} {'SSM':>8}  {'MA':>8}  {'Delta':>8}")
    print(f"  {'-' * 42}")
    for cls in CLASS_NAMES:
        s = report["models"]["ssm"]["per_class"].get(cls, 0)
        m = report["models"]["moving_average_tuned"]["per_class"].get(cls, 0)
        print(f"  {cls:<12} {s:>8.3f}  {m:>8.3f}  {s-m:>+8.3f}")

    if delta > 0.10:
        verdict = (
            f"SSM beats tuned baseline by {delta:+.1%} — above the 10% gate; "
            f"candidate for TSE velocity tracker; validate on real TSE history"
        )
    elif delta > -0.05:
        verdict = (
            f"SSM comparable to tuned baseline ({delta:+.1%}) — below the 10% gate; "
            f"tuned slope classifier is sufficient; no integration"
        )
    else:
        verdict = (
            f"Tuned baseline wins ({delta:+.1%}) — "
            f"SSM does not add value for trend velocity on this data"
        )

    report["verdict"] = verdict
    report["delta_ssm_vs_tuned_ma"] = round(delta, 4)
    report["delta_ssm_vs_fixed_ma_v1"] = round(ssm_acc - ma_untuned_acc, 4)
    print(f"\nVerdict: {verdict}")

    out = results_dir / "05_tse_velocity_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out}")
    print("\nAll playground experiments complete.")
    print("See docs/experiment_results_analysis.md for the full integration decision.")


if __name__ == "__main__":
    main()
