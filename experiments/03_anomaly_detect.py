"""
Experiment 03 — Anomaly Detection in Agent Event Streams

Question: Can a Mamba-style SSM detect anomaly onset (stuck/degrading worker)
          earlier in the event stream than a simple threshold baseline?

Improvements over v1:
  - Variable anomaly onset (20–75% of sequence, not fixed at 50%)
  - Tuned threshold baseline (k swept on validation set, not single untuned value)
  - Distribution shift evaluation: test on onset positions not seen in training
  - FP-matched gate evaluation: SSM detection lag is compared at a false-positive
    rate matched to the baseline (a fixed 0.5 threshold favoured the more
    trigger-happy detector and overstated the lag advantage)

Dataset: synthetic agent pool events (data/generators.py)
Models:  SSMAnomalyDetector vs. ThresholdDetector (best-k, tuned on val)
Metrics:
  - AUC-ROC
  - Mean detection lag per onset-position bucket
  - False positive rate
  - Distribution shift: in-range vs out-of-range onset accuracy

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

CFG = {
    "n_samples": 1500,
    "seq_len": 64,
    "input_dim": POOL_FEATURE_DIM,
    "d_model": 32,
    "n_layers": 2,
    "batch_size": 32,
    "epochs": 25,
    "lr": 1e-3,
    # onset range for training data — 20%–65% of seq_len
    "onset_range_train": (0.20, 0.65),
    # onset range for distribution-shift test set — outside training range
    "onset_range_shift": (0.68, 0.80),
    "detection_threshold": 0.5,
}


# ── dataset construction ───────────────────────────────────────────────────────

def build_anomaly_dataset(
    n_samples: int,
    seq_len: int,
    onset_range: tuple[float, float],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build per-timestep anomaly detection dataset with variable onset.

    onset_range: (min_frac, max_frac) — onset position sampled uniformly
                 from [seq_len * min_frac, seq_len * max_frac)

    Returns:
        X:      (n_samples, seq_len, POOL_FEATURE_DIM)
        y:      (n_samples, seq_len)  — 0 = normal, 1 = anomaly
        onsets: (n_samples,)          — actual onset step per sample
    """
    rng = np.random.default_rng(seed)
    X_list, y_list, onset_list = [], [], []

    anomaly_types = [EventLabel.DEGRADING, EventLabel.STUCK]
    min_onset = int(seq_len * onset_range[0])
    max_onset = int(seq_len * onset_range[1])

    for i in range(n_samples):
        onset = int(rng.integers(min_onset, max_onset + 1))
        atype = anomaly_types[i % len(anomaly_types)]

        normal_part  = generate_pool_sequence(onset,           EventLabel.NORMAL)
        anomaly_part = generate_pool_sequence(seq_len - onset, atype)

        seq    = np.concatenate([normal_part, anomaly_part], axis=0)
        labels = np.zeros(seq_len, dtype=np.float32)
        labels[onset:] = 1.0

        X_list.append(seq)
        y_list.append(labels)
        onset_list.append(onset)

    return (
        np.stack(X_list).astype(np.float32),
        np.stack(y_list).astype(np.float32),
        np.array(onset_list, dtype=np.int32),
    )


# ── threshold baseline ─────────────────────────────────────────────────────────

class ThresholdDetector:
    """Rolling z-score anomaly detector."""

    def __init__(self, window: int = 8, k: float = 2.0):
        self.window = window
        self.k = k

    def score_sequence(self, x: np.ndarray) -> np.ndarray:
        L, _ = x.shape
        scores = np.zeros(L, dtype=np.float32)
        for t in range(self.window, L):
            mu    = x[:t].mean(axis=0)
            sigma = x[:t].std(axis=0) + 1e-6
            z     = np.abs((x[max(0, t - self.window):t].mean(axis=0) - mu) / sigma)
            scores[t] = min(float(z.max()) / (self.k * 3), 1.0)
        return scores

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        return np.stack([self.score_sequence(X[i]) for i in range(len(X))])


def tune_threshold_k(
    X_val: np.ndarray,
    y_val: np.ndarray,
    k_candidates: list[float] | None = None,
) -> tuple[float, float]:
    """
    Sweep k values on the validation set and return (best_k, best_auc).
    Replaces the single untuned k=2.0 from the previous version.
    """
    if k_candidates is None:
        k_candidates = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    best_k, best_auc = k_candidates[0], 0.0
    for k in k_candidates:
        det = ThresholdDetector(window=8, k=k)
        scores = det.score_batch(X_val)
        auc = compute_auc(y_val.flatten(), scores.flatten())
        if auc > best_auc:
            best_auc = auc
            best_k = k
    return best_k, best_auc


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
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                s = model(X_batch.to(device)).cpu().numpy().flatten()
                all_scores.extend(s.tolist())
                all_labels.extend(y_batch.numpy().flatten().tolist())
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


# ── metrics ────────────────────────────────────────────────────────────────────

def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    thresholds = np.linspace(0, 1, 51)
    tprs, fprs = [], []
    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        tprs.append(tp / (tp + fn + 1e-9))
        fprs.append(fp / (fp + tn + 1e-9))
    tprs = np.array(tprs)
    fprs = np.array(fprs)
    order = np.argsort(fprs)
    return float(np.trapezoid(tprs[order], fprs[order]))


def evaluate_detection_lag(
    scores: np.ndarray,
    threshold: float,
    onsets: np.ndarray,
) -> dict:
    """
    Per-sample onset aware evaluation.

    scores:  (n_samples, seq_len)
    onsets:  (n_samples,) — actual onset step for each sample
    """
    n = len(scores)
    lags, missed, false_positives = [], 0, 0

    for i in range(n):
        onset = int(onsets[i])
        if (scores[i, :onset] >= threshold).any():
            false_positives += 1
        flags_after = np.where(scores[i, onset:] >= threshold)[0]
        if len(flags_after) == 0:
            missed += 1
        else:
            lags.append(int(flags_after[0]))

    return {
        "mean_lag_steps": round(float(np.mean(lags)), 2) if lags else None,
        "missed_all": len(lags) == 0,
        "missed_rate": round(missed / n, 3),
        "false_positive_rate": round(false_positives / n, 3),
        "detected_n": len(lags),
    }


def evaluate_by_onset_bucket(
    scores: np.ndarray,
    threshold: float,
    onsets: np.ndarray,
    seq_len: int,
) -> dict:
    """
    Split test set by onset position into thirds and report lag per bucket.
    Reveals whether model degrades on early or late anomaly onsets.
    """
    buckets = {
        "early (≤33%)":  onsets <= seq_len * 0.33,
        "mid (33–66%)":  (onsets > seq_len * 0.33) & (onsets <= seq_len * 0.66),
        "late (>66%)":   onsets > seq_len * 0.66,
    }
    result = {}
    for name, mask in buckets.items():
        if mask.sum() == 0:
            continue
        lag_info = evaluate_detection_lag(scores[mask], threshold, onsets[mask])
        result[name] = {
            "n": int(mask.sum()),
            "mean_lag_steps": lag_info["mean_lag_steps"],
            "missed_rate": lag_info["missed_rate"],
        }
    return result


def sweep_detection_thresholds(
    scores: np.ndarray,
    onsets: np.ndarray,
    thresholds,
) -> list[dict]:
    """
    Evaluate FP rate and detection lag at multiple score thresholds.

    A raw lag comparison at one fixed threshold conflates sensitivity with
    speed: a trigger-happy detector shows a small lag but a high FP rate.
    This sweep exposes the tradeoff explicitly.
    """
    rows = []
    for t in thresholds:
        r = evaluate_detection_lag(scores, float(t), onsets)
        rows.append({
            "threshold": round(float(t), 3),
            "fp_rate": r["false_positive_rate"],
            "mean_lag_steps": r["mean_lag_steps"],
            "missed_rate": r["missed_rate"],
        })
    return rows


def pick_fp_matched(rows: list[dict], target_fp: float, tol: float = 0.02) -> dict | None:
    """
    Return the sweep row whose FP rate is closest to target_fp from below
    (within tolerance) — the SSM operating point comparable to the baseline.
    """
    eligible = [r for r in rows if r["fp_rate"] <= target_fp + tol]
    if not eligible:
        return None
    return min(eligible, key=lambda r: abs(r["fp_rate"] - target_fp))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Experiment 03 — Anomaly Detection in Agent Event Streams")
    print("=" * 60)

    seq_len = CFG["seq_len"]

    # ── Training + standard test set (onset in train range)
    print(f"\nGenerating {CFG['n_samples']} train/test sequences "
          f"(onset range {CFG['onset_range_train']})...")
    X, y, onsets = build_anomaly_dataset(
        CFG["n_samples"], seq_len, CFG["onset_range_train"], seed=SEED,
    )

    n_train = int(len(X) * 0.8)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    onsets_train, onsets_test = onsets[:n_train], onsets[n_train:]

    n_val = int(n_train * 0.15)
    X_val,   y_val   = X_train[-n_val:],   y_train[-n_val:]
    X_train, y_train = X_train[:-n_val],   y_train[:-n_val]

    # ── Distribution shift test set (onset outside training range)
    n_shift = 200
    print(f"Generating {n_shift} distribution-shift sequences "
          f"(onset range {CFG['onset_range_shift']})...")
    X_shift, y_shift, onsets_shift = build_anomaly_dataset(
        n_shift, seq_len, CFG["onset_range_shift"], seed=SEED + 99,
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        CFG["batch_size"], shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        CFG["batch_size"],
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        CFG["batch_size"],
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | "
          f"Test: {len(X_test)} | Shift: {n_shift}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    report = {
        "config": CFG,
        "device": device,
        "onset_range_train": CFG["onset_range_train"],
        "onset_range_shift": CFG["onset_range_shift"],
        "models": {},
    }

    # ── Tune threshold baseline on val set ──
    print("--- ThresholdDetector (tuned k on val set) ---")
    t0 = time.time()
    best_k, best_val_auc = tune_threshold_k(X_val, y_val)
    threshold_detector = ThresholdDetector(window=8, k=best_k)
    thresh_scores = threshold_detector.score_batch(X_test)
    thresh_shift_scores = threshold_detector.score_batch(X_shift)
    thresh_time = time.time() - t0

    thresh_auc   = compute_auc(y_test.flatten(),  thresh_scores.flatten())
    thresh_lag   = evaluate_detection_lag(thresh_scores,       CFG["detection_threshold"], onsets_test)
    thresh_buckets = evaluate_by_onset_bucket(thresh_scores, CFG["detection_threshold"], onsets_test, seq_len)
    thresh_shift_auc = compute_auc(y_shift.flatten(), thresh_shift_scores.flatten())
    thresh_shift_lag = evaluate_detection_lag(thresh_shift_scores, CFG["detection_threshold"], onsets_shift)

    print(f"  Best k={best_k} (val AUC={best_val_auc:.3f})  "
          f"Test AUC={thresh_auc:.3f}  "
          f"Lag={thresh_lag['mean_lag_steps'] if not thresh_lag['missed_all'] else 'missed all'}  "
          f"({thresh_time:.2f}s)")
    report["models"]["threshold"] = {
        "best_k": best_k,
        "val_auc": round(best_val_auc, 4),
        "test_auc": round(thresh_auc, 4),
        "detection_lag": thresh_lag,
        "onset_buckets": thresh_buckets,
        "shift_test_auc": round(thresh_shift_auc, 4),
        "shift_detection_lag": thresh_shift_lag,
    }

    # ── Train SSMAnomalyDetector ──
    print("\n--- SSMAnomalyDetector (Mamba-style) ---")
    ssm_model = SSMAnomalyDetector(
        input_dim=CFG["input_dim"],
        d_model=CFG["d_model"],
        n_layers=CFG["n_layers"],
    )
    n_params = sum(p.numel() for p in ssm_model.parameters())
    print(f"Parameters: {n_params:,}")

    t0 = time.time()
    ssm_train = train_anomaly_model(ssm_model, train_loader, val_loader, CFG)
    ssm_time  = time.time() - t0

    device_t = next(ssm_model.parameters()).device
    ssm_model.eval()

    def collect_scores(X_np):
        loader = DataLoader(
            TensorDataset(torch.tensor(X_np, dtype=torch.float32)),
            CFG["batch_size"],
        )
        parts = []
        with torch.no_grad():
            for (X_b,) in loader:
                parts.append(ssm_model(X_b.to(device_t)).cpu().numpy())
        return np.concatenate(parts, axis=0)

    ssm_scores       = collect_scores(X_test)
    ssm_shift_scores = collect_scores(X_shift)

    ssm_auc        = compute_auc(y_test.flatten(),  ssm_scores.flatten())
    ssm_lag        = evaluate_detection_lag(ssm_scores,       CFG["detection_threshold"], onsets_test)
    ssm_buckets    = evaluate_by_onset_bucket(ssm_scores, CFG["detection_threshold"], onsets_test, seq_len)
    ssm_shift_auc  = compute_auc(y_shift.flatten(), ssm_shift_scores.flatten())
    ssm_shift_lag  = evaluate_detection_lag(ssm_shift_scores, CFG["detection_threshold"], onsets_shift)

    # FP-matched operating point: compare lags when both detectors fire at a
    # similar FP rate, instead of a fixed 0.5 threshold that favours the more
    # trigger-happy model.
    threshold_grid = np.round(np.arange(0.30, 0.91, 0.05), 2)
    ssm_sweep = sweep_detection_thresholds(ssm_scores, onsets_test, threshold_grid)
    baseline_fp = thresh_lag["false_positive_rate"]
    ssm_fp_matched = pick_fp_matched(ssm_sweep, baseline_fp)

    print(f"Test AUC={ssm_auc:.3f}  "
          f"Lag={ssm_lag['mean_lag_steps'] if not ssm_lag['missed_all'] else 'missed all'}  "
          f"({ssm_time:.1f}s)")
    if ssm_fp_matched:
        print(f"  FP-matched (SSM thr={ssm_fp_matched['threshold']:.2f}, "
              f"FP={ssm_fp_matched['fp_rate']:.3f} vs baseline {baseline_fp:.3f}): "
              f"lag={ssm_fp_matched['mean_lag_steps']}")
    report["models"]["ssm"] = {
        **ssm_train,
        "test_auc": round(ssm_auc, 4),
        "detection_lag": ssm_lag,
        "onset_buckets": ssm_buckets,
        "shift_test_auc": round(ssm_shift_auc, 4),
        "shift_detection_lag": ssm_shift_lag,
        "threshold_sweep": ssm_sweep,
        "fp_matched": ssm_fp_matched,
        "params": n_params,
        "train_time_s": round(ssm_time, 1),
    }

    # ── Summary ──
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    def fmt_lag(info):
        return "missed all" if info["missed_all"] else f"{info['mean_lag_steps']:.2f}"

    print(f"\n  {'Model':<25} {'AUC':>6}  {'Lag':>12}  {'Miss':>6}  {'FP':>6}  "
          f"{'Shift AUC':>10}  {'Shift Lag':>12}")
    print(f"  {'-' * 80}")
    for name, key in [("SSMAnomalyDetector", "ssm"), ("ThresholdDetector", "threshold")]:
        m = report["models"][key]
        print(f"  {name:<25} {m['test_auc']:>6.3f}  {fmt_lag(m['detection_lag']):>12}  "
              f"{m['detection_lag']['missed_rate']:>6.3f}  "
              f"{m['detection_lag']['false_positive_rate']:>6.3f}  "
              f"{m['shift_test_auc']:>10.3f}  "
              f"{fmt_lag(m['shift_detection_lag']):>12}")

    print("\n  SSM onset-bucket breakdown (standard test):")
    for bucket, stats in ssm_buckets.items():
        lag_str = "missed all" if stats["mean_lag_steps"] is None else f"{stats['mean_lag_steps']:.2f}"
        print(f"    {bucket:<18} n={stats['n']:3d}  lag={lag_str}  miss={stats['missed_rate']:.3f}")

    # Verdict — gate: SSM detects ≥3 steps earlier than the tuned threshold.
    # The gate metric is the FP-matched lag delta; raw-0.5 numbers are shown for
    # reference but compare detectors at unequal false-positive rates.
    gate_threshold_steps = 3
    ssm_m   = report["models"]["ssm"]
    thr_m   = report["models"]["threshold"]

    thr_missed = thr_m["detection_lag"]["missed_all"]
    ssm_missed = ssm_m["detection_lag"]["missed_all"]
    thr_lag    = thr_m["detection_lag"]["mean_lag_steps"]
    thr_fp     = thr_m["detection_lag"]["false_positive_rate"]
    matched    = ssm_m.get("fp_matched")

    if ssm_missed:
        verdict = "SSM failed to detect — threshold rules are sufficient"
    elif thr_missed or thr_lag is None:
        verdict = (
            f"SSM detects (lag={ssm_m['detection_lag']['mean_lag_steps']:.2f} steps); "
            f"threshold missed all detections even after tuning — "
            f"promising on synthetic data; validate on real traces before production"
        )
    elif matched and matched["mean_lag_steps"] is not None:
        matched_delta = thr_lag - matched["mean_lag_steps"]
        raw_delta = thr_lag - ssm_m["detection_lag"]["mean_lag_steps"]
        if matched_delta >= gate_threshold_steps:
            verdict = (
                f"SSM detects {matched_delta:.1f} steps earlier at matched FP rate "
                f"(SSM thr={matched['threshold']:.2f}: FP={matched['fp_rate']:.3f} vs baseline {thr_fp:.3f}) — "
                f"meets gate (≥{gate_threshold_steps}); validate on real traces before production"
            )
        elif matched_delta > 0:
            verdict = (
                f"SSM detects {matched_delta:.1f} steps earlier at matched FP rate — BELOW gate "
                f"({gate_threshold_steps} steps). Raw-0.5 delta was {raw_delta:.1f} steps but at unequal "
                f"FP rates (SSM {ssm_m['detection_lag']['false_positive_rate']:.1%} vs baseline {thr_fp:.1%})"
            )
        else:
            verdict = "Threshold baseline matches or beats SSM at matched FP rate — static rules are sufficient"
    else:
        lag_delta = thr_lag - ssm_m["detection_lag"]["mean_lag_steps"]
        if lag_delta >= gate_threshold_steps:
            verdict = (
                f"SSM detects {lag_delta:.1f} steps earlier — meets gate (≥{gate_threshold_steps}); "
                f"validate on real traces before production integration"
            )
        elif lag_delta > 0:
            verdict = f"SSM detects {lag_delta:.1f} steps earlier — below gate ({gate_threshold_steps} steps)"
        else:
            verdict = "Threshold baseline matches or beats SSM — static rules are sufficient"

    report["verdict"] = verdict
    print(f"\nVerdict: {verdict}")

    out = results_dir / "03_anomaly_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out}")
    print("\nNext: python experiments/04_routing_sim.py")


if __name__ == "__main__":
    main()
