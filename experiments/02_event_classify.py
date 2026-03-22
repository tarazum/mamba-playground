"""
Experiment 02 — Event Stream Classification

Question: Can a Mamba-style SSM classify agent pool event sequences as
          normal / degrading / stuck better than a simple LSTM baseline?

Dataset: synthetic agent pool events (data/generators.py)
Models:  SSMClassifier (core/ssm.py), LSTMClassifier (baseline)
Metric:  classification accuracy on held-out test set

Output:  results/02_classify_report.json
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generators import generate_pool_dataset
from core.ssm import SSMClassifier

results_dir = Path(__file__).parent.parent / "results"
results_dir.mkdir(exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── config ────────────────────────────────────────────────────────────────────
CFG = {
    "n_samples": 1200,
    "seq_len": 64,
    "input_dim": 6,       # POOL_FEATURE_DIM
    "n_classes": 3,
    "d_model": 32,
    "n_layers": 2,
    "batch_size": 32,
    "epochs": 30,
    "lr": 1e-3,
    "train_ratio": 0.8,
}


# ── baseline LSTM ─────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.classifier(h[-1])


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

        # validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch).argmax(dim=-1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)
        val_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(round(avg_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:2d}/{cfg['epochs']}  "
                  f"loss={avg_loss:.4f}  val_acc={val_acc:.3f}")

    return {"best_val_acc": best_val_acc, "history": history}


def evaluate(model, test_loader) -> dict:
    device = next(model.parameters()).device
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).argmax(dim=-1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(y_batch.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = float((all_preds == all_labels).mean())

    # per-class accuracy
    class_names = ["normal", "degrading", "stuck"]
    per_class = {}
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class[name] = round(float((all_preds[mask] == all_labels[mask]).mean()), 3)

    return {"accuracy": round(accuracy, 4), "per_class": per_class}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Experiment 02 — Event Stream Classification")
    print("=" * 60)

    # Generate data
    print(f"\nGenerating {CFG['n_samples']} sequences (seq_len={CFG['seq_len']})...")
    X, y = generate_pool_dataset(CFG["n_samples"], CFG["seq_len"], seed=SEED)

    # Split
    n_train = int(len(X) * CFG["train_ratio"])
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Further split train into train/val
    n_val = int(n_train * 0.15)
    X_val, y_val = X_train[-n_val:], y_train[-n_val:]
    X_train, y_train = X_train[:-n_val], y_train[:-n_val]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), CFG["batch_size"], shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), CFG["batch_size"])
    test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), CFG["batch_size"])

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    report = {"config": CFG, "device": device, "models": {}}

    # ── Train SSM ──
    print("--- SSMClassifier (Mamba-style, CPU-compatible) ---")
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
    report["models"]["ssm"] = {**ssm_train, **ssm_eval, "params": n_params, "train_time_s": round(ssm_time, 1)}

    # ── Train LSTM baseline ──
    print("\n--- LSTMClassifier (baseline) ---")
    lstm_model = LSTMClassifier(CFG["input_dim"], CFG["d_model"], CFG["n_classes"])
    n_params_lstm = sum(p.numel() for p in lstm_model.parameters())
    print(f"Parameters: {n_params_lstm:,}")
    t0 = time.time()
    lstm_train = train_model(lstm_model, train_loader, val_loader, CFG)
    lstm_time = time.time() - t0
    lstm_eval = evaluate(lstm_model, test_loader)
    print(f"Test accuracy: {lstm_eval['accuracy']:.3f}  ({lstm_time:.1f}s)")
    report["models"]["lstm"] = {**lstm_train, **lstm_eval, "params": n_params_lstm, "train_time_s": round(lstm_time, 1)}

    # ── Summary ──
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    ssm_acc = report["models"]["ssm"]["accuracy"]
    lstm_acc = report["models"]["lstm"]["accuracy"]
    delta = ssm_acc - lstm_acc
    print(f"  SSMClassifier :  {ssm_acc:.3f}")
    print(f"  LSTMClassifier:  {lstm_acc:.3f}")
    print(f"  Delta (SSM-LSTM): {delta:+.3f}")

    print("\nPer-class (SSM):")
    for cls, acc in report["models"]["ssm"]["per_class"].items():
        print(f"  {cls:10s}: {acc:.3f}")

    if delta > 0.05:
        verdict = "SSM clearly better — worth integrating into agent-pool router"
    elif delta > -0.05:
        verdict = "SSM comparable to LSTM — both viable; SSM preferred for streaming"
    else:
        verdict = "LSTM wins on this task — reconsider SSM for classification use case"

    report["verdict"] = verdict
    print(f"\nVerdict: {verdict}")

    out = results_dir / "02_classify_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out}")
    print("\nNext: python experiments/03_anomaly_detect.py")


if __name__ == "__main__":
    main()
