"""
evaluate.py — Milestone 2: Efficiency Metrics & Full Evaluation
INSE 6450: AI in Systems Engineering — Winter 2026
Student: Ismail Mzouri (40335670)

What this script does:
  1. Loads the trained MLP model from models/
  2. Runs full classification metrics on the test set
     (accuracy, precision, recall, F1-macro/micro, AUROC, PR-AUC)
  3. Measures training efficiency (model size, parameter count, FLOPS)
  4. Measures inference efficiency:
       - p50 / p90 per-sample latency
       - Batch latency
       - Throughput (samples/sec)
       - Peak RAM at inference
  5. Saves all results to results/efficiency_metrics.json
  6. Prints a full summary table

Usage:
  python src/evaluate.py

Prerequisites:
  Run src/train.py first to generate model artifacts.

Outputs:
  results/efficiency_metrics.json
  results/roc_curves.png
  results/pr_curves.png
"""

import os
import json
import time
import tracemalloc

import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
from scipy.sparse import load_npz, issparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "features.npz")
LABELS_PATH   = os.path.join(BASE_DIR, "data", "processed", "labels.csv")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42

# ──────────────────────────────────────────────
# MLP DEFINITION (must match train.py exactly)
# ──────────────────────────────────────────────
class EmailMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ──────────────────────────────────────────────
# 1. LOAD ARTIFACTS
# ──────────────────────────────────────────────
def load_artifacts():
    print("[1/5] Loading model artifacts...")
    config_path = os.path.join(MODELS_DIR, "mlp_model_config.json")
    with open(config_path) as f:
        config = json.load(f)

    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmailMLP(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, "mlp_model.pth"), map_location=device)
    )
    model.eval()

    print(f"    Model loaded on: {device}")
    print(f"    Classes: {list(le.classes_)}")
    return model, config, le, device


# ──────────────────────────────────────────────
# 2. REBUILD TEST SET
# ──────────────────────────────────────────────
def get_test_set(le, config):
    print("[2/5] Rebuilding test split...")
    X = load_npz(FEATURES_PATH)
    labels_df = pd.read_csv(LABELS_PATH)
    label_col = "label" if "label" in labels_df.columns else labels_df.columns[0]
    y = le.transform(labels_df[label_col].values)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=SEED, stratify=y
    )
    print(f"    Test set: {X_test.shape[0]:,} samples")
    return X_test, y_test


# ──────────────────────────────────────────────
# 3. CLASSIFICATION METRICS
# ──────────────────────────────────────────────
def compute_classification_metrics(model, X_test, y_test, le, device, config):
    print("[3/5] Computing classification metrics...")

    if issparse(X_test):
        X_dense = X_test.toarray()
    else:
        X_dense = X_test

    X_tensor = torch.tensor(X_dense, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=config["batch_size"], shuffle=False)

    all_logits = []
    with torch.no_grad():
        for (batch,) in loader:
            all_logits.append(model(batch.to(device)).cpu())
    logits = torch.cat(all_logits, dim=0)
    probs  = torch.softmax(logits, dim=1).numpy()
    preds  = logits.argmax(dim=1).numpy()

    acc        = accuracy_score(y_test, preds)
    prec_macro = precision_score(y_test, preds, average="macro",  zero_division=0)
    prec_micro = precision_score(y_test, preds, average="micro",  zero_division=0)
    rec_macro  = recall_score(y_test, preds,    average="macro",  zero_division=0)
    rec_micro  = recall_score(y_test, preds,    average="micro",  zero_division=0)
    f1_macro   = f1_score(y_test, preds,        average="macro",  zero_division=0)
    f1_micro   = f1_score(y_test, preds,        average="micro",  zero_division=0)

    # AUROC & PR-AUC (one-vs-rest)
    y_bin = label_binarize(y_test, classes=list(range(config["num_classes"])))
    auroc  = roc_auc_score(y_bin, probs, multi_class="ovr", average="macro")
    pr_auc = average_precision_score(y_bin, probs, average="macro")

    metrics = {
        "accuracy":          round(float(acc),        4),
        "precision_macro":   round(float(prec_macro),  4),
        "precision_micro":   round(float(prec_micro),  4),
        "recall_macro":      round(float(rec_macro),   4),
        "recall_micro":      round(float(rec_micro),   4),
        "f1_macro":          round(float(f1_macro),    4),
        "f1_micro":          round(float(f1_micro),    4),
        "auroc_macro_ovr":   round(float(auroc),       4),
        "pr_auc_macro":      round(float(pr_auc),      4),
    }

    print(f"    Accuracy:        {acc:.4f}")
    print(f"    F1 Macro:        {f1_macro:.4f}  |  F1 Micro: {f1_micro:.4f}")
    print(f"    AUROC (macro):   {auroc:.4f}  |  PR-AUC:   {pr_auc:.4f}")

    return metrics, preds, probs, y_bin


# ──────────────────────────────────────────────
# 4. EFFICIENCY METRICS
# ──────────────────────────────────────────────
def compute_efficiency_metrics(model, X_test, config, device):
    print("[4/5] Measuring inference efficiency...")

    if issparse(X_test):
        X_dense = X_test.toarray()
    else:
        X_dense = X_test

    # ── Model size on disk ──
    model_path = os.path.join(MODELS_DIR, "mlp_model.pth")
    model_size_mb = os.path.getsize(model_path) / (1024 ** 2)

    # ── Parameter count ──
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── FLOPS estimate (manual for linear layers) ──
    # For a linear layer: FLOPs = 2 * in_features * out_features
    flops = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            flops += 2 * module.in_features * module.out_features
    flops_gflops = flops / 1e9

    # ── Per-sample latency (p50 / p90) ──
    # Warm up the model first
    sample = torch.tensor(X_dense[:1], dtype=torch.float32).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample)

    N_LATENCY = min(500, len(X_dense))
    latencies = []
    model.eval()
    with torch.no_grad():
        for i in range(N_LATENCY):
            x = torch.tensor(X_dense[i:i+1], dtype=torch.float32).to(device)
            t0 = time.perf_counter()
            _ = model(x)
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

    p50_ms = float(np.percentile(latencies, 50))
    p90_ms = float(np.percentile(latencies, 90))
    p95_ms = float(np.percentile(latencies, 95))

    # ── Batch latency & throughput ──
    BATCH_SIZE = config["batch_size"]
    X_batch = torch.tensor(X_dense[:BATCH_SIZE], dtype=torch.float32).to(device)
    with torch.no_grad():
        t0 = time.perf_counter()
        _ = model(X_batch)
        batch_latency_ms = (time.perf_counter() - t0) * 1000
    throughput_sps = BATCH_SIZE / (batch_latency_ms / 1000)

    # ── Peak RAM at inference ──
    tracemalloc.start()
    X_full = torch.tensor(X_dense, dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(X_full), batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for (b,) in loader:
            _ = model(b)
    _, peak_ram_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = peak_ram_bytes / (1024 ** 2)

    # ── Load training config for training efficiency stats ──
    config_path = os.path.join(MODELS_DIR, "mlp_model_config.json")
    with open(config_path) as f:
        train_config = json.load(f)

    efficiency = {
        "hardware": {
            "device":   str(device),
            "cpu":      "CPU (no GPU)",
            "note":     "All measurements on CPU unless CUDA available",
        },
        "training_efficiency": {
            "total_training_time_sec":  train_config.get("total_training_time_sec", "N/A"),
            "avg_time_per_epoch_sec":   train_config.get("avg_time_per_epoch_sec", "N/A"),
            "actual_epochs_trained":    train_config.get("actual_epochs_trained", "N/A"),
            "batch_size":               train_config.get("batch_size", "N/A"),
        },
        "model_size": {
            "on_disk_mb":           round(model_size_mb, 4),
            "total_parameters":     total_params,
            "trainable_parameters": trainable_params,
            "estimated_flops_per_sample": flops,
            "estimated_gflops":     round(flops_gflops, 6),
        },
        "inference_latency": {
            "n_samples_measured":   N_LATENCY,
            "p50_latency_ms":       round(p50_ms, 4),
            "p90_latency_ms":       round(p90_ms, 4),
            "p95_latency_ms":       round(p95_ms, 4),
            "batch_latency_ms":     round(batch_latency_ms, 4),
            "batch_size_used":      BATCH_SIZE,
        },
        "inference_throughput": {
            "samples_per_second":   round(throughput_sps, 1),
            "batch_size":           BATCH_SIZE,
        },
        "memory_footprint": {
            "peak_ram_inference_mb": round(peak_ram_mb, 4),
        },
        "slo_targets": {
            "p95_latency_target_ms":   50,
            "p95_latency_achieved_ms": round(p95_ms, 4),
            "p95_latency_met":         p95_ms <= 50,
            "min_f1_target":           0.70,
            "note": "SLO targets defined for Milestone 4 deployment"
        }
    }

    print(f"    Model size:       {model_size_mb:.4f} MB")
    print(f"    Parameters:       {total_params:,}")
    print(f"    Est. GFLOPs:      {flops_gflops:.6f}")
    print(f"    p50 latency:      {p50_ms:.4f} ms")
    print(f"    p90 latency:      {p90_ms:.4f} ms")
    print(f"    p95 latency:      {p95_ms:.4f} ms  (SLO target: ≤50 ms)")
    print(f"    Throughput:       {throughput_sps:.1f} samples/sec")
    print(f"    Peak RAM:         {peak_ram_mb:.4f} MB")

    return efficiency


# ──────────────────────────────────────────────
# 5. ROC & PR CURVE PLOTS
# ──────────────────────────────────────────────
def plot_roc_curves(y_bin, probs, class_names, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["steelblue", "darkorange", "green", "red", "purple"]
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        auc = roc_auc_score(y_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, color=color, label=f"{cls} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — One-vs-Rest (Test Set)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved: {save_path}")


def plot_pr_curves(y_bin, probs, class_names, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["steelblue", "darkorange", "green", "red", "purple"]
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ap = average_precision_score(y_bin[:, i], probs[:, i])
        ax.plot(rec, prec, color=color, label=f"{cls} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — One-vs-Rest (Test Set)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved: {save_path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    model, config, le, device = load_artifacts()
    X_test, y_test = get_test_set(le, config)

    clf_metrics, preds, probs, y_bin = compute_classification_metrics(
        model, X_test, y_test, le, device, config
    )
    efficiency = compute_efficiency_metrics(model, X_test, config, device)

    # ── Save combined results ──
    print("[5/5] Saving all results...")
    combined = {
        "classification_metrics": clf_metrics,
        "efficiency_metrics": efficiency,
    }
    out_path = os.path.join(RESULTS_DIR, "efficiency_metrics.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"    Saved: {out_path}")

    # ── ROC & PR curves ──
    plot_roc_curves(y_bin, probs, le.classes_,
                    os.path.join(RESULTS_DIR, "roc_curves.png"))
    plot_pr_curves(y_bin, probs, le.classes_,
                   os.path.join(RESULTS_DIR, "pr_curves.png"))

    # ── Final summary ──
    print(f"\n{'='*60}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Accuracy:           {clf_metrics['accuracy']:.4f}")
    print(f"  F1 Macro:           {clf_metrics['f1_macro']:.4f}")
    print(f"  F1 Micro:           {clf_metrics['f1_micro']:.4f}")
    print(f"  AUROC (macro OvR):  {clf_metrics['auroc_macro_ovr']:.4f}")
    print(f"  PR-AUC (macro):     {clf_metrics['pr_auc_macro']:.4f}")
    print(f"  ---")
    print(f"  Model size:         {efficiency['model_size']['on_disk_mb']:.4f} MB")
    print(f"  Parameters:         {efficiency['model_size']['total_parameters']:,}")
    print(f"  p50 latency:        {efficiency['inference_latency']['p50_latency_ms']:.4f} ms")
    print(f"  p90 latency:        {efficiency['inference_latency']['p90_latency_ms']:.4f} ms")
    print(f"  p95 latency:        {efficiency['inference_latency']['p95_latency_ms']:.4f} ms")
    print(f"  Throughput:         {efficiency['inference_throughput']['samples_per_second']:.1f} samples/sec")
    print(f"  Peak RAM:           {efficiency['memory_footprint']['peak_ram_inference_mb']:.4f} MB")
    slo = efficiency["slo_targets"]
    slo_status = "✓ MET" if slo["p95_latency_met"] else "✗ MISSED"
    print(f"  p95 SLO (≤50ms):    {slo_status}")
    print(f"{'='*60}\n")

    print("All outputs saved to results/")
    print("  results/efficiency_metrics.json")
    print("  results/roc_curves.png")
    print("  results/pr_curves.png")


if __name__ == "__main__":
    main()
