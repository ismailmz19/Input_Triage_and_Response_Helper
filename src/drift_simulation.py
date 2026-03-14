"""
drift_simulation.py — Milestone 3: Adaptation & Model Updates
INSE 6450: AI in Systems Engineering — Winter 2026
Student: Ismail Mzouri (40335670)

What this script does:
  1. Simulates three types of data drift:
       - Class prior shift   (rebalance class proportions)
       - Feature mean shift  (shift TF-IDF/metadata feature distributions)
       - Text domain shift   (simulate different vocabulary/style)
  2. Evaluates model BEFORE adaptation on drifted data
  3. Retrains / fine-tunes model on drifted data (incremental head retrain)
  4. Reports before/after metrics: F1, accuracy, latency, model size
  5. Saves versioned model artifacts and configs
  6. Saves all results to results/drift/

Usage:
  python src/drift_simulation.py

Prerequisites:
  Run src/train.py first.

Outputs:
  results/drift/drift_metrics.json
  results/drift/drift_comparison.png
  models/mlp_model_adapted.pth
  models/mlp_model_adapted_config.json
"""

import os
import json
import time
import random
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from scipy.sparse import load_npz, issparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "features.npz")
LABELS_PATH   = os.path.join(BASE_DIR, "data", "processed", "labels.csv")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
RESULTS_DIR   = os.path.join(BASE_DIR, "results", "drift")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Feature layout
N_TFIDF = 10000
N_META  = 14
N_LING  = 5

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
    print("[1/6] Loading model artifacts...")
    with open(os.path.join(MODELS_DIR, "mlp_model_config.json")) as f:
        config = json.load(f)

    le     = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
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
    print(f"    Device: {device} | Classes: {list(le.classes_)}")
    return model, config, le, device


# ──────────────────────────────────────────────
# 2. LOAD & SPLIT DATA
# ──────────────────────────────────────────────
def load_data(le, config):
    print("[2/6] Loading data...")
    X = load_npz(FEATURES_PATH)
    labels_df = pd.read_csv(LABELS_PATH)
    label_col = "label" if "label" in labels_df.columns else labels_df.columns[0]
    y = le.transform(labels_df[label_col].values)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=SEED, stratify=y
    )
    val_relative = config["val_size"] / (1.0 - config["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_relative, random_state=SEED, stratify=y_tv
    )
    print(f"    Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ──────────────────────────────────────────────
# 3. PREDICT HELPER
# ──────────────────────────────────────────────
@torch.no_grad()
def predict(model, X, device, batch_size=512):
    if issparse(X):
        X = X.toarray()
    tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
    all_logits = []
    for (b,) in loader:
        all_logits.append(model(b.to(device)).cpu())
    logits = torch.cat(all_logits, dim=0)
    probs  = torch.softmax(logits, dim=1).numpy()
    preds  = logits.argmax(dim=1).numpy()
    return preds, probs


def measure_latency(model, X, device, n=300):
    if issparse(X):
        X = X[:n].toarray()
    else:
        X = X[:n]
    model.eval()
    latencies = []
    with torch.no_grad():
        for i in range(min(n, len(X))):
            x = torch.tensor(X[i:i+1], dtype=torch.float32).to(device)
            t0 = time.perf_counter()
            _ = model(x)
            latencies.append((time.perf_counter() - t0) * 1000)
    return round(float(np.percentile(latencies, 50)), 4), round(float(np.percentile(latencies, 90)), 4)


# ──────────────────────────────────────────────
# 4. DRIFT SIMULATORS
# ──────────────────────────────────────────────
def simulate_class_prior_shift(X, y, num_classes, shift_config=None):
    """
    Simulate class prior shift by resampling with new class proportions.
    Default: oversample Urgent (rare) and undersample Spam (common).
    shift_config: dict mapping class_idx -> target_fraction
    """
    if shift_config is None:
        # Shift: make Urgent (4) more common, Spam (3) less common
        shift_config = {0: 0.20, 1: 0.25, 2: 0.20, 3: 0.10, 4: 0.25}

    rng = np.random.RandomState(SEED + 1)
    total = len(y)
    indices = []

    for cls_idx, fraction in shift_config.items():
        cls_indices = np.where(y == cls_idx)[0]
        n_target = int(total * fraction)
        if len(cls_indices) == 0:
            continue
        # Sample with replacement if needed
        sampled = rng.choice(cls_indices, size=n_target, replace=(n_target > len(cls_indices)))
        indices.extend(sampled.tolist())

    rng.shuffle(indices)
    indices = np.array(indices)

    if issparse(X):
        return X[indices], y[indices]
    return X[indices], y[indices]


def simulate_feature_mean_shift(X, shift_std=0.5, shift_cols=None, max_samples=8000):
    """
    Simulate covariate shift by adding a systematic offset to metadata features.
    Represents distribution change (e.g., new email client with different metadata).
    """
    if X.shape[0] > max_samples:
        idx = np.random.RandomState(SEED).choice(X.shape[0], max_samples, replace=False)
        X = X[idx]
    if issparse(X):
        X_dense = X.toarray().astype(np.float32)
    else:
        X_dense = X.copy().astype(np.float32)

    rng = np.random.RandomState(SEED + 2)

    # Shift metadata block (cols N_TFIDF to N_TFIDF+N_META)
    shift = rng.normal(0, shift_std, (1, N_META)).astype(np.float32)
    X_shifted = X_dense.copy()
    X_shifted[:, N_TFIDF:N_TFIDF + N_META] += shift

    # Also add mild TF-IDF noise to simulate vocabulary drift
    tfidf_noise = rng.normal(0, 0.05, X_shifted[:, :N_TFIDF].shape).astype(np.float32)
    X_shifted[:, :N_TFIDF] = np.clip(X_shifted[:, :N_TFIDF] + tfidf_noise, 0, None)

    return X_shifted


def simulate_text_domain_shift(X, y=None, domain_bias=None, max_samples=8000):
    """
    Simulate text domain shift by selectively zeroing out certain TF-IDF
    features (simulating a new email domain with different vocabulary)
    and boosting others (new common terms).
    Represents e.g. switching from Enron internal to external customer emails.
    """
    if X.shape[0] > max_samples:
        idx = np.random.RandomState(SEED).choice(X.shape[0], max_samples, replace=False)
        X = X[idx]
    if issparse(X):
        X_dense = X.toarray().astype(np.float32)
    else:
        X_dense = X.copy().astype(np.float32)

    rng = np.random.RandomState(SEED + 3)

    # Zero out 20% of TF-IDF vocabulary (old terms no longer used)
    n_zero = int(N_TFIDF * 0.20)
    zero_cols = rng.choice(N_TFIDF, n_zero, replace=False)
    X_shifted = X_dense.copy()
    X_shifted[:, zero_cols] = 0.0

    # Boost 5% of remaining TF-IDF features (new domain terms more frequent)
    n_boost = int(N_TFIDF * 0.05)
    boost_cols = rng.choice(
        np.setdiff1d(np.arange(N_TFIDF), zero_cols), n_boost, replace=False
    )
    X_shifted[:, boost_cols] *= 1.5

    return X_shifted


# ──────────────────────────────────────────────
# 5. ADAPTATION: RETRAIN CLASSIFICATION HEAD
# ──────────────────────────────────────────────
def retrain_head(model, X_train_drift, y_train_drift, config, device, epochs=10):
    """
    Incremental adaptation: freeze all layers except the final classification head.
    This is the recommended strategy for fast adaptation to distribution shift
    while preserving learned feature representations.
    """
    # Freeze all layers except the last Linear layer
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze only the last layer (classification head)
    last_layer = list(model.network.children())[-1]
    for param in last_layer.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"    Frozen {total - trainable:,} params | Fine-tuning {trainable:,} params (head only)")

    # Prepare data
    if issparse(X_train_drift):
        X_arr = X_train_drift.toarray()
    else:
        X_arr = X_train_drift

    # Subsample for fast adaptation (max 10k samples)
    MAX_ADAPT = 10000
    if len(y_train_drift) > MAX_ADAPT:
        idx = np.random.RandomState(SEED).choice(len(y_train_drift), MAX_ADAPT, replace=False)
        X_arr = X_arr[idx]
        y_arr = y_train_drift[idx]
    else:
        y_arr = y_train_drift

    X_tensor = torch.tensor(X_arr, dtype=torch.float32)
    y_tensor = torch.tensor(y_arr, dtype=torch.long)
    loader   = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=config["batch_size"], shuffle=True
    )

    # Class weights for imbalance
    counts  = np.bincount(y_arr, minlength=config["num_classes"]).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * config["num_classes"]
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(device)
    )
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

    model.train()
    history = []
    for epoch in range(1, epochs + 1):
        total_loss, all_preds, all_labels = 0.0, [], []
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_b)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y_b.cpu().numpy())
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        history.append({"epoch": epoch, "loss": round(total_loss / len(y_arr), 4), "f1": round(float(f1), 4)})
        if epoch % 2 == 0 or epoch == 1:
            print(f"      Epoch {epoch}/{epochs} | Loss={total_loss/len(y_arr):.4f} | F1={f1:.4f}")

    # Unfreeze all for inference
    for param in model.parameters():
        param.requires_grad = True

    model.eval()
    return model, history


# ──────────────────────────────────────────────
# 6. EVALUATE DRIFT SCENARIO
# ──────────────────────────────────────────────
def evaluate_scenario(model, X_drifted, y_drifted, le, device, label):
    preds, probs = predict(model, X_drifted, device)
    f1   = f1_score(y_drifted, preds, average="macro", zero_division=0)
    acc  = accuracy_score(y_drifted, preds)
    p50, p90 = measure_latency(model, X_drifted, device)
    print(f"    [{label}] F1={f1:.4f} | Acc={acc:.4f} | p50={p50:.2f}ms | p90={p90:.2f}ms")
    return {
        "f1_macro":       round(float(f1), 4),
        "accuracy":       round(float(acc), 4),
        "latency_p50_ms": p50,
        "latency_p90_ms": p90,
    }


# ──────────────────────────────────────────────
# 7. PLOTS
# ──────────────────────────────────────────────
def plot_drift_comparison(all_results, save_path):
    """
    Side-by-side bar chart: before vs after adaptation for each drift type.
    """
    drift_types = list(all_results.keys())
    n = len(drift_types)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── F1 comparison ─────────────────────────
    ax = axes[0]
    x  = np.arange(n)
    w  = 0.25
    clean_f1  = [all_results[d]["clean"]["f1_macro"]   for d in drift_types]
    before_f1 = [all_results[d]["before"]["f1_macro"]  for d in drift_types]
    after_f1  = [all_results[d]["after"]["f1_macro"]   for d in drift_types]

    b1 = ax.bar(x - w, clean_f1,  w, label="Clean baseline", color="steelblue")
    b2 = ax.bar(x,     before_f1, w, label="After drift (before adapt)", color="darkorange")
    b3 = ax.bar(x + w, after_f1,  w, label="After adaptation", color="green")

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in drift_types], fontsize=9)
    ax.set_ylabel("Macro F1")
    ax.set_title("F1: Baseline vs Drift vs Adapted")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    # ── Latency comparison ────────────────────
    ax = axes[1]
    before_lat = [all_results[d]["before"]["latency_p50_ms"] for d in drift_types]
    after_lat  = [all_results[d]["after"]["latency_p50_ms"]  for d in drift_types]
    clean_lat  = [all_results[d]["clean"]["latency_p50_ms"]  for d in drift_types]

    ax.bar(x - w, clean_lat,  w, label="Clean baseline", color="steelblue")
    ax.bar(x,     before_lat, w, label="After drift",    color="darkorange")
    ax.bar(x + w, after_lat,  w, label="After adaptation", color="green")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in drift_types], fontsize=9)
    ax.set_ylabel("p50 Latency (ms)")
    ax.set_title("Latency: Baseline vs Drift vs Adapted")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Drift Simulation: Before vs After Model Adaptation", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


def plot_adaptation_history(histories, save_path):
    """Plot fine-tuning loss and F1 curves per drift scenario."""
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    colors = ["steelblue", "darkorange", "green"]
    for ax, (name, hist), color in zip(axes, histories.items(), colors):
        epochs = [h["epoch"] for h in hist]
        f1s    = [h["f1"] for h in hist]
        losses = [h["loss"] for h in hist]
        ax.plot(epochs, f1s,    marker="o", color=color,   label="Train F1", linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(epochs, losses, marker="s", color="gray", label="Loss", linestyle="--")
        ax.set_title(f"Adaptation: {name.replace('_', ' ')}", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro F1", color=color)
        ax2.set_ylabel("Loss", color="gray")
        ax.grid(True, alpha=0.3)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.suptitle("Head Fine-Tuning Curves per Drift Scenario", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Milestone 3 — Drift Simulation & Model Adaptation")
    print("=" * 60)

    model, config, le, device = load_artifacts()
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(le, config)

    # ── Clean baseline ────────────────────────
    print("\n[Baseline] Evaluating on clean test set...")
    clean_metrics = evaluate_scenario(model, X_test, y_test, le, device, "clean")

    # ── Model size baseline ───────────────────
    model_path    = os.path.join(MODELS_DIR, "mlp_model.pth")
    model_size_mb = os.path.getsize(model_path) / (1024 ** 2)

    # ── Define drift scenarios ─────────────────
    scenarios = {
        "class_prior_shift":   None,
        "feature_mean_shift":  None,
        "text_domain_shift":   None,
    }

    all_results  = {}
    all_histories = {}

    print("\n[3/6] Running drift scenarios...")

    # ────────────────────────────────────────
    # SCENARIO 1: Class Prior Shift
    # ────────────────────────────────────────
    print("\n  Scenario 1: Class Prior Shift")
    print("    Simulating: Urgent emails become more frequent (25% vs original ~5%)")

    X_drift1, y_drift1 = simulate_class_prior_shift(X_test, y_test, config["num_classes"])
    X_train1, y_train1 = simulate_class_prior_shift(X_train, y_train, config["num_classes"])

    print("    Before adaptation:")
    before1 = evaluate_scenario(model, X_drift1, y_drift1, le, device, "before")

    # Deep copy model for this scenario
    import copy
    model1 = copy.deepcopy(model)
    print("    Adapting model (head fine-tune, 10 epochs)...")
    model1, hist1 = retrain_head(model1, X_train1, y_train1, config, device, epochs=10)

    print("    After adaptation:")
    after1 = evaluate_scenario(model1, X_drift1, y_drift1, le, device, "after")

    all_results["class_prior_shift"]   = {"clean": clean_metrics, "before": before1, "after": after1}
    all_histories["class_prior_shift"] = hist1

    # ────────────────────────────────────────
    # SCENARIO 2: Feature Mean Shift
    # ────────────────────────────────────────
    print("\n  Scenario 2: Feature Mean Shift")
    print("    Simulating: Metadata distribution shift (new email client/format)")

    MAX_S = 8000
    idx2t  = np.random.RandomState(SEED).choice(len(y_test),  min(MAX_S, len(y_test)),  replace=False)
    idx2tr = np.random.RandomState(SEED).choice(len(y_train), min(MAX_S, len(y_train)), replace=False)
    X_drift2 = simulate_feature_mean_shift(X_test[idx2t],  shift_std=0.5, max_samples=MAX_S)
    X_train2 = simulate_feature_mean_shift(X_train[idx2tr], shift_std=0.5, max_samples=MAX_S)
    y_drift2 = y_test[idx2t][:len(X_drift2)]
    y_train2 = y_train[idx2tr][:len(X_train2)]

    print("    Before adaptation:")
    before2 = evaluate_scenario(model, X_drift2, y_drift2, le, device, "before")

    model2 = copy.deepcopy(model)
    print("    Adapting model (head fine-tune, 10 epochs)...")
    model2, hist2 = retrain_head(model2, X_train2, y_train2, config, device, epochs=10)

    print("    After adaptation:")
    after2 = evaluate_scenario(model2, X_drift2, y_drift2, le, device, "after")

    all_results["feature_mean_shift"]   = {"clean": clean_metrics, "before": before2, "after": after2}
    all_histories["feature_mean_shift"] = hist2

    # ────────────────────────────────────────
    # SCENARIO 3: Text Domain Shift
    # ────────────────────────────────────────
    print("\n  Scenario 3: Text Domain Shift")
    print("    Simulating: 20% vocabulary dropout + 5% new term boost (new email domain)")

    idx3t  = np.random.RandomState(SEED).choice(len(y_test),  min(MAX_S, len(y_test)),  replace=False)
    idx3tr = np.random.RandomState(SEED).choice(len(y_train), min(MAX_S, len(y_train)), replace=False)
    X_drift3 = simulate_text_domain_shift(X_test[idx3t],  max_samples=MAX_S)
    X_train3 = simulate_text_domain_shift(X_train[idx3tr], max_samples=MAX_S)
    y_drift3 = y_test[idx3t][:len(X_drift3)]
    y_train3 = y_train[idx3tr][:len(X_train3)]

    print("    Before adaptation:")
    before3 = evaluate_scenario(model, X_drift3, y_drift3, le, device, "before")

    model3 = copy.deepcopy(model)
    print("    Adapting model (head fine-tune, 10 epochs)...")
    model3, hist3 = retrain_head(model3, X_train3, y_train3, config, device, epochs=10)

    print("    After adaptation:")
    after3 = evaluate_scenario(model3, X_drift3, y_drift3, le, device, "after")

    all_results["text_domain_shift"]   = {"clean": clean_metrics, "before": before3, "after": after3}
    all_histories["text_domain_shift"] = hist3

    # ── Save best adapted model (text domain) ─
    print("\n[4/6] Saving adapted model artifacts...")
    adapted_path = os.path.join(MODELS_DIR, "mlp_model_adapted.pth")
    torch.save(model3.state_dict(), adapted_path)
    adapted_size_mb = os.path.getsize(adapted_path) / (1024 ** 2)

    adapted_config = dict(config)
    adapted_config["version"]          = "adapted_v1"
    adapted_config["adaptation_type"]  = "head_finetune"
    adapted_config["drift_scenario"]   = "text_domain_shift"
    adapted_config["adaptation_epochs"] = 10
    adapted_config["base_model"]       = "mlp_model.pth"
    adapted_config["model_size_mb"]    = round(adapted_size_mb, 4)

    with open(os.path.join(MODELS_DIR, "mlp_model_adapted_config.json"), "w") as f:
        json.dump(adapted_config, f, indent=2)
    print(f"    Saved: {adapted_path}")
    print(f"    Model size: original={model_size_mb:.4f}MB | adapted={adapted_size_mb:.4f}MB")

    # ── Plots ─────────────────────────────────
    print("\n[5/6] Generating plots...")
    plot_drift_comparison(
        all_results,
        os.path.join(RESULTS_DIR, "drift_comparison.png")
    )
    plot_adaptation_history(
        all_histories,
        os.path.join(RESULTS_DIR, "adaptation_history.png")
    )

    # ── Save JSON ─────────────────────────────
    print("[6/6] Saving results...")
    output = {
        "drift_scenarios": all_results,
        "adaptation_strategy": {
            "method":         "Incremental head fine-tuning",
            "description":    "Freeze all hidden layers; retrain only the final classification layer on drifted data",
            "rationale":      "Preserves learned feature representations while adapting decision boundaries to new distribution",
            "epochs":         10,
            "optimizer":      "Adam (lr=1e-3)",
            "max_train_samples": 10000,
        },
        "versioning": {
            "original_model":  "models/mlp_model.pth",
            "original_config": "models/mlp_model_config.json",
            "adapted_model":   "models/mlp_model_adapted.pth",
            "adapted_config":  "models/mlp_model_adapted_config.json",
            "label_encoder":   "models/label_encoder.joblib",
            "feature_pipeline": "data/processed/feature_pipeline.joblib",
        },
        "update_triggers": {
            "drift_threshold":   "PSI > 0.2 on any of top-10 TF-IDF features for >= 3 consecutive windows",
            "kpi_drop":          "Rolling macro F1 drops > 0.05 below baseline for >= 2 monitoring windows",
            "operational":       "New email domain onboarded, seasonal campaign detected, organizational restructure",
            "cadence":           "Monthly periodic check + event-driven triggers",
            "label_acquisition": "Weak supervision rules updated by domain expert; spot-check 200 emails per cycle",
        },
        "retraining_plans": {
            "mild_drift":    "Head fine-tuning (current approach) — fast, low cost, preserves representations",
            "moderate_drift": "Full retrain with mixed old + new data (80/20 ratio) — preserves stability",
            "severe_drift":  "Full retrain on new data only + re-run feature_extraction.py to rebuild TF-IDF vocab",
        },
        "model_size": {
            "original_mb":  round(model_size_mb, 4),
            "adapted_mb":   round(adapted_size_mb, 4),
            "size_change":  "No change — head fine-tuning does not alter model architecture or size",
        },
        "deployment_changes": {
            "changes_needed": [
                "Add confidence threshold abstention (threshold=0.7) to predict.py based on stress test findings",
                "Add pre-flight schema check before feature extraction in production pipeline",
                "Version model artifacts with timestamp to enable rollback",
                "Add TF-IDF vocabulary refresh trigger when OOV rate exceeds 15%",
            ],
            "no_changes_needed": [
                "Model architecture remains optimal — MLP is lightweight and CPU-deployable",
                "Feature pipeline structure is stable — no new feature families needed",
                "Response template system is drift-agnostic — keyword matching is robust",
            ],
        },
    }

    metrics_path = os.path.join(RESULTS_DIR, "drift_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"    Saved: {metrics_path}")

    # ── Summary ───────────────────────────────
    print(f"\n{'='*60}")
    print("  DRIFT SIMULATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Clean baseline F1: {clean_metrics['f1_macro']:.4f}")
    print(f"  {'Scenario':<25} {'Before':>8} {'After':>8} {'Recovery':>10}")
    print(f"  {'-'*55}")
    for name, res in all_results.items():
        before = res["before"]["f1_macro"]
        after  = res["after"]["f1_macro"]
        recovery = after - before
        print(f"  {name:<25} {before:>8.4f} {after:>8.4f} {recovery:>+10.4f}")
    print(f"\n  Model size: {model_size_mb:.4f}MB → {adapted_size_mb:.4f}MB (no change)")
    print(f"{'='*60}")
    print("\nOutputs saved to results/drift/")
    print("  drift_metrics.json")
    print("  drift_comparison.png")
    print("  adaptation_history.png")
    print("  models/mlp_model_adapted.pth")
    print("  models/mlp_model_adapted_config.json")


if __name__ == "__main__":
    main()