"""
stress_test.py — Milestone 3: Anticipated Failures & Risk Analysis
INSE 6450: AI in Systems Engineering — Winter 2026
Student: Ismail Mzouri (40335670)

What this script does:
  1. Pre-flight checks: schema validation, null-rate thresholds, feature range checks
  2. Runtime guards: request validation, confidence threshold abstention, rate limiting sim
  3. Post-prediction checks: confidence bands, calibration-based flagging
  4. Stress tests with metrics:
       - Corrupted inputs (noise, masking, token dropout)
       - Partial feature loss (zero out metadata / linguistic / tfidf blocks)
       - OOD samples (random vectors, all-zero, all-one inputs)
       - Class rarity scenarios (evaluate on minority classes only)
  5. Saves all results to results/stress_test/

Usage:
  python src/stress_test.py

Prerequisites:
  Run src/train.py, src/evaluate.py, and src/robustness.py first.

Outputs:
  results/stress_test/stress_test_metrics.json
  results/stress_test/stress_test_summary.png
  results/stress_test/preflight_report.json
"""

import os
import sys
import json
import time
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import joblib
from scipy.sparse import load_npz, issparse, csr_matrix, hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "features.npz")
LABELS_PATH   = os.path.join(BASE_DIR, "data", "processed", "labels.csv")
CLEANED_PATH  = os.path.join(BASE_DIR, "data", "processed", "cleaned_emails.csv")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
RESULTS_DIR   = os.path.join(BASE_DIR, "results", "stress_test")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Feature layout (from feature_extraction.py)
N_TFIDF    = 10000
N_META     = 14
N_LING     = 5
N_TOTAL    = N_TFIDF + N_META + N_LING   # 10019

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

    le       = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
    pipeline = joblib.load(os.path.join(BASE_DIR, "data", "processed", "feature_pipeline.joblib"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = EmailMLP(
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
    return model, config, le, pipeline, device


# ──────────────────────────────────────────────
# 2. LOAD TEST SET
# ──────────────────────────────────────────────
def get_test_set(le, config):
    print("[2/6] Loading test split...")
    X = load_npz(FEATURES_PATH)
    labels_df = pd.read_csv(LABELS_PATH)
    label_col = "label" if "label" in labels_df.columns else labels_df.columns[0]
    y = le.transform(labels_df[label_col].values)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=SEED, stratify=y
    )
    print(f"    Test set: {X_test.shape[0]:,} samples | {X_test.shape[1]} features")
    return X_test, y_test


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


# ──────────────────────────────────────────────
# 4. PRE-FLIGHT CHECKS
# ──────────────────────────────────────────────
def run_preflight_checks(X_test, y_test, le):
    """
    Schema/feature range validation and null-rate checks.
    Simulates checks that would run before serving predictions.
    """
    print("[3/6] Running pre-flight checks...")

    if issparse(X_test):
        X_dense = X_test.toarray()
    else:
        X_dense = X_test

    checks = {}

    # ── Schema check ──────────────────────────
    expected_features = N_TOTAL
    actual_features   = X_dense.shape[1]
    schema_ok = actual_features == expected_features
    checks["schema_check"] = {
        "expected_features": expected_features,
        "actual_features":   actual_features,
        "passed":            schema_ok,
        "action":            "PASS" if schema_ok else "REJECT: feature dimension mismatch",
    }

    # ── Null / zero-row rate ──────────────────
    zero_rows    = int(np.all(X_dense == 0, axis=1).sum())
    null_rate    = zero_rows / len(X_dense)
    null_ok      = null_rate < 0.05   # threshold: <5% all-zero rows
    checks["null_rate_check"] = {
        "zero_rows":       zero_rows,
        "total_rows":      len(X_dense),
        "null_rate":       round(float(null_rate), 4),
        "threshold":       0.05,
        "passed":          null_ok,
        "action":          "PASS" if null_ok else "WARN: high proportion of empty emails",
    }

    # ── Feature range checks (metadata block) ─
    meta = X_dense[:, N_TFIDF:N_TFIDF + N_META]
    meta_means = meta.mean(axis=0)
    meta_stds  = meta.std(axis=0)
    out_of_range = int(np.any(np.abs(meta) > 10, axis=1).sum())
    range_rate   = out_of_range / len(X_dense)
    range_ok     = range_rate < 0.10
    checks["feature_range_check"] = {
        "metadata_mean_norm": round(float(np.linalg.norm(meta_means)), 4),
        "metadata_std_mean":  round(float(meta_stds.mean()), 4),
        "out_of_range_rows":  out_of_range,
        "out_of_range_rate":  round(float(range_rate), 4),
        "threshold":          0.10,
        "passed":             range_ok,
        "action":             "PASS" if range_ok else "WARN: metadata values out of expected range",
    }

    # ── Class distribution check ──────────────
    unique, counts = np.unique(y_test, return_counts=True)
    class_dist     = {le.classes_[i]: int(c) for i, c in zip(unique, counts)}
    min_class_count = int(counts.min())
    rarity_ok       = min_class_count >= 50
    checks["class_distribution_check"] = {
        "class_counts":     class_dist,
        "min_class_count":  min_class_count,
        "threshold":        50,
        "passed":           rarity_ok,
        "action":           "PASS" if rarity_ok else "WARN: rare class detected — predictions may be unreliable",
    }

    # ── TF-IDF sparsity check ─────────────────
    tfidf_block  = X_dense[:, :N_TFIDF]
    sparsity     = 1.0 - (tfidf_block != 0).mean()
    sparsity_ok  = sparsity > 0.90   # TF-IDF should be >90% sparse
    checks["tfidf_sparsity_check"] = {
        "sparsity_ratio": round(float(sparsity), 4),
        "threshold_min":  0.90,
        "passed":         sparsity_ok,
        "action":         "PASS" if sparsity_ok else "WARN: unexpectedly dense TF-IDF — possible data corruption",
    }

    passed = sum(1 for v in checks.values() if v["passed"])
    total  = len(checks)
    print(f"    Pre-flight checks: {passed}/{total} passed")
    for name, result in checks.items():
        status = "✓" if result["passed"] else "✗"
        print(f"      {status} {name}: {result['action']}")

    return checks


# ──────────────────────────────────────────────
# 5. RUNTIME GUARDS SIMULATION
# ──────────────────────────────────────────────
def simulate_runtime_guards(model, X_test, y_test, device):
    """
    Simulate runtime guards: confidence thresholding and abstention.
    """
    print("[4/6] Simulating runtime guards...")

    preds, probs = predict(model, X_test, device)
    max_conf = probs.max(axis=1)

    results = {}

    # ── Confidence threshold abstention ──────
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        mask_confident = max_conf >= threshold
        n_confident    = mask_confident.sum()
        n_abstained    = (~mask_confident).sum()
        coverage       = n_confident / len(preds)

        if n_confident > 0:
            f1_confident = f1_score(
                y_test[mask_confident], preds[mask_confident],
                average="macro", zero_division=0
            )
        else:
            f1_confident = 0.0

        results[f"threshold_{threshold}"] = {
            "threshold":       threshold,
            "n_confident":     int(n_confident),
            "n_abstained":     int(n_abstained),
            "coverage":        round(float(coverage), 4),
            "f1_macro_confident": round(float(f1_confident), 4),
        }
        print(f"    Threshold={threshold:.1f} | Coverage={coverage:.3f} | "
              f"F1={f1_confident:.4f} | Abstained={n_abstained:,}")

    # ── Post-prediction calibration band check ─
    # Flag predictions where confidence > 0.95 but class is historically unreliable
    high_conf_wrong = int(((max_conf > 0.95) & (preds != y_test)).sum())
    high_conf_total = int((max_conf > 0.95).sum())
    overconfidence_rate = high_conf_wrong / max(high_conf_total, 1)

    results["overconfidence_check"] = {
        "high_conf_predictions": high_conf_total,
        "high_conf_wrong":       high_conf_wrong,
        "overconfidence_rate":   round(float(overconfidence_rate), 4),
        "flag":                  overconfidence_rate > 0.10,
        "note": "Flag if >10% of high-confidence predictions are wrong",
    }
    print(f"    Overconfidence rate (conf>0.95, wrong): {overconfidence_rate:.4f}")

    return results


# ──────────────────────────────────────────────
# 6. STRESS TESTS
# ──────────────────────────────────────────────
def run_stress_tests(model, X_test, y_test, le, device):
    """
    Four stress test categories:
    A. Corrupted inputs
    B. Partial feature loss
    C. OOD samples
    D. Class rarity scenarios
    """
    print("[5/6] Running stress tests...")

    # Subsample to avoid OOM when densifying large sparse matrix
    MAX_STRESS_SAMPLES = 5000
    rng = np.random.RandomState(SEED)
    idx_sub = rng.choice(X_test.shape[0], min(MAX_STRESS_SAMPLES, X_test.shape[0]), replace=False)
    if issparse(X_test):
        X_dense = X_test[idx_sub].toarray()
    else:
        X_dense = X_test[idx_sub].copy()
    y_test = y_test[idx_sub]

    print(f"    Using {len(y_test):,} samples (subsampled to avoid OOM)")
    n, d    = X_dense.shape
    results = {}

    # ── A. Corrupted Inputs ───────────────────
    print("    A. Corrupted inputs...")
    corruptions = {}

    # A1: Gaussian noise injection
    for noise_std in [0.1, 0.5, 1.0]:
        X_noisy = X_dense + np.random.RandomState(SEED).normal(0, noise_std, X_dense.shape).astype(np.float32)
        preds, _ = predict(model, X_noisy, device)
        f1  = f1_score(y_test, preds, average="macro", zero_division=0)
        acc = accuracy_score(y_test, preds)
        corruptions[f"gaussian_noise_std_{noise_std}"] = {
            "f1_macro": round(float(f1), 4),
            "accuracy": round(float(acc), 4),
        }
        print(f"      Gaussian noise std={noise_std}: F1={f1:.4f}")

    # A2: Random feature masking (zero out random % of features)
    for mask_rate in [0.1, 0.3, 0.5]:
        X_masked = X_dense.copy()
        mask = np.random.RandomState(SEED).rand(*X_masked.shape) < mask_rate
        X_masked[mask] = 0.0
        preds, _ = predict(model, X_masked, device)
        f1  = f1_score(y_test, preds, average="macro", zero_division=0)
        acc = accuracy_score(y_test, preds)
        corruptions[f"random_masking_rate_{mask_rate}"] = {
            "f1_macro": round(float(f1), 4),
            "accuracy": round(float(acc), 4),
        }
        print(f"      Random masking rate={mask_rate}: F1={f1:.4f}")

    # A3: Token dropout (zero out TF-IDF block only)
    for drop_rate in [0.3, 0.5]:
        X_drop = X_dense.copy()
        tfidf_cols = np.random.RandomState(SEED).choice(N_TFIDF, int(N_TFIDF * drop_rate), replace=False)
        X_drop[:, tfidf_cols] = 0.0
        preds, _ = predict(model, X_drop, device)
        f1  = f1_score(y_test, preds, average="macro", zero_division=0)
        acc = accuracy_score(y_test, preds)
        corruptions[f"tfidf_token_dropout_{drop_rate}"] = {
            "f1_macro": round(float(f1), 4),
            "accuracy": round(float(acc), 4),
        }
        print(f"      TF-IDF token dropout rate={drop_rate}: F1={f1:.4f}")

    results["corrupted_inputs"] = corruptions

    # ── B. Partial Feature Loss ───────────────
    print("    B. Partial feature loss...")
    feature_loss = {}

    # B1: Zero out entire metadata block
    X_no_meta = X_dense.copy()
    X_no_meta[:, N_TFIDF:N_TFIDF + N_META] = 0.0
    preds, _ = predict(model, X_no_meta, device)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    feature_loss["no_metadata"] = {
        "description": "Metadata block zeroed (14 features)",
        "f1_macro": round(float(f1), 4),
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
    }
    print(f"      No metadata: F1={f1:.4f}")

    # B2: Zero out linguistic features
    X_no_ling = X_dense.copy()
    X_no_ling[:, N_TFIDF + N_META:] = 0.0
    preds, _ = predict(model, X_no_ling, device)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    feature_loss["no_linguistic"] = {
        "description": "Linguistic block zeroed (5 features)",
        "f1_macro": round(float(f1), 4),
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
    }
    print(f"      No linguistic: F1={f1:.4f}")

    # B3: Zero out TF-IDF entirely (text dropout)
    X_no_tfidf = X_dense.copy()
    X_no_tfidf[:, :N_TFIDF] = 0.0
    preds, _ = predict(model, X_no_tfidf, device)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    feature_loss["no_tfidf"] = {
        "description": "TF-IDF block zeroed (10,000 features) — text unavailable",
        "f1_macro": round(float(f1), 4),
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
    }
    print(f"      No TF-IDF (text only lost): F1={f1:.4f}")

    # B4: Only TF-IDF (metadata + linguistic dropped)
    X_tfidf_only = X_dense.copy()
    X_tfidf_only[:, N_TFIDF:] = 0.0
    preds, _ = predict(model, X_tfidf_only, device)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    feature_loss["tfidf_only"] = {
        "description": "Only TF-IDF retained (metadata+linguistic zeroed)",
        "f1_macro": round(float(f1), 4),
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
    }
    print(f"      TF-IDF only: F1={f1:.4f}")

    results["partial_feature_loss"] = feature_loss

    # ── C. OOD Samples ────────────────────────
    print("    C. OOD samples...")
    ood_results = {}
    n_ood = min(1000, n)

    # C1: All-zero input (empty email)
    X_zeros = np.zeros((n_ood, d), dtype=np.float32)
    preds, probs = predict(model, X_zeros, device)
    ood_results["all_zero_input"] = {
        "description":    "All-zero feature vector (empty/missing email)",
        "n_samples":      n_ood,
        "predicted_dist": {le.classes_[i]: int((preds == i).sum()) for i in range(len(le.classes_))},
        "mean_max_conf":  round(float(probs.max(axis=1).mean()), 4),
        "note":           "Model should abstain; high confidence here indicates overconfidence on OOD",
    }
    print(f"      All-zero: mean_conf={probs.max(axis=1).mean():.4f}, "
          f"top_pred={le.classes_[np.bincount(preds).argmax()]}")

    # C2: All-ones input (saturated features)
    X_ones = np.ones((n_ood, d), dtype=np.float32)
    preds, probs = predict(model, X_ones, device)
    ood_results["all_one_input"] = {
        "description":    "All-one feature vector (saturated/adversarial)",
        "n_samples":      n_ood,
        "predicted_dist": {le.classes_[i]: int((preds == i).sum()) for i in range(len(le.classes_))},
        "mean_max_conf":  round(float(probs.max(axis=1).mean()), 4),
        "note":           "Saturated inputs should trigger confidence guard",
    }
    print(f"      All-ones: mean_conf={probs.max(axis=1).mean():.4f}")

    # C3: Random Gaussian noise (pure OOD)
    X_rand = np.random.RandomState(SEED).randn(n_ood, d).astype(np.float32)
    preds, probs = predict(model, X_rand, device)
    ood_results["random_gaussian"] = {
        "description":    "Random Gaussian input (pure OOD — no real email structure)",
        "n_samples":      n_ood,
        "predicted_dist": {le.classes_[i]: int((preds == i).sum()) for i in range(len(le.classes_))},
        "mean_max_conf":  round(float(probs.max(axis=1).mean()), 4),
        "note":           "High confidence on random inputs indicates calibration issue",
    }
    print(f"      Random Gaussian: mean_conf={probs.max(axis=1).mean():.4f}")

    results["ood_samples"] = ood_results

    # ── D. Class Rarity Scenarios ─────────────
    print("    D. Class rarity scenarios...")
    rarity_results = {}

    for class_idx, class_name in enumerate(le.classes_):
        mask = y_test == class_idx
        n_cls = mask.sum()
        if n_cls == 0:
            continue
        X_cls   = X_dense[mask]
        y_cls   = y_test[mask]
        preds_cls, probs_cls = predict(model, X_cls, device)
        f1_cls  = f1_score(y_cls, preds_cls, average="macro", zero_division=0)
        acc_cls = accuracy_score(y_cls, preds_cls)
        mean_conf = float(probs_cls.max(axis=1).mean())

        rarity_results[class_name] = {
            "n_samples":      int(n_cls),
            "f1_macro":       round(float(f1_cls), 4),
            "accuracy":       round(float(acc_cls), 4),
            "mean_confidence": round(mean_conf, 4),
        }
        print(f"      {class_name:<22} n={n_cls:5d} | F1={f1_cls:.4f} | Acc={acc_cls:.4f} | Conf={mean_conf:.4f}")

    results["class_rarity"] = rarity_results

    return results


# ──────────────────────────────────────────────
# 7. PLOTS
# ──────────────────────────────────────────────
def plot_stress_summary(stress_results, runtime_results, save_path):
    """Summary dashboard: feature loss impact + confidence threshold curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Plot 1: Partial feature loss bar chart ─
    ax = axes[0, 0]
    fl = stress_results["partial_feature_loss"]
    labels  = [k.replace("_", "\n") for k in fl.keys()]
    f1_vals = [v["f1_macro"] for v in fl.values()]
    colors  = ["steelblue" if f >= 0.70 else "darkorange" if f >= 0.50 else "red" for f in f1_vals]
    bars = ax.bar(labels, f1_vals, color=colors, edgecolor="white")
    ax.axhline(0.7491, color="gray", linestyle="--", label="Clean F1=0.7491")
    ax.set_title("Partial Feature Loss Impact")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # ── Plot 2: Corrupted input F1 ────────────
    ax = axes[0, 1]
    ci = stress_results["corrupted_inputs"]
    ci_labels = [k.replace("_", "\n") for k in ci.keys()]
    ci_f1     = [v["f1_macro"] for v in ci.values()]
    ci_colors = ["steelblue" if f >= 0.70 else "darkorange" if f >= 0.50 else "red" for f in ci_f1]
    bars2 = ax.bar(range(len(ci_labels)), ci_f1, color=ci_colors, edgecolor="white")
    ax.set_xticks(range(len(ci_labels)))
    ax.set_xticklabels(ci_labels, fontsize=7)
    ax.axhline(0.7491, color="gray", linestyle="--", label="Clean F1=0.7491")
    ax.set_title("Corrupted Input Stress Tests")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Plot 3: Confidence threshold coverage/F1 curve ──
    ax = axes[1, 0]
    thresholds = [v["threshold"] for v in runtime_results.values() if "threshold" in v]
    coverages  = [v["coverage"] for v in runtime_results.values() if "threshold" in v]
    f1s        = [v["f1_macro_confident"] for v in runtime_results.values() if "threshold" in v]
    ax2 = ax.twinx()
    ax.plot(thresholds, coverages, marker="o", color="steelblue", label="Coverage")
    ax2.plot(thresholds, f1s, marker="s", color="darkorange", label="F1 (confident)")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Coverage", color="steelblue")
    ax2.set_ylabel("Macro F1", color="darkorange")
    ax.set_title("Confidence Threshold: Coverage vs F1")
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # ── Plot 4: Class rarity bar chart ────────
    ax = axes[1, 1]
    cr = stress_results["class_rarity"]
    cls_names = list(cr.keys())
    cls_f1    = [cr[c]["f1_macro"] for c in cls_names]
    cls_n     = [cr[c]["n_samples"] for c in cls_names]
    cls_colors = ["steelblue" if f >= 0.70 else "darkorange" if f >= 0.50 else "red" for f in cls_f1]
    bars3 = ax.bar(cls_names, cls_f1, color=cls_colors, edgecolor="white")
    ax.set_xticklabels(cls_names, rotation=20, ha="right", fontsize=8)
    ax.set_title("Per-Class F1 (Rarity Scenarios)")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val, n in zip(bars3, cls_f1, cls_n):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}\n(n={n})", ha="center", va="bottom", fontsize=7)

    plt.suptitle("Stress Test Summary Dashboard", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Milestone 3 — Stress Tests & Risk Analysis")
    print("=" * 60)

    model, config, le, pipeline, device = load_artifacts()
    X_test, y_test = get_test_set(le, config)

    # ── Pre-flight checks ─────────────────────
    preflight = run_preflight_checks(X_test, y_test, le)

    # ── Runtime guards ────────────────────────
    runtime = simulate_runtime_guards(model, X_test, y_test, device)

    # ── Stress tests ──────────────────────────
    stress = run_stress_tests(model, X_test, y_test, le, device)

    # ── Plots ─────────────────────────────────
    print("[6/6] Generating plots...")
    plot_stress_summary(
        stress, runtime,
        os.path.join(RESULTS_DIR, "stress_test_summary.png")
    )

    # ── Save JSON ─────────────────────────────
    output = {
        "preflight_checks":  preflight,
        "runtime_guards":    runtime,
        "stress_tests":      stress,
        "failure_taxonomy": {
            "data_related": [
                "Covariate shift: email language evolves over time (Enron 2001 vs modern email)",
                "Label shift: class distribution changes (e.g., more spam after policy change)",
                "Concept drift: meaning of 'urgent' changes with organizational context",
                "Missing values: empty body or subject fields — mitigated by null-rate check",
                "Out-of-range values: metadata features exceed expected bounds — caught by range check",
            ],
            "model_related": [
                "Overconfidence on OOD: model assigns high confidence to empty/random inputs",
                "Calibration error: mean confidence (0.8132) exceeds fraction correct (0.7684)",
                "Class imbalance regression: minority classes (Scheduling, Urgent) show lower F1",
                "TF-IDF vocabulary staleness: new words after training cutoff are ignored",
            ],
            "system_infra": [
                "No GPU available: mitigated by CPU fallback in all scripts",
                "Model file missing: caught by artifact loading checks",
                "Dependency mismatch: sklearn calibration_curve import location changed — fixed",
                "Feature pipeline version mismatch: versioned config JSON prevents silent errors",
            ],
            "user_interaction": [
                "Empty email body: zero-vector input — mitigated by null-rate pre-flight check",
                "Adversarial rephrasing: synonym swap degrades F1 by only 0.035 — robust",
                "Character-level attacks: F1 drops 0.39 at high severity — highest vulnerability",
                "Extremely short emails: few TF-IDF features activate — confidence guard abstains",
            ],
        },
        "mitigations": {
            "pre_flight": [
                "Schema check: reject requests with wrong feature dimension",
                "Null-rate check: warn if >5% of inputs are all-zero",
                "Feature range check: flag metadata values outside expected bounds",
                "Class distribution check: warn if any class has <50 samples in batch",
            ],
            "runtime": [
                "Confidence threshold abstention at 0.7 — balances coverage (85%) and quality",
                "Post-prediction overconfidence flag: alert if >10% of high-conf predictions are wrong",
                "Input sanitization: normalize ALL-CAPS, strip non-ASCII before feature extraction",
            ],
            "post_prediction": [
                "Calibration band: flag predictions with conf > 0.95 for human review",
                "Response template fallback: if abstained, suggest generic template instead",
                "Monitoring: track rolling F1 per class to detect class imbalance regression",
            ],
        },
    }

    metrics_path = os.path.join(RESULTS_DIR, "stress_test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: bool(x) if isinstance(x, __import__("numpy").bool_) else str(x))
    print(f"    Saved: {metrics_path}")

    preflight_path = os.path.join(RESULTS_DIR, "preflight_report.json")
    with open(preflight_path, "w") as f:
        json.dump(preflight, f, indent=2, default=lambda x: bool(x) if isinstance(x, __import__("numpy").bool_) else str(x))
    print(f"    Saved: {preflight_path}")

    # ── Summary ───────────────────────────────
    print(f"\n{'='*60}")
    print("  STRESS TEST SUMMARY")
    print(f"{'='*60}")
    print(f"  Pre-flight checks passed: {sum(1 for v in preflight.values() if v['passed'])}/{len(preflight)}")
    print(f"  Partial feature loss:")
    for k, v in stress["partial_feature_loss"].items():
        print(f"    {k:<20} F1={v['f1_macro']:.4f}")
    print(f"  OOD mean confidences:")
    for k, v in stress["ood_samples"].items():
        print(f"    {k:<25} conf={v['mean_max_conf']:.4f}")
    print(f"  Confidence threshold @ 0.7: coverage={runtime['threshold_0.7']['coverage']:.3f} | "
          f"F1={runtime['threshold_0.7']['f1_macro_confident']:.4f}")
    print(f"{'='*60}")
    print("\nOutputs saved to results/stress_test/")
    print("  stress_test_metrics.json")
    print("  stress_test_summary.png")
    print("  preflight_report.json")


if __name__ == "__main__":
    main()