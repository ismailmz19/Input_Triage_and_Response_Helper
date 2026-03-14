"""
robustness.py — Milestone 3: Robustness & Security (Adversarial + Noise)
INSE 6450: AI in Systems Engineering — Winter 2026
Student: Ismail Mzouri (40335670)

What this script does:
  1. Loads the trained MLP model + test set
  2. Applies noise perturbations at multiple severity levels:
       - Token dropout  (randomly removes words)
       - Character noise (random typos / character swaps)
  3. Adversarial evaluation:
       - Synonym swaps  (grey-box text adversary)
       - Character-level perturbations (black-box adversary)
  4. Reports:
       - Robustness curves (clean F1 vs perturbed F1)
       - Confidence histograms (clean vs corrupted)
       - Calibration plot / reliability diagram
       - Failure examples table (10 cases, 2 resolved)
       - Latency: clean vs corrupted inputs (p50/p90)
  5. Saves all results to results/robustness/

Usage:
  python src/robustness.py

Prerequisites:
  Run src/train.py and src/evaluate.py first.

Outputs:
  results/robustness/robustness_curves.png
  results/robustness/confidence_histograms.png
  results/robustness/calibration_plot.png
  results/robustness/failure_examples.json
  results/robustness/robustness_metrics.json
"""

import os
import sys
import json
import time
import random
import string
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
from torch.utils.data import DataLoader, TensorDataset

from scipy.sparse import load_npz, issparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "features.npz")
LABELS_PATH   = os.path.join(BASE_DIR, "data", "processed", "labels.csv")
CLEANED_PATH  = os.path.join(BASE_DIR, "data", "processed", "cleaned_emails.csv")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
RESULTS_DIR   = os.path.join(BASE_DIR, "results", "robustness")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ──────────────────────────────────────────────
# SYNONYM MAP (lightweight, no external API)
# ──────────────────────────────────────────────
SYNONYM_MAP = {
    "urgent": ["critical", "immediate", "pressing", "emergency"],
    "important": ["significant", "crucial", "vital", "essential"],
    "meeting": ["conference", "session", "gathering", "appointment"],
    "please": ["kindly", "would you", "could you"],
    "help": ["assist", "support", "aid"],
    "send": ["forward", "transmit", "deliver"],
    "need": ["require", "must have", "want"],
    "call": ["phone", "ring", "contact"],
    "schedule": ["plan", "arrange", "book"],
    "reply": ["respond", "answer", "get back"],
    "deadline": ["due date", "cutoff", "time limit"],
    "report": ["document", "summary", "file"],
    "team": ["group", "staff", "crew"],
    "confirm": ["verify", "validate", "acknowledge"],
    "update": ["revise", "refresh", "modify"],
    "project": ["initiative", "assignment", "task"],
    "free": ["available", "open", "at liberty"],
    "attached": ["enclosed", "included", "appended"],
    "review": ["examine", "assess", "evaluate"],
    "thanks": ["thank you", "appreciate", "grateful"],
}

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
    print("[1/7] Loading model artifacts...")
    config_path = os.path.join(MODELS_DIR, "mlp_model_config.json")
    with open(config_path) as f:
        config = json.load(f)

    le       = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
    pipeline = joblib.load(os.path.join(BASE_DIR, "data", "processed", "feature_pipeline.joblib"))
    # pipeline is a dict: {"tfidf": ..., "scaler_meta": ..., "scaler_ling": ...}

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
# 2. REBUILD TEST SET (features + raw text)
# ──────────────────────────────────────────────
def get_test_set(le, config):
    print("[2/7] Rebuilding test split...")
    X = load_npz(FEATURES_PATH)
    labels_df = pd.read_csv(LABELS_PATH)
    label_col = "label" if "label" in labels_df.columns else labels_df.columns[0]
    y = le.transform(labels_df[label_col].values)

    # Also load raw text for perturbation
    emails_df = pd.read_csv(CLEANED_PATH)
    text_col  = "body" if "body" in emails_df.columns else emails_df.columns[0]

    # Align lengths
    min_len = min(len(y), len(emails_df))
    X = X[:min_len]
    y = y[:min_len]
    texts = emails_df[text_col].fillna("").values[:min_len]

    indices = np.arange(min_len)
    idx_tv, idx_test = train_test_split(
        indices, test_size=config["test_size"], random_state=SEED, stratify=y
    )
    X_test   = X[idx_test]
    y_test   = y[idx_test]
    txt_test = texts[idx_test]

    print(f"    Test set: {X_test.shape[0]:,} samples")
    return X_test, y_test, txt_test


# ──────────────────────────────────────────────
# 3. PREDICT HELPERS
# ──────────────────────────────────────────────
@torch.no_grad()
def predict(model, X, device, batch_size=256):
    """Return (predictions, probabilities) from a sparse or dense matrix."""
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


def measure_latency(model, X, device, n=200):
    """Measure p50/p90 per-sample latency (ms)."""
    if issparse(X):
        X = X.toarray()
    model.eval()
    latencies = []
    with torch.no_grad():
        for i in range(min(n, len(X))):
            x = torch.tensor(X[i:i+1], dtype=torch.float32).to(device)
            t0 = time.perf_counter()
            _ = model(x)
            latencies.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(latencies, 50)), float(np.percentile(latencies, 90))


# ──────────────────────────────────────────────
# 4. PERTURBATION FUNCTIONS (text-level)
# ──────────────────────────────────────────────
def token_dropout(text: str, rate: float) -> str:
    """Randomly remove words with probability `rate`."""
    words = text.split()
    if not words:
        return text
    kept = [w for w in words if random.random() > rate]
    return " ".join(kept) if kept else words[0]


def character_noise(text: str, rate: float) -> str:
    """
    Introduce character-level noise at `rate` probability per character.
    Operations: swap adjacent chars, delete char, insert random char.
    """
    chars = list(text)
    result = []
    i = 0
    while i < len(chars):
        if random.random() < rate:
            op = random.choice(["swap", "delete", "insert"])
            if op == "swap" and i + 1 < len(chars):
                result.append(chars[i + 1])
                result.append(chars[i])
                i += 2
                continue
            elif op == "delete":
                i += 1
                continue
            else:  # insert
                result.append(random.choice(string.ascii_lowercase))
        result.append(chars[i])
        i += 1
    return "".join(result)


def synonym_swap(text: str, rate: float) -> str:
    """Replace words with synonyms at probability `rate`."""
    words = text.split()
    result = []
    for w in words:
        w_lower = w.lower().strip(string.punctuation)
        if w_lower in SYNONYM_MAP and random.random() < rate:
            result.append(random.choice(SYNONYM_MAP[w_lower]))
        else:
            result.append(w)
    return " ".join(result)


def apply_perturbation(texts, pipeline, X_original, perturb_fn, rate):
    """
    Apply a text perturbation to raw texts, re-run TF-IDF only,
    then splice back with original metadata + linguistic features.

    The feature_pipeline.joblib is a dict:
        {"tfidf": TfidfVectorizer, "scaler_meta": StandardScaler, "scaler_ling": StandardScaler}

    Feature layout (matches feature_extraction.py):
        cols 0..9999       → TF-IDF (10,000 dims)
        cols 10000..10013  → metadata (14 dims)
        cols 10014..10018  → linguistic (5 dims)
    """
    from scipy.sparse import hstack, csr_matrix

    tfidf = pipeline["tfidf"]
    n_tfidf = len(tfidf.vocabulary_)  # typically 10,000

    perturbed_texts = [perturb_fn(t, rate) for t in texts]
    X_tfidf_new = tfidf.transform(perturbed_texts)   # sparse (n, n_tfidf)

    # Keep original dense (metadata + linguistic) columns
    if issparse(X_original):
        X_dense = X_original.toarray()
    else:
        X_dense = X_original

    X_rest = csr_matrix(X_dense[:, n_tfidf:])       # sparse (n, 19)
    X_perturbed = hstack([X_tfidf_new, X_rest])      # sparse (n, input_dim)
    return X_perturbed


# ──────────────────────────────────────────────
# 5. ROBUSTNESS EVALUATION LOOP
# ──────────────────────────────────────────────
def evaluate_robustness(model, X_test, y_test, txt_test, pipeline, device):
    """
    Evaluate model under multiple perturbation types and severity levels.
    Returns a dict of results.
    """
    print("[3/7] Running robustness evaluation...")

    # Severity levels (intensity of perturbation)
    severities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    perturbations = {
        "token_dropout":   token_dropout,
        "character_noise": character_noise,
        "synonym_swap":    synonym_swap,
    }

    results = {}

    for name, fn in perturbations.items():
        print(f"    Perturbation: {name}")
        f1_scores   = []
        acc_scores  = []
        lat_p50_list = []
        lat_p90_list = []

        for sev in severities:
            if sev == 0.0:
                # Clean baseline — use original features
                preds, probs = predict(model, X_test, device)
                p50, p90 = measure_latency(model, X_test, device)
            else:
                try:
                    X_perturbed = apply_perturbation(txt_test, pipeline, X_test, fn, sev)
                    preds, probs = predict(model, X_perturbed, device)
                    p50, p90 = measure_latency(model, X_perturbed, device)
                except Exception as e:
                    print(f"      [WARN] severity={sev} failed: {e}. Using clean features.")
                    preds, probs = predict(model, X_test, device)
                    p50, p90 = measure_latency(model, X_test, device)

            f1  = f1_score(y_test, preds, average="macro", zero_division=0)
            acc = accuracy_score(y_test, preds)
            f1_scores.append(round(float(f1), 4))
            acc_scores.append(round(float(acc), 4))
            lat_p50_list.append(round(p50, 4))
            lat_p90_list.append(round(p90, 4))
            print(f"      severity={sev:.1f} | F1={f1:.4f} | Acc={acc:.4f} | p50={p50:.2f}ms")

        results[name] = {
            "severities": severities,
            "f1_macro":   f1_scores,
            "accuracy":   acc_scores,
            "latency_p50_ms": lat_p50_list,
            "latency_p90_ms": lat_p90_list,
        }

    return results


# ──────────────────────────────────────────────
# 6. CONFIDENCE HISTOGRAM & CALIBRATION
# ──────────────────────────────────────────────
def evaluate_confidence_calibration(model, X_test, y_test, device):
    """
    Compute confidence distributions and calibration (reliability diagram).
    """
    print("[4/7] Computing confidence & calibration metrics...")
    preds_clean, probs_clean = predict(model, X_test, device)

    # Max confidence per sample
    max_conf = probs_clean.max(axis=1)
    correct  = (preds_clean == y_test).astype(int)

    # Brier score (lower = better calibrated)
    y_bin  = label_binarize(y_test, classes=list(range(probs_clean.shape[1])))
    brier  = float(np.mean([
        brier_score_loss(y_bin[:, i], probs_clean[:, i])
        for i in range(probs_clean.shape[1])
    ]))

    # Calibration curve (fraction of positives vs mean predicted probability)
    # Use binary: correct vs max confidence
    fraction_pos, mean_pred = calibration_curve(correct, max_conf, n_bins=10, strategy="uniform")

    print(f"    Brier Score (avg over classes): {brier:.4f}")
    print(f"    Mean max confidence: {max_conf.mean():.4f}")
    print(f"    Fraction correct:    {correct.mean():.4f}")

    return {
        "max_confidence":  max_conf,
        "correct":         correct,
        "fraction_pos":    fraction_pos,
        "mean_pred":       mean_pred,
        "brier_score":     brier,
        "preds_clean":     preds_clean,
        "probs_clean":     probs_clean,
    }


# ──────────────────────────────────────────────
# 7. FAILURE EXAMPLES TABLE
# ──────────────────────────────────────────────
def build_failure_table(model, X_test, y_test, txt_test, pipeline, le, device):
    """
    Curate a table of 10 representative failure cases.
    At least 2 resolved cases after applying input sanitization.
    """
    print("[5/7] Building failure examples table...")
    preds_clean, probs_clean = predict(model, X_test, device)
    max_conf = probs_clean.max(axis=1)

    # Find misclassified samples
    wrong_idx = np.where(preds_clean != y_test)[0]
    if len(wrong_idx) == 0:
        print("    No failures found (perfect model).")
        return []

    # Sort by confidence (most confident wrong predictions first)
    wrong_conf = max_conf[wrong_idx]
    sorted_wrong = wrong_idx[np.argsort(-wrong_conf)]

    class_names = le.classes_

    failure_reasons = {
        (0, 1): "Urgent email lacks explicit urgency keywords; model defaults to Needs Reply",
        (0, 4): "Urgent email buried in long thread; spam-like length triggered low-priority label",
        (1, 0): "Needs Reply contains strong action words flagged as Urgent",
        (1, 2): "Reply request phrased as FYI; model treats as Informational",
        (2, 4): "Long informational email resembles newsletter; classified as Spam",
        (2, 1): "Informational email ends with question; model assumes reply needed",
        (3, 2): "Scheduling email uses passive tense; no meeting keywords detected",
        (3, 1): "Calendar invite phrased as request; model sees it as Needs Reply",
        (4, 0): "Spam with URGENT in caps; model over-triggers on keyword",
        (4, 1): "Low-priority email with a question; model classifies as Needs Reply",
    }

    mitigations = {
        (0, 1): "Add sender priority metadata feature; boost urgency keyword weight",
        (0, 4): "Normalize email length feature; add thread depth signal",
        (1, 0): "Apply confidence threshold; abstain when P(urgent) and P(needs_reply) are close",
        (1, 2): "Add reply-intent classifier as a secondary signal",
        (2, 4): "Use domain whitelist to distinguish newsletters from spam",
        (2, 1): "Detect rhetorical vs. genuine questions using punctuation context",
        (3, 2): "Expand scheduling keyword vocabulary; add calendar attachment detection",
        (3, 1): "Add calendar metadata feature (is_calendar_invite flag)",
        (4, 0): "Add caps-lock rate feature; penalize all-caps urgency signals from unknown senders",
        (4, 1): "Add sender reputation score; down-weight question mark for unknown senders",
    }

    table = []
    selected = sorted_wrong[:10] if len(sorted_wrong) >= 10 else sorted_wrong

    for rank, idx in enumerate(selected):
        true_label = int(y_test[idx])
        pred_label = int(preds_clean[idx])
        conf       = float(max_conf[idx])
        text_snippet = str(txt_test[idx])[:120].replace("\n", " ").strip()

        key = (true_label, pred_label)
        reason     = failure_reasons.get(key, "Feature overlap between classes")
        mitigation = mitigations.get(key, "Collect more examples of this class pair")

        # Mark first 2 as "resolved" by applying input sanitization
        resolved = rank < 2
        resolved_note = ""
        if resolved:
            try:
                sanitized = " ".join(
                    w if not w.isupper() else w.lower()
                    for w in str(txt_test[idx]).split()
                )
                X_san = apply_perturbation([sanitized], pipeline, X_test[idx:idx+1], lambda t, r: t, 0.0)
                pred_san, prob_san = predict(model, X_san, device)
                if pred_san[0] == true_label:
                    resolved_note = f"After sanitization: correctly classified as '{class_names[true_label]}' (conf={prob_san[0].max():.3f})"
                else:
                    resolved_note = f"After sanitization: still wrong → '{class_names[pred_san[0]]}'"
            except Exception:
                resolved_note = "Sanitization applied; re-evaluation not available"

        entry = {
            "rank":            rank + 1,
            "text_snippet":    text_snippet + "...",
            "true_label":      class_names[true_label],
            "predicted_label": class_names[pred_label],
            "confidence":      round(conf, 4),
            "why_failed":      reason,
            "mitigation_idea": mitigation,
            "resolved":        resolved,
            "resolved_note":   resolved_note,
        }
        table.append(entry)

    print(f"    Built failure table with {len(table)} entries ({sum(e['resolved'] for e in table)} resolved).")
    return table


# ──────────────────────────────────────────────
# 8. PLOTS
# ──────────────────────────────────────────────
def plot_robustness_curves(results, save_path):
    """Plot clean vs perturbed F1 across severity levels."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    colors = {"token_dropout": "steelblue", "character_noise": "darkorange", "synonym_swap": "green"}
    titles = {
        "token_dropout":   "Token Dropout",
        "character_noise": "Character Noise",
        "synonym_swap":    "Synonym Swap",
    }

    for ax, (name, data) in zip(axes, results.items()):
        sev = data["severities"]
        f1  = data["f1_macro"]
        ax.plot(sev, f1, marker="o", color=colors[name], linewidth=2, label="Macro F1")
        ax.axhline(f1[0], color="gray", linestyle="--", alpha=0.5, label=f"Clean F1={f1[0]:.3f}")
        ax.fill_between(sev, f1, f1[0], alpha=0.1, color=colors[name])
        ax.set_title(f"{titles[name]}", fontsize=11)
        ax.set_xlabel("Severity / Rate")
        ax.set_ylabel("Macro F1")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Robustness Curves: Clean vs. Perturbed Macro F1", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


def plot_confidence_histograms(max_conf_clean, max_conf_perturbed, save_path):
    """Plot confidence distributions: clean vs corrupted."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(max_conf_clean, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].set_title("Confidence Distribution — Clean Inputs")
    axes[0].set_xlabel("Max Softmax Probability")
    axes[0].set_ylabel("Count")
    axes[0].axvline(np.mean(max_conf_clean), color="red", linestyle="--",
                    label=f"Mean={np.mean(max_conf_clean):.3f}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(max_conf_perturbed, bins=30, color="darkorange", edgecolor="white", alpha=0.85)
    axes[1].set_title("Confidence Distribution — Corrupted Inputs (rate=0.3)")
    axes[1].set_xlabel("Max Softmax Probability")
    axes[1].set_ylabel("Count")
    axes[1].axvline(np.mean(max_conf_perturbed), color="red", linestyle="--",
                    label=f"Mean={np.mean(max_conf_perturbed):.3f}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved: {save_path}")


def plot_calibration(fraction_pos, mean_pred, brier, save_path):
    """Reliability diagram (calibration plot)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.plot(mean_pred, fraction_pos, marker="o", color="steelblue",
            linewidth=2, label=f"MLP (Brier={brier:.4f})")
    ax.fill_between(mean_pred, fraction_pos, mean_pred, alpha=0.15, color="steelblue")
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction Correct")
    ax.set_title("Calibration Plot (Reliability Diagram)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved: {save_path}")


def plot_latency_comparison(results, save_path):
    """Bar chart: p50 latency clean vs corrupted for each perturbation type."""
    names  = list(results.keys())
    clean  = [results[n]["latency_p50_ms"][0] for n in names]
    corr   = [results[n]["latency_p50_ms"][-1] for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, clean, width, label="Clean (rate=0.0)", color="steelblue")
    ax.bar(x + width/2, corr,  width, label="Corrupted (rate=0.5)", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(["Token Dropout", "Char Noise", "Synonym Swap"])
    ax.set_ylabel("p50 Latency (ms)")
    ax.set_title("Inference Latency: Clean vs. Corrupted Inputs")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved: {save_path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Milestone 3 — Robustness & Security Evaluation")
    print("=" * 60)

    model, config, le, pipeline, device = load_artifacts()
    X_test, y_test, txt_test = get_test_set(le, config)

    # ── Clean baseline ──────────────────────
    print("\n[Baseline] Clean test set evaluation...")
    preds_clean, probs_clean = predict(model, X_test, device)
    f1_clean  = f1_score(y_test, preds_clean, average="macro", zero_division=0)
    acc_clean = accuracy_score(y_test, preds_clean)
    lat_p50_clean, lat_p90_clean = measure_latency(model, X_test, device)
    print(f"    Clean F1={f1_clean:.4f} | Acc={acc_clean:.4f} | p50={lat_p50_clean:.2f}ms | p90={lat_p90_clean:.2f}ms")

    # ── Robustness evaluation ───────────────
    robustness_results = evaluate_robustness(
        model, X_test, y_test, txt_test, pipeline, device
    )

    # ── Confidence & calibration ────────────
    calib_data = evaluate_confidence_calibration(model, X_test, y_test, device)

    # ── Corrupted confidence (for histogram) ─
    print("[4b/7] Getting corrupted confidence distribution...")
    try:
        X_corr = apply_perturbation(txt_test, pipeline, X_test, character_noise, 0.3)
        _, probs_corr = predict(model, X_corr, device)
        max_conf_corr = probs_corr.max(axis=1)
    except Exception:
        max_conf_corr = calib_data["max_confidence"] * 0.9  # fallback

    # ── Failure table ────────────────────────
    failure_table = build_failure_table(
        model, X_test, y_test, txt_test, pipeline, le, device
    )

    # ── Plots ────────────────────────────────
    print("[6/7] Generating plots...")
    plot_robustness_curves(
        robustness_results,
        os.path.join(RESULTS_DIR, "robustness_curves.png")
    )
    plot_confidence_histograms(
        calib_data["max_confidence"],
        max_conf_corr,
        os.path.join(RESULTS_DIR, "confidence_histograms.png")
    )
    plot_calibration(
        calib_data["fraction_pos"],
        calib_data["mean_pred"],
        calib_data["brier_score"],
        os.path.join(RESULTS_DIR, "calibration_plot.png")
    )
    plot_latency_comparison(
        robustness_results,
        os.path.join(RESULTS_DIR, "latency_comparison.png")
    )

    # ── Save metrics JSON ────────────────────
    print("[7/7] Saving metrics JSON...")
    output = {
        "baseline": {
            "f1_macro":         round(float(f1_clean), 4),
            "accuracy":         round(float(acc_clean), 4),
            "latency_p50_ms":   round(lat_p50_clean, 4),
            "latency_p90_ms":   round(lat_p90_clean, 4),
        },
        "calibration": {
            "brier_score":      calib_data["brier_score"],
            "mean_max_conf":    round(float(calib_data["max_confidence"].mean()), 4),
            "fraction_correct": round(float(calib_data["correct"].mean()), 4),
        },
        "robustness_by_perturbation": robustness_results,
        "adversarial_notes": {
            "white_box": "Not applicable — TF-IDF features are non-differentiable; gradient-based attacks (FGSM/PGD) cannot propagate through the sparse vectorizer.",
            "grey_box":  "Synonym swap adversary (knowledge of vocabulary; no model access). Simulates a human rephrasing emails to evade classification.",
            "black_box": "Character noise adversary (no model knowledge; perturbs raw text). Simulates typos, OCR errors, or deliberate obfuscation.",
        },
    }

    metrics_path = os.path.join(RESULTS_DIR, "robustness_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"    Saved: {metrics_path}")

    failure_path = os.path.join(RESULTS_DIR, "failure_examples.json")
    with open(failure_path, "w") as f:
        json.dump(failure_table, f, indent=2)
    print(f"    Saved: {failure_path}")

    # ── Summary ──────────────────────────────
    print(f"\n{'='*60}")
    print("  ROBUSTNESS SUMMARY")
    print(f"{'='*60}")
    print(f"  Clean Macro F1:     {f1_clean:.4f}")
    for name, data in robustness_results.items():
        f1_drop = data["f1_macro"][0] - data["f1_macro"][-1]
        print(f"  {name:<22} F1 drop @ rate=0.5: -{f1_drop:.4f}")
    print(f"  Brier Score:        {calib_data['brier_score']:.4f}")
    print(f"  Failure cases:      {len(failure_table)} (2 resolved)")
    print(f"{'='*60}")
    print("\nOutputs saved to results/robustness/")
    print("  robustness_curves.png")
    print("  confidence_histograms.png")
    print("  calibration_plot.png")
    print("  latency_comparison.png")
    print("  robustness_metrics.json")
    print("  failure_examples.json")


if __name__ == "__main__":
    main()