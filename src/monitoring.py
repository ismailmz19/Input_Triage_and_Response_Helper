"""
monitoring.py — Milestone 3: Monitoring
INSE 6450: AI in Systems Engineering — Winter 2026
Student: Ismail Mzouri (40335670)

What this script does:
  1. Simulates a stream of batched email data arriving over 10 time windows
  2. Injects gradual drift starting at window 5 (class prior + feature shift)
  3. Computes drift statistics per window:
       - Population Stability Index (PSI) on TF-IDF features
       - KL divergence on class distribution
       - JS divergence on confidence scores
       - Rolling macro F1 and AUROC (proxy online metrics)
  4. Triggers alerts when drift thresholds are exceeded
  5. Renders a compact monitoring dashboard (6-panel figure)
  6. Saves all results to results/monitoring/

Usage:
  python src/monitoring.py

Prerequisites:
  Run src/train.py first.

Outputs:
  results/monitoring/monitoring_dashboard.png
  results/monitoring/monitoring_metrics.json
  results/monitoring/alert_log.json
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
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scipy.sparse import load_npz, issparse
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "features.npz")
LABELS_PATH   = os.path.join(BASE_DIR, "data", "processed", "labels.csv")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
RESULTS_DIR   = os.path.join(BASE_DIR, "results", "monitoring")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

N_TFIDF  = 10000
N_META   = 14
N_LING   = 5
N_WINDOWS = 10
WINDOW_SIZE = 1000   # samples per monitoring window

# ── Alert thresholds ──────────────────────────
PSI_WARN    = 0.1    # mild drift
PSI_ALERT   = 0.2    # significant drift → trigger retraining
F1_DROP     = 0.05   # F1 drops more than 5% below baseline
CONF_DROP   = 0.05   # mean confidence drops more than 5%

# ──────────────────────────────────────────────
# MLP DEFINITION (must match train.py)
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
    with open(os.path.join(MODELS_DIR, "mlp_model_config.json")) as f:
        config = json.load(f)
    le     = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
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
    return model, config, le, device


# ──────────────────────────────────────────────
# 2. LOAD TEST DATA (reference + stream)
# ──────────────────────────────────────────────
def load_data(le, config):
    print("[2/5] Loading data for stream simulation...")
    X = load_npz(FEATURES_PATH)
    labels_df = pd.read_csv(LABELS_PATH)
    label_col = "label" if "label" in labels_df.columns else labels_df.columns[0]
    y = le.transform(labels_df[label_col].values)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=SEED, stratify=y
    )
    print(f"    Test set (stream source): {X_test.shape[0]:,} samples")
    return X_test, y_test


# ──────────────────────────────────────────────
# 3. PREDICT HELPER
# ──────────────────────────────────────────────
@torch.no_grad()
def predict_batch(model, X, device, batch_size=512):
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
# 4. DRIFT STATISTICS
# ──────────────────────────────────────────────
def compute_psi(reference, current, n_bins=10, eps=1e-6):
    """
    Population Stability Index (PSI).
    PSI < 0.1:  No significant change
    PSI 0.1-0.2: Moderate change — monitor closely
    PSI > 0.2:  Significant change — retrain
    """
    combined = np.concatenate([reference, current])
    bins = np.percentile(combined, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current,   bins=bins)

    ref_pct = (ref_counts + eps) / (len(reference) + eps * n_bins)
    cur_pct = (cur_counts + eps) / (len(current)   + eps * n_bins)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_kl_divergence(p, q, eps=1e-6):
    """KL divergence between two probability distributions."""
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    return float(entropy(p, q))


def compute_js_divergence(p, q, eps=1e-6):
    """Jensen-Shannon divergence (symmetric, bounded [0,1])."""
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * entropy(p, m) + 0.5 * entropy(q, m))


# ──────────────────────────────────────────────
# 5. STREAM SIMULATION
# ──────────────────────────────────────────────
def simulate_stream(X_test, y_test, n_windows, window_size, drift_start=5):
    """
    Simulate N_WINDOWS batches of incoming data.
    Windows 0..drift_start-1: clean data (sampled from test set)
    Windows drift_start..N-1: increasingly drifted data

    Drift types injected:
      - Class prior shift: Urgent becomes overrepresented
      - Feature mean shift: metadata offset grows each window
    """
    rng = np.random.RandomState(SEED)
    n   = X_test.shape[0]
    windows = []

    for w in range(n_windows):
        if w < drift_start:
            # Clean window: random sample
            idx = rng.choice(n, window_size, replace=True)
            if issparse(X_test):
                X_w = X_test[idx].toarray().astype(np.float32)
            else:
                X_w = X_test[idx].astype(np.float32)
            y_w = y_test[idx]
        else:
            # Drifted window: shift class priors + add feature noise
            drift_intensity = (w - drift_start + 1) / (n_windows - drift_start)

            # Oversample class 4 (Urgent)
            urgent_idx   = np.where(y_test == 4)[0]
            other_idx    = np.where(y_test != 4)[0]
            n_urgent     = int(window_size * min(0.1 + drift_intensity * 0.3, 0.4))
            n_other      = window_size - n_urgent

            if len(urgent_idx) == 0:
                idx = rng.choice(n, window_size, replace=True)
            else:
                idx_u = rng.choice(urgent_idx, n_urgent, replace=True)
                idx_o = rng.choice(other_idx,  n_other,  replace=True)
                idx   = np.concatenate([idx_u, idx_o])
                rng.shuffle(idx)

            if issparse(X_test):
                X_w = X_test[idx].toarray().astype(np.float32)
            else:
                X_w = X_test[idx].astype(np.float32)
            y_w = y_test[idx]

            # Add growing metadata shift
            meta_shift = rng.normal(0, drift_intensity * 0.8, (1, N_META)).astype(np.float32)
            X_w[:, N_TFIDF:N_TFIDF + N_META] += meta_shift

            # Add mild TF-IDF noise
            tfidf_noise = rng.normal(0, drift_intensity * 0.1,
                                     X_w[:, :N_TFIDF].shape).astype(np.float32)
            X_w[:, :N_TFIDF] = np.clip(X_w[:, :N_TFIDF] + tfidf_noise, 0, None)

        windows.append((X_w, y_w))

    return windows


# ──────────────────────────────────────────────
# 6. MONITORING LOOP
# ──────────────────────────────────────────────
def run_monitoring(model, windows, le, device, config):
    """
    Process each window: predict, compute drift stats, check alerts.
    """
    print("[4/5] Running monitoring loop...")

    num_classes  = config["num_classes"]
    class_names  = list(le.classes_)

    # ── Reference window (window 0) ──────────
    X_ref, y_ref = windows[0]
    _, probs_ref  = predict_batch(model, X_ref, device)
    ref_conf      = probs_ref.max(axis=1)
    ref_class_dist = np.bincount(y_ref, minlength=num_classes) / len(y_ref)

    # Reference PSI features: use mean of top-50 TF-IDF features + metadata
    ref_meta   = X_ref[:, N_TFIDF:N_TFIDF + N_META].mean(axis=1)  # scalar per sample
    ref_preds, _ = predict_batch(model, X_ref, device)
    ref_f1    = f1_score(y_ref, ref_preds, average="macro", zero_division=0)
    y_ref_bin = label_binarize(y_ref, classes=list(range(num_classes)))
    ref_auroc = roc_auc_score(y_ref_bin, probs_ref, multi_class="ovr", average="macro")

    print(f"    Reference window: F1={ref_f1:.4f} | AUROC={ref_auroc:.4f} | "
          f"MeanConf={ref_conf.mean():.4f}")

    # ── Per-window metrics ────────────────────
    window_metrics = []
    alert_log      = []

    for w_idx, (X_w, y_w) in enumerate(windows):
        preds_w, probs_w = predict_batch(model, X_w, device)

        # Classification metrics
        f1_w    = f1_score(y_w, preds_w, average="macro", zero_division=0)
        y_w_bin = label_binarize(y_w, classes=list(range(num_classes)))
        try:
            auroc_w = roc_auc_score(y_w_bin, probs_w, multi_class="ovr", average="macro")
        except Exception:
            auroc_w = 0.0

        # Confidence stats
        conf_w      = probs_w.max(axis=1)
        mean_conf_w = float(conf_w.mean())
        std_conf_w  = float(conf_w.std())

        # Class distribution
        class_dist_w = np.bincount(y_w, minlength=num_classes) / len(y_w)
        kl_class     = compute_kl_divergence(ref_class_dist, class_dist_w)
        js_class     = compute_js_divergence(ref_class_dist, class_dist_w)

        # PSI on metadata features (scalar per sample)
        meta_w = X_w[:, N_TFIDF:N_TFIDF + N_META].mean(axis=1)
        psi_meta = compute_psi(ref_meta, meta_w)

        # PSI on confidence scores
        psi_conf = compute_psi(ref_conf, conf_w)

        # JS on confidence distribution (binned)
        conf_bins_ref = np.histogram(ref_conf, bins=10, range=(0, 1))[0].astype(float)
        conf_bins_w   = np.histogram(conf_w,   bins=10, range=(0, 1))[0].astype(float)
        js_conf       = compute_js_divergence(conf_bins_ref, conf_bins_w)

        # ── Alert logic ───────────────────────
        alerts = []
        if psi_meta > PSI_ALERT:
            alerts.append({
                "type":    "PSI_ALERT",
                "window":  w_idx,
                "metric":  "psi_metadata",
                "value":   round(psi_meta, 4),
                "threshold": PSI_ALERT,
                "action":  "TRIGGER_RETRAIN",
                "message": f"Metadata PSI={psi_meta:.4f} > {PSI_ALERT} — significant distribution shift",
            })
        elif psi_meta > PSI_WARN:
            alerts.append({
                "type":    "PSI_WARN",
                "window":  w_idx,
                "metric":  "psi_metadata",
                "value":   round(psi_meta, 4),
                "threshold": PSI_WARN,
                "action":  "INCREASE_MONITORING",
                "message": f"Metadata PSI={psi_meta:.4f} > {PSI_WARN} — moderate drift detected",
            })

        if ref_f1 - f1_w > F1_DROP:
            alerts.append({
                "type":    "F1_DROP_ALERT",
                "window":  w_idx,
                "metric":  "f1_macro",
                "value":   round(float(f1_w), 4),
                "baseline": round(float(ref_f1), 4),
                "drop":    round(float(ref_f1 - f1_w), 4),
                "threshold": F1_DROP,
                "action":  "INVESTIGATE_DATA_QUALITY",
                "message": f"F1 dropped {ref_f1 - f1_w:.4f} below baseline",
            })

        if ref_conf.mean() - mean_conf_w > CONF_DROP:
            alerts.append({
                "type":    "CONFIDENCE_DROP",
                "window":  w_idx,
                "metric":  "mean_confidence",
                "value":   round(mean_conf_w, 4),
                "baseline": round(float(ref_conf.mean()), 4),
                "action":  "CHECK_INPUT_QUALITY",
                "message": f"Mean confidence dropped {ref_conf.mean() - mean_conf_w:.4f}",
            })

        alert_log.extend(alerts)

        metrics_w = {
            "window":          w_idx,
            "n_samples":       len(y_w),
            "f1_macro":        round(float(f1_w), 4),
            "auroc":           round(float(auroc_w), 4),
            "mean_confidence": round(mean_conf_w, 4),
            "std_confidence":  round(std_conf_w, 4),
            "psi_metadata":    round(psi_meta, 4),
            "psi_confidence":  round(psi_conf, 4),
            "kl_class_dist":   round(float(kl_class), 4),
            "js_class_dist":   round(float(js_class), 4),
            "class_distribution": {class_names[i]: round(float(class_dist_w[i]), 4)
                                   for i in range(num_classes)},
            "alerts":          len(alerts),
            "alert_types":     [a["type"] for a in alerts],
        }
        window_metrics.append(metrics_w)

        alert_str = f" ⚠ {len(alerts)} alert(s)" if alerts else ""
        print(f"    Window {w_idx:2d} | F1={f1_w:.4f} | PSI_meta={psi_meta:.4f} | "
              f"KL_class={kl_class:.4f} | Conf={mean_conf_w:.4f}{alert_str}")

    return window_metrics, alert_log, ref_f1, ref_auroc


# ──────────────────────────────────────────────
# 7. MONITORING DASHBOARD
# ──────────────────────────────────────────────
def plot_dashboard(window_metrics, alert_log, class_names, save_path):
    """
    6-panel monitoring dashboard:
    1. Rolling F1 over windows
    2. Rolling AUROC over windows
    3. PSI metadata over windows
    4. Mean confidence over windows
    5. KL divergence (class distribution) over windows
    6. Class distribution heatmap over windows
    """
    windows     = [m["window"] for m in window_metrics]
    f1s         = [m["f1_macro"] for m in window_metrics]
    aurocs      = [m["auroc"] for m in window_metrics]
    psi_metas   = [m["psi_metadata"] for m in window_metrics]
    confs       = [m["mean_confidence"] for m in window_metrics]
    kl_classes  = [m["kl_class_dist"] for m in window_metrics]
    alert_wins  = set(a["window"] for a in alert_log)

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Rolling F1 ───────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(windows, f1s, marker="o", color="steelblue", linewidth=2, label="Macro F1")
    ax1.axhline(f1s[0], color="gray", linestyle="--", alpha=0.6, label=f"Baseline={f1s[0]:.3f}")
    ax1.axhline(f1s[0] - F1_DROP, color="red", linestyle=":", alpha=0.6, label=f"Alert threshold")
    for w in alert_wins:
        ax1.axvline(w, color="red", alpha=0.2, linewidth=8)
    ax1.set_title("Rolling Macro F1")
    ax1.set_xlabel("Window")
    ax1.set_ylabel("Macro F1")
    ax1.set_ylim(max(0, min(f1s) - 0.05), min(1, max(f1s) + 0.05))
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Rolling AUROC ────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(windows, aurocs, marker="s", color="darkorange", linewidth=2, label="AUROC")
    ax2.axhline(aurocs[0], color="gray", linestyle="--", alpha=0.6, label=f"Baseline={aurocs[0]:.3f}")
    ax2.set_title("Rolling AUROC")
    ax2.set_xlabel("Window")
    ax2.set_ylabel("AUROC")
    ax2.set_ylim(max(0, min(aurocs) - 0.05), min(1, max(aurocs) + 0.05))
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: PSI Metadata ─────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    colors_psi = ["green" if p < PSI_WARN else "orange" if p < PSI_ALERT else "red"
                  for p in psi_metas]
    ax3.bar(windows, psi_metas, color=colors_psi, edgecolor="white")
    ax3.axhline(PSI_WARN,  color="orange", linestyle="--", alpha=0.8, label=f"Warn={PSI_WARN}")
    ax3.axhline(PSI_ALERT, color="red",    linestyle="--", alpha=0.8, label=f"Alert={PSI_ALERT}")
    ax3.set_title("PSI — Metadata Features")
    ax3.set_xlabel("Window")
    ax3.set_ylabel("PSI")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

    # ── Panel 4: Mean Confidence ──────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(windows, confs, marker="^", color="purple", linewidth=2, label="Mean Confidence")
    ax4.axhline(confs[0], color="gray", linestyle="--", alpha=0.6, label=f"Baseline={confs[0]:.3f}")
    ax4.axhline(confs[0] - CONF_DROP, color="red", linestyle=":", alpha=0.6, label="Alert threshold")
    for w in alert_wins:
        ax4.axvline(w, color="red", alpha=0.2, linewidth=8)
    ax4.set_title("Mean Prediction Confidence")
    ax4.set_xlabel("Window")
    ax4.set_ylabel("Mean Max Softmax")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── Panel 5: KL Divergence class dist ─────
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(windows, kl_classes, marker="D", color="brown", linewidth=2, label="KL Divergence")
    ax5.axhline(0.1, color="orange", linestyle="--", alpha=0.8, label="Warn=0.1")
    ax5.axhline(0.3, color="red",    linestyle="--", alpha=0.8, label="Alert=0.3")
    ax5.set_title("KL Divergence — Class Distribution")
    ax5.set_xlabel("Window")
    ax5.set_ylabel("KL Divergence")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ── Panel 6: Class distribution heatmap ───
    ax6 = fig.add_subplot(gs[2, 1])
    dist_matrix = np.array([
        [m["class_distribution"].get(c, 0) for c in class_names]
        for m in window_metrics
    ])
    im = ax6.imshow(dist_matrix.T, aspect="auto", cmap="YlOrRd",
                    vmin=0, vmax=dist_matrix.max())
    ax6.set_yticks(range(len(class_names)))
    ax6.set_yticklabels([c[:12] for c in class_names], fontsize=8)
    ax6.set_xticks(windows)
    ax6.set_xticklabels([str(w) for w in windows])
    ax6.set_title("Class Distribution Heatmap")
    ax6.set_xlabel("Window")
    ax6.set_ylabel("Class")
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)

    # Drift start annotation
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.axvline(5, color="black", linestyle="-.", alpha=0.4, linewidth=1.5)

    fig.suptitle(
        "Production Monitoring Dashboard — Inbox Triage & Response Helper\n"
        "(Drift injected at window 5 — black dashed line | Red shading = alert windows)",
        fontsize=12, y=1.01
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {save_path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Milestone 3 — Monitoring Dashboard")
    print("=" * 60)

    model, config, le, device = load_artifacts()
    X_test, y_test = load_data(le, config)

    # ── Simulate stream ───────────────────────
    print(f"\n[3/5] Simulating {N_WINDOWS} monitoring windows "
          f"({WINDOW_SIZE} samples each, drift starts at window 5)...")
    windows = simulate_stream(
        X_test, y_test,
        n_windows=N_WINDOWS,
        window_size=WINDOW_SIZE,
        drift_start=5
    )

    # ── Run monitoring ────────────────────────
    window_metrics, alert_log, ref_f1, ref_auroc = run_monitoring(
        model, windows, le, device, config
    )

    # ── Dashboard ─────────────────────────────
    print("\n[5/5] Generating monitoring dashboard...")
    plot_dashboard(
        window_metrics, alert_log, list(le.classes_),
        os.path.join(RESULTS_DIR, "monitoring_dashboard.png")
    )

    # ── Save JSON ─────────────────────────────
    output = {
        "monitoring_config": {
            "n_windows":       N_WINDOWS,
            "window_size":     WINDOW_SIZE,
            "drift_start":     5,
            "psi_warn":        PSI_WARN,
            "psi_alert":       PSI_ALERT,
            "f1_drop_alert":   F1_DROP,
            "conf_drop_alert": CONF_DROP,
        },
        "reference_window": {
            "f1_macro": round(float(ref_f1), 4),
            "auroc":    round(float(ref_auroc), 4),
        },
        "window_metrics": window_metrics,
        "total_alerts":   len(alert_log),
        "monitoring_plan": {
            "data_quality": [
                "Schema checks: feature dimension validated before every prediction batch",
                "Null rate: alert if >5% of inputs are all-zero in a window",
                "Distribution stats: PSI on metadata features per window (warn>0.1, alert>0.2)",
                "Rare-category presence: flag if any class drops below 3% of window",
                "Drift detector: PSI + KL divergence computed every 1,000 emails",
            ],
            "model_quality": [
                "Online proxy: mean confidence per window (drop >5% triggers alert)",
                "Rolling F1: computed per window using weak-supervision labels as proxy",
                "Rolling AUROC: computed per window for ranking quality",
                "Class distribution shift: KL divergence on predicted class distribution",
                "Delayed truth: re-evaluate F1 monthly when human-reviewed labels arrive",
            ],
            "architecture": {
                "metric_logging":   "Server-side batch logging (per 1,000 email window)",
                "storage":          "JSON logs retained for 90 days; aggregated stats for 1 year",
                "batching_window":  "1,000 emails or 24 hours, whichever comes first",
                "retention":        "Raw predictions: 30 days | Aggregated metrics: 1 year",
            },
            "alerting": {
                "psi_warn":         f"PSI > {PSI_WARN}: increase monitoring frequency to every 500 emails",
                "psi_alert":        f"PSI > {PSI_ALERT}: page on-call, trigger retraining pipeline",
                "f1_drop":          f"F1 drops > {F1_DROP} below baseline: investigate data quality",
                "confidence_drop":  f"Mean confidence drops > {CONF_DROP}: check input preprocessing",
                "runbook": [
                    "1. Verify data pipeline integrity (schema check, null rate)",
                    "2. Inspect recent emails for vocabulary shift or new email types",
                    "3. Run stress_test.py pre-flight checks on new batch",
                    "4. If PSI > 0.2: trigger drift_simulation.py head fine-tune",
                    "5. Deploy adapted model, monitor for 2 windows before retiring old model",
                    "6. If F1 does not recover: escalate to full retrain",
                ],
            },
        },
    }

    metrics_path = os.path.join(RESULTS_DIR, "monitoring_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"    Saved: {metrics_path}")

    alert_path = os.path.join(RESULTS_DIR, "alert_log.json")
    with open(alert_path, "w") as f:
        json.dump(alert_log, f, indent=2)
    print(f"    Saved: {alert_path}")

    # ── Summary ───────────────────────────────
    print(f"\n{'='*60}")
    print("  MONITORING SUMMARY")
    print(f"{'='*60}")
    print(f"  Reference F1:    {ref_f1:.4f} | AUROC: {ref_auroc:.4f}")
    print(f"  Windows monitored: {N_WINDOWS} ({WINDOW_SIZE} samples each)")
    print(f"  Total alerts fired: {len(alert_log)}")
    alert_types = {}
    for a in alert_log:
        alert_types[a["type"]] = alert_types.get(a["type"], 0) + 1
    for atype, cnt in alert_types.items():
        print(f"    {atype}: {cnt}")
    final_f1   = window_metrics[-1]["f1_macro"]
    final_psi  = window_metrics[-1]["psi_metadata"]
    print(f"  Final window F1:  {final_f1:.4f} (baseline: {ref_f1:.4f})")
    print(f"  Final window PSI: {final_psi:.4f}")
    print(f"{'='*60}")
    print("\nOutputs saved to results/monitoring/")
    print("  monitoring_dashboard.png")
    print("  monitoring_metrics.json")
    print("  alert_log.json")


if __name__ == "__main__":
    main()