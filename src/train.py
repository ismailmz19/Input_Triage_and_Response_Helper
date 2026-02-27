"""
train.py — Milestone 2: Model Training & Experimentation
INSE 6450: AI in Systems Engineering — Winter 2026
Student: Ismail Mzouri (40335670)

What this script does:
  1. Loads features.npz + labels.csv from Milestone 1
  2. Stratified 70 / 15 / 15 train / val / test split
  3. Trains a PyTorch MLP classifier (primary model)
  4. Trains a Logistic Regression baseline (for ablation comparison)
  5. Handles class imbalance via class-weighted cross-entropy loss
  6. Logs loss + macro-F1 every epoch → saves learning_curves.png
  7. Saves model weights, config JSON, and label encoder
  8. Prints a full classification report on the test set

Usage:
  python src/train.py

Outputs:
  models/mlp_model.pth
  models/mlp_model_config.json
  models/label_encoder.joblib
  models/logreg_baseline.joblib
  results/learning_curves.png
  results/confusion_matrix.png
  results/classification_report.txt
  results/ablation_comparison.json
"""

import os
import json
import time
import random
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
from scipy.sparse import load_npz, issparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ──────────────────────────────────────────────
# 0. REPRODUCIBILITY
# ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ──────────────────────────────────────────────
# 1. HYPERPARAMETERS  (versioned config)
# ──────────────────────────────────────────────
CONFIG = {
    "seed": SEED,
    "test_size": 0.15,
    "val_size": 0.15,
    "hidden_dims": [512, 256, 128],
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "epochs": 50,
    "early_stopping_patience": 7,
    "optimizer": "Adam",
    "loss": "CrossEntropyLoss (class-weighted)",
    "activation": "ReLU",
    "num_classes": 5,
}

# ──────────────────────────────────────────────
# 2. PATHS
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "features.npz")
LABELS_PATH   = os.path.join(BASE_DIR, "data", "processed", "labels.csv")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 3. LOAD DATA
# ──────────────────────────────────────────────
def load_data():
    print("[1/6] Loading features and labels...")
    X = load_npz(FEATURES_PATH)          # sparse matrix
    labels_df = pd.read_csv(LABELS_PATH)

    # Accept either a 'label' or 'category' column
    label_col = "label" if "label" in labels_df.columns else labels_df.columns[0]
    y_raw = labels_df[label_col].values

    # Encode string labels → integers
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"    Dataset: {X.shape[0]:,} samples | {X.shape[1]:,} features | {len(le.classes_)} classes")
    print(f"    Classes: {list(le.classes_)}")

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(le.classes_, counts):
        print(f"      {cls}: {cnt:,} ({cnt/len(y)*100:.1f}%)")

    return X, y, le


# ──────────────────────────────────────────────
# 4. PYTORCH MLP MODEL
# ──────────────────────────────────────────────
class EmailMLP(nn.Module):
    """
    Multi-Layer Perceptron for email classification.

    Architecture:
      Input → [Linear → BatchNorm → ReLU → Dropout] × N layers → Linear → logits

    Inductive bias: fully connected layers treat each TF-IDF/metadata
    feature independently and learn arbitrary non-linear combinations,
    which is appropriate for bag-of-words style feature vectors where
    spatial/sequential structure is not meaningful.
    """
    def __init__(self, input_dim: int, hidden_dims: list, num_classes: int, dropout: float):
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
# 5. TRAINING UTILITIES
# ──────────────────────────────────────────────
def compute_class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights to handle imbalance."""
    counts = np.bincount(y_train, minlength=num_classes).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes   # normalize
    return torch.tensor(weights, dtype=torch.float32)


def make_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size):
    """Convert numpy/sparse arrays to PyTorch DataLoaders."""
    def to_tensor(X, y):
        if issparse(X):
            X = X.toarray()
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )

    Xt, yt = to_tensor(X_train, y_train)
    Xv, yv = to_tensor(X_val,   y_val)
    Xs, ys = to_tensor(X_test,  y_test)

    train_loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xv, yv), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(Xs, ys), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, Xs, ys


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1, all_preds, all_labels


# ──────────────────────────────────────────────
# 6. LEARNING CURVE PLOT
# ──────────────────────────────────────────────
def plot_learning_curves(history: dict, save_path: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   color="orange")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Loss vs. Epoch"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_f1"], label="Train F1 (macro)", color="steelblue")
    ax2.plot(epochs, history["val_f1"],   label="Val F1 (macro)",   color="orange")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro F1")
    ax2.set_title("Macro F1 vs. Epoch"); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved: {save_path}")


# ──────────────────────────────────────────────
# 7. CONFUSION MATRIX PLOT
# ──────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30)
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved: {save_path}")


# ──────────────────────────────────────────────
# 8. MAIN TRAINING PIPELINE
# ──────────────────────────────────────────────
def main():
    t_start = time.time()

    # ── Load ──────────────────────────────────
    X, y, le = load_data()
    num_classes = len(le.classes_)
    CONFIG["num_classes"] = num_classes
    input_dim = X.shape[1]
    CONFIG["input_dim"] = input_dim

    # ── Split: 70 / 15 / 15 ──────────────────
    print("\n[2/6] Splitting dataset (stratified 70/15/15)...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=CONFIG["test_size"], random_state=SEED, stratify=y
    )
    val_relative = CONFIG["val_size"] / (1.0 - CONFIG["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_relative, random_state=SEED, stratify=y_train_val
    )
    print(f"    Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,}")

    # ── DataLoaders ───────────────────────────
    train_loader, val_loader, test_loader, X_test_tensor, y_test_tensor = \
        make_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, CONFIG["batch_size"])

    # ── Device ────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    # ── Class weights (imbalance mitigation) ──
    class_weights = compute_class_weights(y_train, num_classes).to(device)
    print(f"    Class weights: {class_weights.cpu().numpy().round(3)}")

    # ── Model ─────────────────────────────────
    print("\n[3/6] Building PyTorch MLP...")
    model = EmailMLP(
        input_dim=input_dim,
        hidden_dims=CONFIG["hidden_dims"],
        num_classes=num_classes,
        dropout=CONFIG["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    CONFIG["total_parameters"] = total_params
    CONFIG["trainable_parameters"] = trainable_params
    print(f"    Parameters: {total_params:,} total | {trainable_params:,} trainable")
    print(f"    Architecture: input({input_dim}) → {CONFIG['hidden_dims']} → {num_classes}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    # ── Training Loop ─────────────────────────
    print(f"\n[4/6] Training for up to {CONFIG['epochs']} epochs (early stopping patience={CONFIG['early_stopping_patience']})...")
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_val_f1  = 0.0
    patience_ctr = 0
    best_epoch   = 0

    epoch_times = []
    for epoch in range(1, CONFIG["epochs"] + 1):
        t_ep = time.time()
        tr_loss, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_f1, _, _ = evaluate(model, val_loader, criterion, device)
        ep_time = time.time() - t_ep
        epoch_times.append(ep_time)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(vl_f1)

        scheduler.step(vl_f1)

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            best_epoch  = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "mlp_model.pth"))
        else:
            patience_ctr += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{CONFIG['epochs']} | "
                  f"Train Loss: {tr_loss:.4f} F1: {tr_f1:.4f} | "
                  f"Val Loss: {vl_loss:.4f} F1: {vl_f1:.4f} | "
                  f"Time: {ep_time:.1f}s")

        if patience_ctr >= CONFIG["early_stopping_patience"]:
            print(f"    Early stopping at epoch {epoch} (best val F1: {best_val_f1:.4f} at epoch {best_epoch})")
            break

    CONFIG["actual_epochs_trained"] = epoch
    CONFIG["best_epoch"] = best_epoch
    CONFIG["best_val_f1"] = round(float(best_val_f1), 4)
    CONFIG["avg_time_per_epoch_sec"] = round(float(np.mean(epoch_times)), 3)
    CONFIG["total_training_time_sec"] = round(float(time.time() - t_start), 2)

    # ── Learning Curves ───────────────────────
    print("\n[5/6] Generating plots...")
    plot_learning_curves(history, os.path.join(RESULTS_DIR, "learning_curves.png"))

    # ── Test Set Evaluation ───────────────────
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "mlp_model.pth"), map_location=device))
    _, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    print(f"\n    ── Test Results ──")
    print(f"    Macro F1: {test_f1:.4f}")
    report_str = classification_report(
        test_labels, test_preds,
        target_names=le.classes_, zero_division=0
    )
    print(report_str)

    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write("MLP Classifier — Test Set Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(report_str)
    print(f"    Saved: {os.path.join(RESULTS_DIR, 'classification_report.txt')}")

    plot_confusion_matrix(
        test_labels, test_preds, le.classes_,
        os.path.join(RESULTS_DIR, "confusion_matrix.png")
    )

    # ── Ablation: Logistic Regression Baseline ─
    # Use 'saga' solver which works natively on sparse matrices (no .toarray() needed)
    # Subsample to 30k training examples to keep memory manageable on CPU
    print("\n[6/6] Training Logistic Regression baseline (ablation)...")
    t_lr = time.time()
    lr_model = LogisticRegression(
        max_iter=1000, random_state=SEED,
        class_weight="balanced", solver="saga", multi_class="multinomial",
        n_jobs=-1
    )
    # Subsample training set if too large to avoid OOM
    MAX_LR_SAMPLES = 30000
    if len(y_train) > MAX_LR_SAMPLES:
        idx = np.random.RandomState(SEED).choice(len(y_train), MAX_LR_SAMPLES, replace=False)
        X_lr_train = X_train[idx]
        y_lr_train = y_train[idx]
        print(f"    Subsampled to {MAX_LR_SAMPLES:,} samples for LogReg baseline")
    else:
        X_lr_train = X_train
        y_lr_train = y_train
    lr_model.fit(X_lr_train, y_lr_train)
    lr_train_time = time.time() - t_lr
    lr_preds = lr_model.predict(X_test)
    lr_f1 = f1_score(y_test, lr_preds, average="macro", zero_division=0)

    ablation = {
        "logistic_regression_baseline": {
            "test_macro_f1":      round(float(lr_f1), 4),
            "training_time_sec":  round(lr_train_time, 2),
            "model_type":         "LogisticRegression (balanced class weights)",
        },
        "mlp_final_model": {
            "test_macro_f1":      round(float(test_f1), 4),
            "training_time_sec":  CONFIG["total_training_time_sec"],
            "model_type":         f"PyTorch MLP {CONFIG['hidden_dims']}",
        },
        "f1_improvement":         round(float(test_f1 - lr_f1), 4),
    }
    print(f"    Logistic Regression F1:  {lr_f1:.4f}")
    print(f"    MLP F1:                  {test_f1:.4f}")
    print(f"    Improvement:             +{test_f1 - lr_f1:.4f}")

    with open(os.path.join(RESULTS_DIR, "ablation_comparison.json"), "w") as f:
        json.dump(ablation, f, indent=2)
    print(f"    Saved: {os.path.join(RESULTS_DIR, 'ablation_comparison.json')}")

    # ── Save artifacts ────────────────────────
    joblib.dump(le,       os.path.join(MODELS_DIR, "label_encoder.joblib"))
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "logreg_baseline.joblib"))

    with open(os.path.join(MODELS_DIR, "mlp_model_config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"\n    Saved model config:   {os.path.join(MODELS_DIR, 'mlp_model_config.json')}")
    print(f"    Saved model weights:  {os.path.join(MODELS_DIR, 'mlp_model.pth')}")
    print(f"    Saved label encoder:  {os.path.join(MODELS_DIR, 'label_encoder.joblib')}")

    print(f"\n{'='*60}")
    print(f"  Training complete in {CONFIG['total_training_time_sec']:.1f}s")
    print(f"  Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}")
    print(f"  Test Macro F1 (MLP): {test_f1:.4f}")
    print(f"  Test Macro F1 (LR baseline): {lr_f1:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()