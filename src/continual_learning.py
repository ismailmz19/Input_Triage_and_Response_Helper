"""
continual_learning.py
Milestone 4 — Continual Learning Strategy
INSE 6450: AI in Systems Engineering
Student: Ismail Mzibri (40335670)

Strategy: Elastic Weight Consolidation (EWC) + drift-triggered head fine-tuning.
- Frozen TF-IDF/feature pipeline (M1/M2)
- Only MLP output head updated on drift
- EWC penalty prevents catastrophic forgetting
- Model versioning via joblib with timestamped snapshots
"""

import os
import copy
import time
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
from scipy.stats import entropy as kl_div

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("continual_learning.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_NAMES = ["Urgent", "Needs Reply", "Informational", "Scheduling", "Spam/Low Priority"]
NUM_CLASSES  = len(LABEL_NAMES)
RANDOM_SEED  = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Continual-learning hyperparameters
EWC_LAMBDA       = 400     # EWC regularisation strength
CL_LR            = 1e-3    # head fine-tune learning rate
CL_EPOCHS        = 10      # epochs per update
CL_BATCH_SIZE    = 64
PSI_THRESHOLD    = 0.2     # drift-trigger threshold (same as M3)
REPLAY_BUFFER_SZ = 500     # reservoir-replay buffer size
MODEL_VERSION_DIR = "model_versions"

os.makedirs(MODEL_VERSION_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# MLP definition (must match M2 architecture exactly)
# ---------------------------------------------------------------------------
class EmailMLP(nn.Module):
    """Three-layer MLP with BatchNorm and Dropout — identical to M2."""

    def __init__(self, input_dim: int, hidden_dims=(256, 128), num_classes=NUM_CLASSES,
                 dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def get_head(self):
        """Return the final classification layer."""
        return self.net[-1]

# ---------------------------------------------------------------------------
# EWC helper
# ---------------------------------------------------------------------------
class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017).
    Computes Fisher information on the 'anchor' dataset and penalises
    parameter drift during continual updates.
    """

    def __init__(self, model: nn.Module, dataloader: DataLoader, device: str = "cpu"):
        self.device   = device
        self.params   = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher   = self._compute_fisher(model, dataloader)

    def _compute_fisher(self, model, dataloader):
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        model.eval()
        criterion = nn.CrossEntropyLoss()
        for xb, yb in dataloader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            model.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
        n_batches = max(len(dataloader), 1)
        return {n: f / n_batches for n, f in fisher.items()}

    def penalty(self, model: nn.Module) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return loss

# ---------------------------------------------------------------------------
# Reservoir Replay Buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """
    Fixed-size reservoir buffer for experience replay.
    Randomly replaces old samples using reservoir sampling.
    """

    def __init__(self, max_size: int = REPLAY_BUFFER_SZ):
        self.max_size = max_size
        self.X: list = []
        self.y: list = []
        self._n_seen = 0

    def add(self, X: np.ndarray, y: np.ndarray):
        for xi, yi in zip(X, y):
            self._n_seen += 1
            if len(self.X) < self.max_size:
                self.X.append(xi)
                self.y.append(yi)
            else:
                j = np.random.randint(0, self._n_seen)
                if j < self.max_size:
                    self.X[j] = xi
                    self.y[j] = yi

    def sample(self, n: int):
        n = min(n, len(self.X))
        idx = np.random.choice(len(self.X), n, replace=False)
        return np.array([self.X[i] for i in idx]), np.array([self.y[i] for i in idx])

    def __len__(self):
        return len(self.X)

# ---------------------------------------------------------------------------
# Drift detection (PSI — consistent with M3 monitoring.py)
# ---------------------------------------------------------------------------
def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two feature distributions."""
    eps = 1e-8
    ref_counts, edges = np.histogram(reference, bins=bins)
    cur_counts, _     = np.histogram(current,   bins=edges)
    ref_pct = (ref_counts + eps) / (ref_counts.sum() + eps)
    cur_pct = (cur_counts + eps) / (cur_counts.sum() + eps)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

def drift_detected(X_ref: np.ndarray, X_new: np.ndarray,
                   threshold: float = PSI_THRESHOLD) -> bool:
    """Return True if PSI on the first principal component exceeds threshold."""
    psi = compute_psi(X_ref[:, 0], X_new[:, 0])
    logger.info(f"PSI = {psi:.4f}  (threshold={threshold})")
    return psi > threshold

# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------
def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray,
             device: str = "cpu") -> dict:
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(Xt)
        preds  = logits.argmax(dim=1).cpu().numpy()
    acc = float((preds == y).mean())
    # Per-class F1
    f1_per_class = []
    for c in range(NUM_CLASSES):
        tp = ((preds == c) & (y == c)).sum()
        fp = ((preds == c) & (y != c)).sum()
        fn = ((preds != c) & (y == c)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1_per_class.append(2 * prec * rec / (prec + rec + 1e-8))
    macro_f1 = float(np.mean(f1_per_class))
    return {"accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
            "f1_per_class": [round(f, 4) for f in f1_per_class]}

# ---------------------------------------------------------------------------
# Continual Learning Update (EWC + head fine-tuning)
# ---------------------------------------------------------------------------
def continual_update(
    model:       nn.Module,
    ewc:         EWC,
    X_new:       np.ndarray,
    y_new:       np.ndarray,
    replay_buf:  ReplayBuffer,
    device:      str = "cpu",
    version_tag: str = "",
) -> dict:
    """
    Fine-tune the model on X_new with EWC regularisation.
    Blends in replay samples to prevent catastrophic forgetting.
    Returns timing + memory metrics.
    """
    t0   = time.time()
    mem0 = _get_memory_mb()

    # ---- mix new data with replay ----
    if len(replay_buf) > 0:
        X_rep, y_rep = replay_buf.sample(min(CL_BATCH_SIZE, len(replay_buf)))
        X_train = np.vstack([X_new, X_rep])
        y_train = np.concatenate([y_new, y_rep])
    else:
        X_train, y_train = X_new, y_new

    # update replay buffer with new samples
    replay_buf.add(X_new, y_new)

    # ---- build DataLoader ----
    Xt = torch.tensor(X_train, dtype=torch.float32).to(device)
    yt = torch.tensor(y_train, dtype=torch.long).to(device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=CL_BATCH_SIZE, shuffle=True)

    # ---- only fine-tune: output head + last hidden layer ----
    for name, param in model.named_parameters():
        param.requires_grad = ("net.12" in name or "net.8" in name or
                               "net.9" in name or "net.11" in name or
                               "net.10" in name)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CL_LR)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(CL_EPOCHS):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits  = model(xb)
            ce_loss = criterion(logits, yb)
            ewc_pen = EWC_LAMBDA * ewc.penalty(model)
            loss    = ce_loss + ewc_pen
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            logger.info(f"  Epoch {epoch+1}/{CL_EPOCHS}  loss={epoch_loss/len(loader):.4f}")

    # re-enable all params
    for p in model.parameters():
        p.requires_grad = True

    update_time_s = time.time() - t0
    mem_used      = _get_memory_mb() - mem0

    # ---- save versioned snapshot ----
    tag  = version_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(MODEL_VERSION_DIR, f"model_v_{tag}.pt")
    torch.save(model.state_dict(), path)
    logger.info(f"Model snapshot saved → {path}")

    return {
        "update_time_s":  round(update_time_s, 3),
        "memory_delta_mb": round(mem_used, 2),
        "version_path":   path,
    }

# ---------------------------------------------------------------------------
# Drift-triggered continual learning pipeline
# ---------------------------------------------------------------------------
def run_continual_learning_pipeline(
    model:        nn.Module,
    pipeline,                         # sklearn pipeline from M1/M2
    X_ref:        np.ndarray,
    y_ref:        np.ndarray,
    drift_batches: list,               # list of (X_raw_batch, y_batch) tuples
    device:       str = "cpu",
) -> pd.DataFrame:
    """
    Main loop: for each incoming data batch —
      1. Extract features with M2 pipeline
      2. Check PSI drift
      3. If drift detected → run EWC continual update
      4. Log before/after metrics
    Returns a DataFrame of results across all time steps.
    """
    logger.info("=" * 60)
    logger.info("Starting Continual Learning Pipeline")
    logger.info("=" * 60)

    # Build EWC anchor on reference data
    Xr_t  = torch.tensor(X_ref, dtype=torch.float32)
    yr_t  = torch.tensor(y_ref, dtype=torch.long)
    ref_loader = DataLoader(TensorDataset(Xr_t, yr_t), batch_size=64)
    ewc         = EWC(model, ref_loader, device)
    replay_buf  = ReplayBuffer(REPLAY_BUFFER_SZ)
    replay_buf.add(X_ref[:200], y_ref[:200])   # seed buffer with reference data

    results = []
    X_ref_feat = X_ref  # reference features for PSI

    for step, (X_batch, y_batch) in enumerate(drift_batches):
        logger.info(f"\n--- Time step {step + 1} / {len(drift_batches)} ---")

        # Before metrics
        metrics_before = evaluate(model, X_batch, y_batch, device)
        logger.info(f"  Before update: {metrics_before}")

        # Drift check
        triggered = drift_detected(X_ref_feat, X_batch)
        psi_val   = compute_psi(X_ref_feat[:, 0], X_batch[:, 0])

        row = {
            "step":              step + 1,
            "drift_triggered":   triggered,
            "psi":               round(psi_val, 4),
            "acc_before":        metrics_before["accuracy"],
            "f1_before":         metrics_before["macro_f1"],
            "acc_after":         metrics_before["accuracy"],   # default: unchanged
            "f1_after":          metrics_before["macro_f1"],
            "update_time_s":     0.0,
            "memory_delta_mb":   0.0,
        }

        if triggered:
            logger.info("  ⚡ Drift detected → running continual update")
            perf = continual_update(
                model, ewc, X_batch, y_batch, replay_buf,
                device=device, version_tag=f"step{step+1}"
            )
            metrics_after = evaluate(model, X_batch, y_batch, device)
            logger.info(f"  After  update: {metrics_after}")

            row["acc_after"]       = metrics_after["accuracy"]
            row["f1_after"]        = metrics_after["macro_f1"]
            row["update_time_s"]   = perf["update_time_s"]
            row["memory_delta_mb"] = perf["memory_delta_mb"]

            # Update EWC anchor to include new data
            X_ref_feat = np.vstack([X_ref_feat[:300], X_batch])
            Xn_t = torch.tensor(X_batch, dtype=torch.float32)
            yn_t = torch.tensor(y_batch, dtype=torch.long)
            new_loader = DataLoader(TensorDataset(Xn_t, yn_t), batch_size=64)
            ewc = EWC(model, new_loader, device)
        else:
            logger.info("  ✓ No significant drift — model unchanged")

        results.append(row)

    df = pd.DataFrame(results)
    logger.info("\n" + df.to_string(index=False))
    return df

# ---------------------------------------------------------------------------
# Synthetic drift generation (mirrors drift_simulation.py from M3)
# ---------------------------------------------------------------------------
def generate_drift_batches(X_ref: np.ndarray, y_ref: np.ndarray,
                            n_steps: int = 5, batch_size: int = 200):
    """
    Create progressively drifting batches.
    Steps 1-2: mild drift (noise σ=0.3)
    Steps 3-4: moderate drift (σ=0.7)
    Step 5: severe drift (σ=1.2) + mean shift
    """
    rng     = np.random.default_rng(RANDOM_SEED)
    batches = []
    for i in range(n_steps):
        idx   = rng.choice(len(X_ref), batch_size, replace=True)
        X_b   = X_ref[idx].copy()
        y_b   = y_ref[idx].copy()
        sigma = [0.3, 0.3, 0.7, 0.7, 1.2][i]
        noise = rng.normal(0, sigma, X_b.shape)
        X_b  += noise
        if i == 4:                        # severe: add mean shift
            X_b[:, :50] += 2.5
        batches.append((X_b, y_b))
    return batches

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _get_memory_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0

def load_model_and_pipeline(model_path: str, pipeline_path: str,
                             input_dim: int = 1000):
    """Load serialised M2 MLP + sklearn feature pipeline."""
    pipeline = joblib.load(pipeline_path)
    model    = EmailMLP(input_dim=input_dim)
    state    = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    logger.info(f"Loaded model from {model_path}")
    return model, pipeline

def plot_results(df: pd.DataFrame, output_path: str = "cl_results.png"):
    """Plot F1 and accuracy trajectories across time steps."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        steps = df["step"].values

        # F1
        axes[0].plot(steps, df["f1_before"], "o--", label="Before update", color="#e07b39")
        axes[0].plot(steps, df["f1_after"],  "s-",  label="After update",  color="#3b82f6")
        for s, trig in zip(steps, df["drift_triggered"]):
            if trig:
                axes[0].axvline(s, color="red", alpha=0.3, linestyle=":")
        axes[0].set_title("Macro F1 Across Time Steps")
        axes[0].set_xlabel("Time Step")
        axes[0].set_ylabel("Macro F1")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)

        # Accuracy
        axes[1].plot(steps, df["acc_before"], "o--", label="Before update", color="#e07b39")
        axes[1].plot(steps, df["acc_after"],  "s-",  label="After update",  color="#3b82f6")
        for s, trig in zip(steps, df["drift_triggered"]):
            if trig:
                axes[1].axvline(s, color="red", alpha=0.3, linestyle=":",
                                label="Drift triggered" if s == df.loc[df["drift_triggered"], "step"].iloc[0] else "")
        axes[1].set_title("Accuracy Across Time Steps")
        axes[1].set_xlabel("Time Step")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot saved → {output_path}")
    except Exception as e:
        logger.warning(f"Plotting skipped: {e}")

# ---------------------------------------------------------------------------
# Main — self-contained demo run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Continual Learning — self-contained demo (synthetic data)")

    # --- Synthesise reference data mimicking M2 feature space (1000-dim TF-IDF) ---
    rng       = np.random.default_rng(RANDOM_SEED)
    INPUT_DIM = 1000
    N_REF     = 1000

    X_ref = rng.random((N_REF, INPUT_DIM)).astype(np.float32)
    y_ref = rng.integers(0, NUM_CLASSES, N_REF)

    # --- Build and pre-train model (proxy for loaded M2 weights) ---
    model     = EmailMLP(input_dim=INPUT_DIM)
    criterion = nn.CrossEntropyLoss()
    opt       = optim.Adam(model.parameters(), lr=1e-3)
    Xt_ref    = torch.tensor(X_ref, dtype=torch.float32)
    yt_ref    = torch.tensor(y_ref, dtype=torch.long)
    loader    = DataLoader(TensorDataset(Xt_ref, yt_ref), batch_size=64, shuffle=True)

    logger.info("Pre-training proxy model on reference data ...")
    for _ in range(20):
        for xb, yb in loader:
            opt.zero_grad()
            nn.CrossEntropyLoss()(model(xb), yb).backward()
            opt.step()

    # Save initial model version
    init_path = os.path.join(MODEL_VERSION_DIR, "model_v_initial.pt")
    torch.save(model.state_dict(), init_path)
    logger.info(f"Initial model saved → {init_path}")

    # --- Generate drift batches ---
    drift_batches = generate_drift_batches(X_ref, y_ref, n_steps=5, batch_size=200)

    # --- Run continual learning pipeline ---
    results_df = run_continual_learning_pipeline(
        model=model,
        pipeline=None,        # no sklearn pipeline in demo
        X_ref=X_ref,
        y_ref=y_ref,
        drift_batches=drift_batches,
        device="cpu",
    )

    # --- Save results & plot ---
    results_df.to_csv("cl_results.csv", index=False)
    plot_results(results_df, "cl_results.png")

    logger.info("\n=== FINAL RESULTS TABLE ===")
    logger.info("\n" + results_df[["step","drift_triggered","psi","f1_before",
                                   "f1_after","update_time_s"]].to_string(index=False))

    # Inference latency measurement
    model.eval()
    sample = torch.tensor(X_ref[:1], dtype=torch.float32)
    times  = []
    for _ in range(500):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(sample)
        times.append((time.perf_counter() - t0) * 1000)
    logger.info(f"\nInference latency: mean={np.mean(times):.3f}ms  "
                f"p95={np.percentile(times, 95):.3f}ms")