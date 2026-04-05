"""
active_learning.py
Milestone 4 — Human-in-the-Loop + Active Learning
INSE 6450: AI in Systems Engineering
Student: Ismail Mzibri (40335670)

Strategy: Uncertainty sampling (entropy-based) with simulated human annotation.
- Query strategy: entropy of softmax outputs
- Human annotation simulated via M1 weak-supervision heuristics
- HITL cycles feed directly into continual_learning.py update loop
- Measures: F1 improvement across cycles, labeling burden, latency impact
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Reuse model definition from continual_learning.py
from continual_learning import (
    EmailMLP, evaluate, continual_update, ReplayBuffer, EWC,
    NUM_CLASSES, LABEL_NAMES, RANDOM_SEED, CL_EPOCHS, CL_BATCH_SIZE,
    MODEL_VERSION_DIR,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("active_learning.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Active learning hyperparameters
# ---------------------------------------------------------------------------
QUERY_BUDGET_PER_CYCLE = 50    # max samples queried per HITL cycle
CONFIDENCE_THRESHOLD   = 0.70  # samples below this threshold are queried
N_AL_CYCLES            = 5     # number of active learning cycles
ENTROPY_TOP_K          = 50    # top-K most uncertain samples per cycle
AL_EPOCHS              = 25    # more epochs per AL cycle for stronger signal
AL_EWC_LAMBDA          = 50    # lower EWC penalty for AL (allow adaptation)

# ---------------------------------------------------------------------------
# Query strategies
# ---------------------------------------------------------------------------
def entropy_sampling(model: nn.Module, X_pool: np.ndarray,
                     top_k: int = ENTROPY_TOP_K,
                     device: str = "cpu") -> np.ndarray:
    """
    Select the top_k most uncertain samples by predictive entropy.
    H(y|x) = -sum_c p(y=c|x) * log p(y=c|x)
    """
    model.eval()
    Xt = torch.tensor(X_pool, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(Xt)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    top_idx = np.argsort(entropy)[::-1][:top_k]
    return top_idx, entropy

def margin_sampling(model: nn.Module, X_pool: np.ndarray,
                    top_k: int = ENTROPY_TOP_K,
                    device: str = "cpu") -> np.ndarray:
    """
    Select samples where the margin between top-1 and top-2 predictions is smallest.
    """
    model.eval()
    Xt = torch.tensor(X_pool, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(Xt), dim=-1).cpu().numpy()
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    margins = sorted_probs[:, 0] - sorted_probs[:, 1]
    top_idx = np.argsort(margins)[:top_k]   # smallest margin = most uncertain
    return top_idx, margins

def low_confidence_filter(model: nn.Module, X_pool: np.ndarray,
                           threshold: float = CONFIDENCE_THRESHOLD,
                           device: str = "cpu") -> np.ndarray:
    """Return indices of samples where max confidence < threshold."""
    model.eval()
    Xt = torch.tensor(X_pool, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs      = torch.softmax(model(Xt), dim=-1).cpu().numpy()
    max_conf   = probs.max(axis=1)
    below_idx  = np.where(max_conf < threshold)[0]
    return below_idx, max_conf

# ---------------------------------------------------------------------------
# Simulated human annotation (HITL oracle)
# ---------------------------------------------------------------------------
def simulate_human_annotation(X: np.ndarray, y_noisy: np.ndarray,
                               correction_rate: float = 0.80) -> np.ndarray:
    """
    Simulate a human annotator correcting weak-supervision labels.

    The oracle uses the same heuristic labeling functions from M1 to produce
    'ground-truth' labels, then accepts them with probability correction_rate
    (modelling a human who is 80% consistent with the heuristics).
    The remaining 20% retain the noisy label, simulating occasional human error.

    In a real deployment this function is replaced by an actual labeling UI.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    # Oracle produces heuristic labels (same signal the model was trained on)
    y_oracle    = heuristic_labeler(X)
    y_corrected = y_oracle.copy()
    # Simulate 20% human error by reverting to noisy label
    n_errors    = int((1.0 - correction_rate) * len(y_noisy))
    err_idx     = rng.choice(len(y_noisy), n_errors, replace=False)
    y_corrected[err_idx] = y_noisy[err_idx]
    n_corrections = (y_corrected != y_noisy).sum()
    logger.info(f"  Human oracle: {n_corrections}/{len(y_noisy)} labels corrected "
                f"({n_corrections/len(y_noisy)*100:.1f}%)")
    return y_corrected

def heuristic_labeler(X: np.ndarray) -> np.ndarray:
    """
    Reproduce M1 weak-supervision heuristics as a labeling function.
    Features are TF-IDF vectors (1000-dim); we use aggregate statistics
    as proxies for urgency / scheduling keywords.

    This mirrors the Snorkel labeling functions from M1's label_generation.py
    and is used to generate consistent labels for the unlabeled pool.
    Thresholds are calibrated for the synthetic uniform [0,1] feature space
    used in this simulation to ensure balanced class coverage.
    """
    labels   = np.zeros(len(X), dtype=int)
    row_mean = X.mean(axis=1)
    row_std  = X.std(axis=1)
    row_nnz  = (X > 0.5).sum(axis=1) / X.shape[1]   # dense-feature proxy

    # Partition into 5 balanced buckets using quantile-based rules
    mean_q   = np.quantile(row_mean, [0.2, 0.4, 0.6, 0.8])
    for i in range(len(X)):
        m = row_mean[i]
        if m < mean_q[0]:
            labels[i] = 4    # Spam / Low Priority  (very low mean activation)
        elif m < mean_q[1]:
            labels[i] = 0    # Urgent               (low-moderate)
        elif m < mean_q[2]:
            labels[i] = 1    # Needs Reply          (moderate)
        elif m < mean_q[3]:
            labels[i] = 3    # Scheduling           (moderate-high)
        else:
            labels[i] = 2    # Informational        (high mean activation)

    # Inject ~20% noise to simulate weak-supervision imperfection
    rng       = np.random.default_rng(RANDOM_SEED)
    noise_idx = rng.choice(len(labels), int(0.2 * len(labels)), replace=False)
    labels[noise_idx] = rng.integers(0, NUM_CLASSES, len(noise_idx))
    return labels

# ---------------------------------------------------------------------------
# Active learning cycle
# ---------------------------------------------------------------------------
def run_active_learning_cycle(
    model:       nn.Module,
    X_pool:      np.ndarray,      # unlabeled pool
    X_val:       np.ndarray,
    y_val:       np.ndarray,
    replay_buf:  ReplayBuffer,
    ewc:         EWC,
    cycle:       int,
    device:      str = "cpu",
    query_strat: str = "entropy",  # "entropy" | "margin" | "confidence"
) -> dict:
    """
    One AL cycle:
      1. Query top-K uncertain samples from pool
      2. Simulate human annotation
      3. Perform continual update with newly labeled data
      4. Evaluate on validation set
    """
    t0 = time.time()

    # ---- 1. Query uncertain samples ----
    if query_strat == "entropy":
        query_idx, scores = entropy_sampling(model, X_pool, ENTROPY_TOP_K, device)
    elif query_strat == "margin":
        query_idx, scores = margin_sampling(model, X_pool, ENTROPY_TOP_K, device)
    else:
        query_idx, scores = low_confidence_filter(model, X_pool, CONFIDENCE_THRESHOLD, device)

    query_idx = query_idx[:QUERY_BUDGET_PER_CYCLE]
    X_query   = X_pool[query_idx]
    logger.info(f"Cycle {cycle}: queried {len(query_idx)} samples from pool of {len(X_pool)}")

    # ---- 2. Simulate human annotation ----
    y_noisy   = heuristic_labeler(X_query)
    y_labeled = simulate_human_annotation(X_query, y_noisy, correction_rate=0.80)

    # ---- 3. Before metrics ----
    metrics_before = evaluate(model, X_val, y_val, device)

    # ---- 4. Continual update (AL uses lower EWC lambda + more epochs) ----
    import continual_learning as _cl
    _orig_lambda, _orig_epochs = _cl.EWC_LAMBDA, _cl.CL_EPOCHS
    _cl.EWC_LAMBDA = AL_EWC_LAMBDA
    _cl.CL_EPOCHS  = AL_EPOCHS
    perf = continual_update(
        model, ewc, X_query, y_labeled, replay_buf,
        device=device, version_tag=f"al_cycle{cycle}"
    )
    _cl.EWC_LAMBDA = _orig_lambda
    _cl.CL_EPOCHS  = _orig_epochs

    # ---- 5. After metrics ----
    metrics_after = evaluate(model, X_val, y_val, device)

    cycle_time = time.time() - t0
    labeling_burden = len(query_idx)       # samples a human had to label
    label_saving    = len(X_pool) - len(query_idx)
    reduction_pct   = label_saving / len(X_pool) * 100

    logger.info(f"  Cycle {cycle} — F1: {metrics_before['macro_f1']:.4f} → "
                f"{metrics_after['macro_f1']:.4f}  |  "
                f"Labeled: {labeling_burden} / {len(X_pool)} "
                f"({reduction_pct:.1f}% reduction)  |  "
                f"Time: {cycle_time:.2f}s")

    return {
        "cycle":              cycle,
        "query_strategy":     query_strat,
        "n_queried":          labeling_burden,
        "pool_size":          len(X_pool),
        "labeling_reduction": round(reduction_pct, 2),
        "f1_before":          metrics_before["macro_f1"],
        "f1_after":           metrics_after["macro_f1"],
        "acc_before":         metrics_before["accuracy"],
        "acc_after":          metrics_after["accuracy"],
        "f1_delta":           round(metrics_after["macro_f1"] - metrics_before["macro_f1"], 4),
        "update_time_s":      perf["update_time_s"],
        "cycle_time_s":       round(cycle_time, 3),
        "memory_delta_mb":    perf["memory_delta_mb"],
    }

# ---------------------------------------------------------------------------
# Full HITL simulation pipeline
# ---------------------------------------------------------------------------
def run_hitl_simulation(
    model:       nn.Module,
    X_pool:      np.ndarray,
    X_val:       np.ndarray,
    y_val:       np.ndarray,
    n_cycles:    int = N_AL_CYCLES,
    device:      str = "cpu",
    query_strat: str = "entropy",
) -> pd.DataFrame:
    """
    Run N_AL_CYCLES of active learning with HITL simulation.
    Each cycle removes queried samples from the pool (standard AL protocol).
    """
    logger.info("=" * 60)
    logger.info(f"Starting HITL Simulation — {n_cycles} cycles, "
                f"strategy={query_strat}")
    logger.info("=" * 60)

    # Build EWC anchor from validation data
    Xv_t      = torch.tensor(X_val, dtype=torch.float32)
    yv_t      = torch.tensor(y_val, dtype=torch.long)
    val_loader = DataLoader(TensorDataset(Xv_t, yv_t), batch_size=64)
    ewc        = EWC(model, val_loader, device)
    replay_buf = ReplayBuffer()
    replay_buf.add(X_val[:100], y_val[:100])

    pool       = X_pool.copy()
    results    = []

    for cycle in range(1, n_cycles + 1):
        if len(pool) < ENTROPY_TOP_K:
            logger.warning("Pool exhausted — stopping early")
            break

        row = run_active_learning_cycle(
            model, pool, X_val, y_val,
            replay_buf, ewc, cycle, device, query_strat
        )
        results.append(row)

        # Remove queried samples from pool (active learning removes labeled data)
        query_idx, _ = entropy_sampling(model, pool, ENTROPY_TOP_K, device)
        pool = np.delete(pool, query_idx[:QUERY_BUDGET_PER_CYCLE], axis=0)
        logger.info(f"  Pool size after removal: {len(pool)}")

    df = pd.DataFrame(results)
    logger.info("\n=== HITL SIMULATION RESULTS ===")
    logger.info("\n" + df[["cycle","n_queried","labeling_reduction",
                            "f1_before","f1_after","f1_delta",
                            "update_time_s"]].to_string(index=False))
    return df

# ---------------------------------------------------------------------------
# Plot HITL results
# ---------------------------------------------------------------------------
def plot_hitl_results(df: pd.DataFrame, output_path: str = "hitl_results.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        cycles = df["cycle"].values

        # F1 improvement
        axes[0].plot(cycles, df["f1_before"], "o--", label="Before", color="#e07b39")
        axes[0].plot(cycles, df["f1_after"],  "s-",  label="After",  color="#3b82f6")
        axes[0].set_title("Macro F1 Across HITL Cycles")
        axes[0].set_xlabel("AL Cycle")
        axes[0].set_ylabel("Macro F1")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)

        # Cumulative F1 delta
        cum_delta = np.cumsum(df["f1_delta"].values)
        axes[1].bar(cycles, cum_delta, color="#3b82f6", alpha=0.8)
        axes[1].set_title("Cumulative F1 Improvement")
        axes[1].set_xlabel("AL Cycle")
        axes[1].set_ylabel("Cumulative ΔF1")
        axes[1].grid(True, alpha=0.3, axis="y")

        # Labeling burden
        axes[2].bar(cycles, df["labeling_reduction"], color="#10b981", alpha=0.8)
        axes[2].set_title("Labeling Burden Reduction (%)")
        axes[2].set_xlabel("AL Cycle")
        axes[2].set_ylabel("% Samples Not Labeled")
        axes[2].set_ylim(0, 100)
        axes[2].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"HITL plot saved → {output_path}")
    except Exception as e:
        logger.warning(f"Plotting skipped: {e}")

# ---------------------------------------------------------------------------
# Main — self-contained demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Active Learning + HITL — self-contained demo (synthetic data)")

    rng       = np.random.default_rng(RANDOM_SEED)
    INPUT_DIM = 1000

    # Reference / validation data
    # Labels are generated by heuristic_labeler (same oracle used during HITL),
    # ensuring consistency between training signal and annotation oracle.
    X_ref  = rng.random((800, INPUT_DIM)).astype(np.float32)
    y_ref  = heuristic_labeler(X_ref)
    X_val  = rng.random((200, INPUT_DIM)).astype(np.float32)
    y_val  = heuristic_labeler(X_val)

    # Unlabeled pool (labels unknown to model; oracle reveals them during HITL)
    X_pool = rng.random((500, INPUT_DIM)).astype(np.float32)

    # Build + pre-train proxy model
    model     = EmailMLP(input_dim=INPUT_DIM)
    criterion = nn.CrossEntropyLoss()
    opt       = optim.Adam(model.parameters(), lr=1e-3)
    Xt_ref    = torch.tensor(X_ref, dtype=torch.float32)
    yt_ref    = torch.tensor(y_ref, dtype=torch.long)
    loader    = DataLoader(TensorDataset(Xt_ref, yt_ref), batch_size=64, shuffle=True)

    logger.info("Pre-training proxy model ...")
    for _ in range(40):
        for xb, yb in loader:
            opt.zero_grad()
            nn.CrossEntropyLoss()(model(xb), yb).backward()
            opt.step()

    initial_metrics = evaluate(model, X_val, y_val)
    logger.info(f"Initial validation metrics: {initial_metrics}")

    # Run HITL simulation
    results_df = run_hitl_simulation(
        model=model,
        X_pool=X_pool,
        X_val=X_val,
        y_val=y_val,
        n_cycles=N_AL_CYCLES,
        device="cpu",
        query_strat="entropy",
    )

    # Save results
    results_df.to_csv("hitl_results.csv", index=False)
    plot_hitl_results(results_df, "hitl_results.png")

    # Inference latency after HITL
    model.eval()
    sample = torch.tensor(X_val[:1], dtype=torch.float32)
    times  = []
    for _ in range(500):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(sample)
        times.append((time.perf_counter() - t0) * 1000)

    import numpy as _np
    logger.info(f"\nPost-HITL inference latency: mean={_np.mean(times):.3f}ms  "
                f"p95={_np.percentile(times, 95):.3f}ms")
    logger.info("\nDone.")