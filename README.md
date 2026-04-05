# Inbox Triage and Response Helper

**INSE 6450: AI in Systems Engineering — Concordia University, Winter 2026**
**Student:** Ismail Mzouri | **ID:** 40335670

An end-to-end AI pipeline that classifies emails into five categories and suggests response templates, built on the Enron Email Corpus using weak supervision, an MLP classifier, drift monitoring, and continual learning with human-in-the-loop adaptation.

**Labels:** Urgent · Needs Reply · Informational · Scheduling · Spam/Low Priority

---

## Project Structure

```
Input_Triage_and_Response_Helper/
├── src/
│   ├── data_loading.py          # M1: Enron corpus loading
│   ├── preprocessing.py         # M1: Email cleaning and normalisation
│   ├── label_generator.py       # M1: Snorkel weak-supervision labeling
│   ├── feature_extraction.py    # M1: TF-IDF feature pipeline
│   ├── train.py                 # M2: MLP training
│   ├── evaluate.py              # M2: Evaluation metrics
│   ├── predict.py               # M2: Inference + response templates
│   ├── response_templates.py    # M2: Template suggestion module
│   ├── robustness.py            # M3: Adversarial robustness tests
│   ├── stress_test.py           # M3: Stress and perturbation tests
│   ├── drift_simulation.py      # M3: Distribution drift simulation
│   ├── monitoring.py            # M3: PSI/KL divergence monitoring
│   ├── continual_learning.py    # M4: EWC continual learning + versioning
│   └── active_learning.py       # M4: Entropy-based AL + HITL simulation
├── notebooks/
│   └── demo.ipynb               # M4: End-to-end demo notebook
├── data/                        # Enron corpus (not tracked by git)
├── models/                      # Serialised model artifacts
├── model_versions/              # Versioned model snapshots (M4)
├── results/                     # Output CSVs, plots, logs
├── config/                      # Configuration files
└── README.md
```

---

## Dependencies

```bash
pip install torch numpy pandas scikit-learn scipy matplotlib joblib psutil snorkel nltk
```

Python 3.10+ recommended.

---

## Run Commands

### Milestone 1 — Data Preparation

```bash
python src/data_loading.py        # Load and parse Enron corpus
python src/preprocessing.py       # Clean email text
python src/label_generator.py     # Generate weak-supervision labels (Snorkel)
python src/feature_extraction.py  # Build TF-IDF feature pipeline
```

**Outputs:** `data/processed/`, `models/tfidf_pipeline.joblib`

---

### Milestone 2 — Model Training and Evaluation

```bash
python src/train.py               # Train MLP classifier
python src/evaluate.py            # Evaluate on test set (F1, AUROC, latency)
python src/predict.py             # Run inference + response template suggestion
```

**Outputs:** `models/mlp_model.pt`, `results/evaluation_report.csv`

**Key results:** Macro F1 = 0.749 · AUROC = 0.963 · Inference = 2.1ms · Throughput = 46,190 emails/s

---

### Milestone 3 — Robustness, Monitoring, and Adaptation

```bash
python src/robustness.py          # Synonym, character noise, word insertion attacks
python src/stress_test.py         # Load and boundary stress tests
python src/drift_simulation.py    # Simulate temporal distribution drift
python src/monitoring.py          # PSI + KL divergence drift monitoring
```

**Outputs:** `results/robustness_results.csv`, `results/drift_results.csv`, `results/monitoring_log.csv`

**Key results:** F1 under synonym attack = 0.712 · Character noise = 0.431 · PSI threshold = 0.2

---

### Milestone 4 — Continual Learning and Human-in-the-Loop

```bash
# Continual learning: EWC + drift-triggered head fine-tuning
python src/continual_learning.py
# Outputs: cl_results.csv, cl_results.png, model_versions/model_v_*.pt

# Active learning + HITL simulation: entropy sampling + oracle annotation
python src/active_learning.py
# Outputs: hitl_results.csv, hitl_results.png, model_versions/model_v_al_*.pt

# End-to-end demo: load -> infer -> detect drift -> query human -> update model
jupyter notebook notebooks/demo.ipynb
```

**Outputs:** `cl_results.csv` · `cl_results.png` · `hitl_results.csv` · `hitl_results.png` · `model_versions/`

**Key results:**
- Drift detected at PSI = 7.71 (threshold 0.2); F1 recovered from 0.220 → 0.319 (+45%) in 0.136s
- Active learning reduces labeling burden by 83–90% per cycle (50 queries from 500-sample pool)
- HITL F1 trend: 0.239 → 0.260 (+9%) across 5 cycles; inference latency unchanged at 0.19ms

---

## Output Locations

| Milestone | Artifact | Location |
|---|---|---|
| M1 | Feature pipeline | `models/tfidf_pipeline.joblib` |
| M2 | Trained MLP | `models/mlp_model.pt` |
| M2 | Evaluation report | `results/evaluation_report.csv` |
| M3 | Robustness results | `results/robustness_results.csv` |
| M3 | Drift/monitoring logs | `results/drift_results.csv`, `results/monitoring_log.csv` |
| M4 | CL results + plot | `cl_results.csv`, `cl_results.png` |
| M4 | HITL results + plot | `hitl_results.csv`, `hitl_results.png` |
| M4 | Versioned snapshots | `model_versions/model_v_*.pt` |

---

## Notes

- No hardcoded absolute paths — all paths are relative to the repo root.
- All scripts are reproducible with `RANDOM_SEED = 42`.
- M4 scripts include self-contained synthetic demos in their `__main__` blocks; connect to `models/mlp_model.pt` and `models/tfidf_pipeline.joblib` for real Enron data runs.
- Model versioning: each drift-triggered update saves a timestamped `.pt` snapshot under `model_versions/` and can be rolled back by loading the previous snapshot.
