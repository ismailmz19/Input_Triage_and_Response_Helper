# Inbox Triage and Response Helper

**INSE 6450: AI in Systems Engineering — Concordia University, Winter 2026**
**Student:** Ismail Mzouri | **ID:** 40335670

An end-to-end AI pipeline that classifies emails into five categories and suggests response templates, built on the Enron Email Corpus using weak supervision, an MLP classifier, drift monitoring, and continual learning with human-in-the-loop adaptation.

**Labels:** Urgent · Needs Reply · Informational · Scheduling · Spam/Low Priority

---

## Project Structure

Input_Triage_and_Response_Helper/
├── src/
│   ├── data_loading.py          # M1: Enron corpus loading
│   ├── preprocessing.py         # M1: Email cleaning
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
├── data/                        # Enron corpus (not tracked)
├── models/                      # Serialised model artifacts
├── model_versions/              # Versioned model snapshots (M4)
├── results/                     # Output CSVs, plots, logs
└── README.md

---

## Dependencies

pip install torch numpy pandas scikit-learn scipy matplotlib joblib psutil snorkel nltk

---

## Running the Full Pipeline

### Milestone 1 — Data Preparation
python src/data_loading.py
python src/preprocessing.py
python src/label_generator.py
python src/feature_extraction.py

Outputs: data/processed/, models/tfidf_pipeline.joblib

### Milestone 2 — Model Training and Evaluation
python src/train.py
python src/evaluate.py
python src/predict.py

Outputs: models/mlp_model.pt, results/evaluation_report.csv
Key results: Macro F1 = 0.749, AUROC = 0.963, Inference = 2.1ms

### Milestone 3 — Robustness, Monitoring, and Adaptation
python src/robustness.py
python src/stress_test.py
python src/drift_simulation.py
python src/monitoring.py

Outputs: results/robustness_results.csv, results/drift_results.csv, results/monitoring_log.csv

### Milestone 4 — Continual Learning and Human-in-the-Loop
python src/continual_learning.py
python src/active_learning.py
jupyter notebook notebooks/demo.ipynb

Outputs:
- cl_results.csv, cl_results.png        (CL experiment results and trajectory plot)
- hitl_results.csv, hitl_results.png    (HITL simulation results)
- model_versions/model_v_*.pt           (versioned model snapshots)

Key results: Drift detected at PSI=7.71, F1 recovered 0.220 to 0.319 (+45%) in 0.136s.
Active learning reduces labeling burden by 83-90% per cycle.

---

## Output Locations

| Milestone | Artifact | Location |
|---|---|---|
| M1 | Feature pipeline | models/tfidf_pipeline.joblib |
| M2 | Trained MLP | models/mlp_model.pt |
| M2 | Evaluation report | results/evaluation_report.csv |
| M3 | Robustness results | results/robustness_results.csv |
| M3 | Monitoring logs | results/monitoring_log.csv |
| M4 | CL results + plot | cl_results.csv, cl_results.png |
| M4 | HITL results + plot | hitl_results.csv, hitl_results.png |
| M4 | Versioned snapshots | model_versions/model_v_*.pt |

---

## Notes
- No hardcoded absolute paths. All paths are relative to the repo root.
- All scripts reproducible with RANDOM_SEED = 42.
- Model versioning: each drift-triggered update saves a timestamped .pt snapshot under model_versions/.