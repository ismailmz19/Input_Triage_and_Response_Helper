# Inbox Triage and Response Helper

**INSE 6450: AI in Systems Engineering — Winter 2026**  
**Student:** Ismail Mzouri (40335670)  
**Instructor:** Prof. Ali Ayub  

An AI-powered email management system that classifies emails into 5 productivity categories and suggests response templates. Built on the Enron Email Corpus using weak supervision labeling, a PyTorch MLP classifier, drift monitoring, continual learning, and human-in-the-loop active learning.

**Key results:** Macro F1 = 0.749 | AUROC = 0.963 | p95 Latency = 2.635 ms | CL recovery +45% | AL labeling reduction 83–90%

---

## Categories

| Label | Description |
|---|---|
| **Urgent** | Time-sensitive, requires immediate action |
| **Needs Reply** | Requires a response but not time-critical |
| **Informational** | FYI, no action needed |
| **Scheduling** | Meeting or calendar related |
| **Spam / Low Priority** | Irrelevant or promotional |

---

## Setup

```bash
git clone https://github.com/ismailmz19/Input_Triage_and_Response_Helper.git
cd Input_Triage_and_Response_Helper
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

Download `emails.csv` from [Kaggle Enron Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset) and place it in `data/raw/`.

---

## Running the Full Pipeline

### Milestone 1 — Data Preparation
```bash
python src/data_loading.py
python src/preprocessing.py
python src/label_generator.py
python src/feature_extraction.py
```
Outputs: `data/processed/`, `models/tfidf_pipeline.joblib`

### Milestone 2 — Model Training and Evaluation
```bash
python src/train.py
python src/evaluate.py
python src/predict.py
```
Outputs: `models/mlp_model.pth`, `results/evaluation_report.csv`  
Key results: Macro F1 = 0.749, AUROC = 0.963, p95 Inference = 2.635 ms

### Milestone 3 — Robustness, Monitoring and Adaptation
```bash
python src/robustness.py
python src/stress_test.py
python src/drift_simulation.py
python src/monitoring.py
```
Outputs: `results/robustness_results.csv`, `results/drift_results.csv`, `results/monitoring_log.csv`

### Milestone 4 — Continual Learning and Human-in-the-Loop
```bash
python src/continual_learning.py
python src/active_learning.py
jupyter notebook notebooks/demo.ipynb
```
Outputs:
- `cl_results.csv`, `cl_results.png` — CL experiment results and trajectory plot
- `hitl_results.csv`, `hitl_results.png` — HITL simulation results
- `model_versions/model_v_*.pt` — versioned model snapshots

Key results: Drift detected at PSI = 7.71, F1 recovered 0.220 → 0.319 (+45%) in 0.136 s. Active learning reduces labeling burden by 83–90% per cycle.

---

## Project Structure
Input_Triage_and_Response_Helper/
├── src/
│   ├── data_loading.py          # M1: Parse Enron emails from CSV
│   ├── preprocessing.py         # M1: Clean text, remove HTML/signatures
│   ├── label_generator.py       # M1: Heuristic weak supervision labeling
│   ├── feature_extraction.py    # M1: TF-IDF + metadata + linguistic features
│   ├── train.py                 # M2: PyTorch MLP training
│   ├── evaluate.py              # M2: Evaluation metrics and plots
│   ├── predict.py               # M2: End-to-end inference + template suggestion
│   ├── response_templates.py    # M2: Keyword-ranked reply templates
│   ├── robustness.py            # M3: Adversarial and noise evaluation
│   ├── stress_test.py           # M3: Failure analysis and pre-flight checks
│   ├── drift_simulation.py      # M3: Three drift scenarios + head fine-tuning
│   ├── monitoring.py            # M3: PSI/KL divergence monitoring
│   ├── continual_learning.py    # M4: EWC continual learning + versioning
│   └── active_learning.py       # M4: Entropy-based AL + HITL simulation
├── notebooks/
│   └── demo.ipynb               # M4: End-to-end demo (load→infer→drift→HITL→update)
├── data/
│   ├── raw/                     # Place emails.csv here (gitignored)
│   └── processed/               # Pipeline outputs
├── models/                      # Serialised model artifacts
├── model_versions/              # Versioned model snapshots (M4)
├── results/                     # Output CSVs, plots, logs
├── config/
│   └── config.yaml
├── requirements.txt
└── README.md

---

## Output Locations

| Milestone | Artifact | Location |
|---|---|---|
| M1 | Feature pipeline | `models/tfidf_pipeline.joblib` |
| M2 | Trained MLP | `models/mlp_model.pth` |
| M2 | Evaluation report | `results/evaluation_report.csv` |
| M3 | Robustness results | `results/robustness_results.csv` |
| M3 | Monitoring logs | `results/monitoring_log.csv` |
| M4 | CL results + plot | `cl_results.csv`, `cl_results.png` |
| M4 | HITL results + plot | `hitl_results.csv`, `hitl_results.png` |
| M4 | Versioned snapshots | `model_versions/model_v_*.pt` |

---

## Notes

- No hardcoded absolute paths. All paths are relative to the repo root.
- All scripts reproducible with `RANDOM_SEED = 42`.
- Model versioning: each drift-triggered update saves a timestamped `.pt` snapshot under `model_versions/`