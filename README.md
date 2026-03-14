# Inbox Triage & Response Helper

**INSE 6450: AI in Systems Engineering — Winter 2026**
**Student:** Ismail Mzouri (40335670)
**Instructor:** Prof. Ali Ayub — Concordia University

An AI-powered email management system that classifies emails into 5 productivity
categories and suggests response templates. Built on the Enron Email Corpus using
a PyTorch MLP classifier with TF-IDF + metadata + linguistic features.

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
# Clone the repo
git clone https://github.com/ismailmz19/Input_Triage_and_Response_Helper.git
cd Input_Triage_and_Response_Helper

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies (includes PyTorch)
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

---

## Data

Download `emails.csv` from the [Kaggle Enron Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset) and place it in `data/raw/`.

---

## Running the Full Pipeline

### Milestone 1 — Data Preparation

```bash
# Step 1: Load and parse raw emails
python src/data_loading.py
# Output: data/processed/parsed_emails.csv

# Step 2: Clean and preprocess
python src/preprocessing.py
# Output: data/processed/cleaned_emails.csv

# Step 3: Generate labels using heuristic weak supervision
python src/label_generator.py
# Output: data/processed/labeled_emails.csv

# Step 4: Extract features (TF-IDF + metadata + linguistic)
python src/feature_extraction.py
# Output: data/processed/features.npz
#         data/processed/labels.csv
#         data/processed/feature_pipeline.joblib
```

### Milestone 2 — Model Training & Evaluation

```bash
# Step 5: Train PyTorch MLP classifier
python src/train.py
# Output: models/mlp_model.pth
#         models/mlp_model_config.json
#         models/label_encoder.joblib
#         models/logreg_baseline.joblib
#         results/learning_curves.png
#         results/confusion_matrix.png
#         results/classification_report.txt
#         results/ablation_comparison.json

# Step 6: Evaluate efficiency metrics
python src/evaluate.py
# Output: results/efficiency_metrics.json
#         results/roc_curves.png
#         results/pr_curves.png

# Step 7: End-to-end prediction with response templates
python src/predict.py
```

### Milestone 3 — Robustness, Monitoring & Adaptation

```bash
# Step 8: Robustness & adversarial evaluation
python src/robustness.py
# Output: results/robustness/robustness_curves.png
#         results/robustness/confidence_histograms.png
#         results/robustness/calibration_plot.png
#         results/robustness/latency_comparison.png
#         results/robustness/robustness_metrics.json
#         results/robustness/failure_examples.json

# Step 9: Stress tests & risk analysis
python src/stress_test.py
# Output: results/stress_test/stress_test_summary.png
#         results/stress_test/stress_test_metrics.json
#         results/stress_test/preflight_report.json

# Step 10: Drift simulation & model adaptation
python src/drift_simulation.py
# Output: results/drift/drift_comparison.png
#         results/drift/adaptation_history.png
#         results/drift/drift_metrics.json
#         models/mlp_model_adapted.pth
#         models/mlp_model_adapted_config.json

# Step 11: Monitoring dashboard
python src/monitoring.py
# Output: results/monitoring/monitoring_dashboard.png
#         results/monitoring/monitoring_metrics.json
#         results/monitoring/alert_log.json
```

---

## Project Structure

```
Input_Triage_and_Response_Helper/
├── README.md
├── requirements.txt
├── .gitignore
├── src/                            # All source code (12 files)
│   ├── data_loading.py             # Parse Enron emails from CSV
│   ├── preprocessing.py            # Clean text, remove HTML/signatures
│   ├── label_generator.py          # Heuristic weak supervision labeling
│   ├── feature_extraction.py       # TF-IDF + metadata + linguistic features
│   ├── train.py                    # PyTorch MLP training pipeline
│   ├── evaluate.py                 # Efficiency metrics & full evaluation
│   ├── response_templates.py       # 14 response templates + keyword ranking
│   ├── predict.py                  # End-to-end inference pipeline
│   ├── robustness.py               # Adversarial & noise robustness (M3)
│   ├── stress_test.py              # Stress tests & failure analysis (M3)
│   ├── drift_simulation.py         # Drift simulation & adaptation (M3)
│   └── monitoring.py               # Monitoring dashboard (M3)
├── notebooks/
│   └── 01_data_exploration.ipynb
├── data/
│   ├── raw/                        # Place emails.csv here (gitignored)
│   └── processed/                  # Pipeline outputs
│       ├── parsed_emails.csv
│       ├── cleaned_emails.csv
│       ├── labeled_emails.csv
│       ├── labels.csv
│       └── feature_pipeline.joblib # TF-IDF vocab + scalers
├── models/                         # Saved model artifacts
│   ├── mlp_model.pth               # Trained MLP weights
│   ├── mlp_model_config.json       # Hyperparameters + input_dim
│   ├── mlp_model_adapted.pth       # Head-finetuned model (M3)
│   ├── mlp_model_adapted_config.json
│   ├── label_encoder.joblib        # Class label mapping
│   └── logreg_baseline.joblib      # Logistic regression baseline
├── results/                        # All plots and metrics
│   ├── learning_curves.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── pr_curves.png
│   ├── classification_report.txt
│   ├── ablation_comparison.json
│   ├── efficiency_metrics.json
│   ├── robustness/                 # Milestone 3 robustness outputs
│   ├── stress_test/                # Milestone 3 stress test outputs
│   ├── drift/                      # Milestone 3 drift outputs
│   └── monitoring/                 # Milestone 3 monitoring outputs
└── config/
    └── config.yaml
```

---

## Model Summary

| Component | Choice | Rationale |
|---|---|---|
| **Classifier** | PyTorch MLP (512→256→128→5) | Lightweight, CPU-deployable, learns non-linear feature interactions |
| **Features** | TF-IDF (10k) + metadata (14) + linguistic (5) | Complementary signal families |
| **Labeling** | Weak supervision (heuristic rules) | No manual annotation needed at scale |
| **Baseline** | Logistic Regression | Ablation comparison |

## Key Results (Milestone 2)

| Metric | Value |
|---|---|
| Macro F1 | 0.7491 |
| AUROC | 0.9626 |
| p50 latency | 2.15 ms |
| p95 latency | 2.63 ms |
| Model size | 20.22 MB |

## Robustness Results (Milestone 3)

| Attack | F1 drop @ severity 0.5 |
|---|---|
| Synonym swap (grey-box) | −0.035 |
| Token dropout | −0.114 |
| Character noise (black-box) | −0.391 |

---

## Dependencies

- Python 3.10+
- PyTorch >= 2.0.0
- scikit-learn, NLTK, TextBlob, BeautifulSoup4
- pandas, numpy, matplotlib, scipy, joblib

## Tech Stack

| Component | Technology |
|---|---|
| ML Framework | PyTorch (MLP classifier) |
| Text Processing | NLTK, scikit-learn TF-IDF |
| Feature Engineering | scikit-learn, TextBlob |
| Deployment (M4) | Streamlit |