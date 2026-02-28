# Inbox Triage & Response Helper

**INSE 6450: AI in Systems Engineering — Winter 2026**
**Student:** Ismail Mzouri (40335670)
**Instructor:** Prof. Ali Ayub — Concordia University

An AI-powered email management system with two core features:
1. **Email classification** — automatically categorizes emails into 5 productivity labels
2. **Response template suggestion** — recommends ranked reply templates based on the predicted category

## Email Categories

| Label | Description |
|---|---|
| Urgent | Time-sensitive, requires immediate action |
| Needs Reply | Requires a response, not time-critical |
| Informational | FYI only, no action needed |
| Scheduling | Meeting or calendar related |
| Spam / Low Priority | Irrelevant or promotional |

## Milestone 2 Results

| Metric | Value |
|---|---|
| Test Accuracy | 76.84% |
| Macro F1 | 0.7491 |
| AUROC | 0.9626 |
| PR-AUC | 0.8631 |
| p95 Latency | 2.635 ms |
| Throughput | 10,931 samples/sec |
| Peak Training RAM | 343.44 MB |
| Model Size | 20.22 MB |
| Parameters | 5,296,901 |
| Training Time | ~19.5 min (CPU) |

## Project Structure

src/ contains 8 files:
- data_loading.py, preprocessing.py, label_generator.py, feature_extraction.py (Milestone 1)
- train.py, evaluate.py, response_templates.py, predict.py (Milestone 2)

## Setup

git clone https://github.com/ismailmz19/Input_Triage_and_Response_Helper.git
pip install -r requirements.txt

## Pipeline

python src/data_loading.py
python src/preprocessing.py
python src/label_generator.py
python src/feature_extraction.py
python src/train.py
python src/evaluate.py
python src/predict.py
