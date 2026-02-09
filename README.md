# Input Triage and Response Helper

**INSE 6450: AI in Systems Engineering — Winter 2026**  
**Student:** Ismail Mzouri (40335670)

An AI-powered email management system that classifies emails into 5 productivity categories and suggests response templates.

## Categories

| Label | Description |
|-------|-------------|
| **Urgent** | Time-sensitive, requires immediate action |
| **Needs Reply** | Requires a response but not time-critical |
| **Informational** | FYI, no action needed |
| **Scheduling** | Meeting or calendar related |
| **Spam / Low Priority** | Irrelevant or promotional |

## Setup

```bash
# Clone the repo
git clone https://github.com/ismailmz19/Input_Triage_and_Response_Helper.git
cd Input_Triage_and_Response_Helper

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

## Data

Download `emails.csv` from [Kaggle Enron Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset) and place it in `data/raw/`.

## Running the Pipeline

```bash
# Step 1: Load and parse raw emails
python src/data_loading.py
# Output: data/processed/parsed_emails.csv

# Step 2: Clean and preprocess
python src/preprocessing.py
# Output: data/processed/cleaned_emails.csv

# Step 3: Generate labels using heuristic rules
python src/label_generator.py
# Output: data/processed/labeled_emails.csv

# Step 4: Extract features
python src/feature_extraction.py
# Output: data/processed/features.npz, data/processed/labels.csv
```

## Outputs

All saved output artifacts are in `data/processed/`:
- `parsed_emails.csv` — Structured emails parsed from raw data
- `cleaned_emails.csv` — Cleaned and deduplicated emails
- `labeled_emails.csv` — Emails with heuristic category labels
- `features.npz` — Sparse feature matrix (TF-IDF + metadata + linguistic)
- `labels.csv` — Label array for classification
- `feature_pipeline.joblib` — Serialized feature extraction pipeline

## Project Structure

```
Input_Triage_and_Response_Helper/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── data_loading.py         # Parse Enron emails from CSV
│   ├── preprocessing.py        # Clean text, remove HTML/signatures
│   ├── label_generator.py      # Heuristic rule-based labeling
│   └── feature_extraction.py   # TF-IDF + metadata + linguistic features
├── notebooks/
│   └── 01_data_exploration.ipynb
├── data/
│   ├── raw/                    # Place emails.csv here (gitignored)
│   └── processed/              # Pipeline outputs saved here
├── models/                     # Saved model artifacts (Milestone 2)
└── config/
    └── config.yaml
```

## Dependencies

- Python 3.10+
- scikit-learn, NLTK, TextBlob, BeautifulSoup4, pandas, numpy, matplotlib

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Framework | scikit-learn (+ PyTorch in Milestone 2) |
| Text Processing | NLTK, BeautifulSoup |
| Web App | Streamlit (Milestone 4) |
| Deployment | Streamlit Cloud |
