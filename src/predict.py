"""
predict.py — Full End-to-End Inference Pipeline
INSE 6450: AI in Systems Engineering — Winter 2026
Student: Ismail Mzouri (40335670)

What this script does:
  Combines the MLP classifier + response template suggestion into one
  complete pipeline. Given a raw email (subject + body), it:
    1. Preprocesses and extracts features (reusing Phase 1 pipeline)
    2. Predicts the email label using the trained MLP
    3. Returns confidence scores for all 5 classes
    4. Suggests top-3 ranked response templates for the predicted label

Usage (interactive demo):
  python src/predict.py

Usage (as a module):
  from predict import predict_email
  result = predict_email(subject="Server down", body="Production is broken...")
  print(result["label"], result["confidence"])
  for t in result["templates"]:
      print(t["template"])

Prerequisites:
  Run src/train.py first to generate model artifacts.
"""

import os
import sys
import json
import time
import joblib
import re

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import issparse

# Add src/ to path so we can import response_templates
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from response_templates import suggest_responses

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
CONFIG_PATH = os.path.join(MODELS_DIR, "mlp_model_config.json")
MODEL_PATH  = os.path.join(MODELS_DIR, "mlp_model.pth")
LE_PATH     = os.path.join(MODELS_DIR, "label_encoder.joblib")
PIPELINE_PATH = os.path.join(BASE_DIR, "data", "processed", "feature_pipeline.joblib")


# ──────────────────────────────────────────────
# MLP DEFINITION (must match train.py exactly)
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
# LOAD MODEL (once at import time)
# ──────────────────────────────────────────────
def load_model():
    """Load MLP, label encoder, feature pipeline, and config."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run 'python src/train.py' first."
        )

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    le = joblib.load(LE_PATH)
    device = torch.device("cpu")  # CPU inference for deployment

    model = EmailMLP(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Load the Phase 1 feature pipeline if available
    pipeline = None
    if os.path.exists(PIPELINE_PATH):
        pipeline = joblib.load(PIPELINE_PATH)

    return model, le, config, device, pipeline


# ──────────────────────────────────────────────
# FEATURE EXTRACTION FOR A SINGLE EMAIL
# ──────────────────────────────────────────────
def _basic_clean(text: str) -> str:
    """Minimal cleaning: lowercase, remove HTML tags, collapse whitespace."""
    text = re.sub(r'<[^>]+>', ' ', text)          # strip HTML
    text = re.sub(r'[^\w\s]', ' ', text.lower())  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_features_single(subject: str, body: str, pipeline, config):
    """
    Extract features for a single email using the saved Phase 1 pipeline.
    Falls back to a zero vector if the pipeline is unavailable.
    """
    if pipeline is not None:
        try:
            combined_text = _basic_clean(subject + " " + body)
            # The pipeline expects a list of strings
            X = pipeline.transform([combined_text])
            if issparse(X):
                X = X.toarray()
            return X.astype(np.float32)
        except Exception as e:
            print(f"    Warning: Feature pipeline failed ({e}). Using zero vector.")

    # Fallback: zero vector of correct input dimension
    return np.zeros((1, config["input_dim"]), dtype=np.float32)


# ──────────────────────────────────────────────
# MAIN PREDICTION FUNCTION
# ──────────────────────────────────────────────
def predict_email(
    subject: str,
    body: str,
    model=None,
    le=None,
    config=None,
    device=None,
    pipeline=None,
    confidence_threshold: float = 0.50,
    top_k_templates: int = 3,
) -> dict:
    """
    Full end-to-end prediction for a single email.

    Args:
        subject:              Email subject line.
        body:                 Email body text.
        model, le, config,
        device, pipeline:     Pre-loaded artifacts (loaded if None).
        confidence_threshold: If max softmax prob < this, use heuristic fallback label.
        top_k_templates:      Number of response templates to return.

    Returns:
        dict with keys:
          label          — predicted class name (str)
          confidence     — max softmax probability (float)
          all_scores     — dict of {class_name: probability} for all 5 classes
          templates      — list of top_k ranked response template dicts
          latency_ms     — inference latency in milliseconds
          used_fallback  — True if confidence was below threshold
    """
    # Load artifacts if not provided
    if model is None:
        model, le, config, device, pipeline = load_model()

    t0 = time.perf_counter()

    # Feature extraction
    X = extract_features_single(subject, body, pipeline, config)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # Inference
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    latency_ms = (time.perf_counter() - t0) * 1000

    # Decode prediction
    pred_idx = int(np.argmax(probs))
    pred_label = le.classes_[pred_idx]
    confidence = float(probs[pred_idx])
    all_scores = {cls: round(float(p), 4) for cls, p in zip(le.classes_, probs)}

    # Confidence threshold — fallback to heuristic if uncertain
    used_fallback = False
    if confidence < confidence_threshold:
        pred_label = _heuristic_label(subject, body)
        used_fallback = True

    # Response template suggestion
    templates = suggest_responses(pred_label, body, top_k=top_k_templates)

    return {
        "label":        pred_label,
        "confidence":   round(confidence, 4),
        "all_scores":   all_scores,
        "templates":    templates,
        "latency_ms":   round(latency_ms, 3),
        "used_fallback": used_fallback,
    }


# ──────────────────────────────────────────────
# HEURISTIC FALLBACK (replicates Phase 1 logic)
# ──────────────────────────────────────────────
def _heuristic_label(subject: str, body: str) -> str:
    """Simple keyword-based fallback when model confidence is low."""
    text = (subject + " " + body).lower()

    urgent_kw     = ["urgent", "asap", "immediately", "critical", "emergency", "deadline"]
    scheduling_kw = ["meeting", "schedule", "calendar", "invite", "availability", "reschedule"]
    spam_kw       = ["unsubscribe", "promotion", "offer", "newsletter", "deal", "discount"]

    if any(k in text for k in urgent_kw):
        return "Urgent"
    if any(k in text for k in scheduling_kw):
        return "Scheduling"
    if any(k in text for k in spam_kw):
        return "Spam/Low Priority"
    if "?" in body:
        return "Needs Reply"
    return "Informational"


# ──────────────────────────────────────────────
# INTERACTIVE DEMO
# ──────────────────────────────────────────────
def _print_result(result: dict):
    print(f"\n  Predicted Label : {result['label']}")
    print(f"  Confidence      : {result['confidence']:.1%}")
    if result["used_fallback"]:
        print(f"  [Fallback mode: confidence below threshold]")
    print(f"  Latency         : {result['latency_ms']:.2f} ms")
    print(f"\n  Class Probabilities:")
    for cls, score in sorted(result["all_scores"].items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 20)
        print(f"    {cls:<20} {score:.3f}  {bar}")
    print(f"\n  Top Response Templates:")
    for t in result["templates"]:
        print(f"  ─── #{t['rank']} [{t['id']}]  Subject: {t['subject']}")
        lines = t["template"].split("\n")
        for line in lines[:5]:  # show first 5 lines
            print(f"      {line}")
        if len(lines) > 5:
            print(f"      ...")
        print()


def main():
    print("Loading model artifacts...")
    try:
        model, le, config, device, pipeline = load_model()
        print(f"Model loaded. Classes: {list(le.classes_)}\n")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    demo_emails = [
        {
            "subject": "URGENT: Production server is down",
            "body": "The main production server crashed 10 minutes ago. We have a critical client demo in 2 hours. Please escalate this immediately to the infrastructure team. This needs to be fixed ASAP.",
        },
        {
            "subject": "Team meeting next week",
            "body": "Hi, I'd like to schedule a team sync for next Thursday to discuss Q2 planning. Could you let me know your availability? We'll need about 1 hour.",
        },
        {
            "subject": "Question about the onboarding process",
            "body": "Hi, I just joined the team and I was wondering if you could walk me through the onboarding documentation? I have a few questions about the development environment setup.",
        },
        {
            "subject": "FYI: Q1 report published",
            "body": "Just a heads up — the Q1 financial report has been published on the internal portal. No action needed from your side, just sharing for your awareness.",
        },
        {
            "subject": "Exclusive deal just for you!",
            "body": "Don't miss our limited time offer! Get 50% off all premium subscriptions this weekend only. Click here to unsubscribe from future promotions.",
        },
    ]

    print("=" * 65)
    print("  INBOX TRIAGE & RESPONSE HELPER — Full Pipeline Demo")
    print("=" * 65)

    for i, email in enumerate(demo_emails, 1):
        print(f"\n{'─'*65}")
        print(f"  Email #{i}")
        print(f"  Subject : {email['subject']}")
        print(f"  Body    : {email['body'][:80]}...")
        result = predict_email(
            subject=email["subject"],
            body=email["body"],
            model=model, le=le, config=config,
            device=device, pipeline=pipeline,
        )
        _print_result(result)

    # Interactive mode
    print("\n" + "=" * 65)
    print("  INTERACTIVE MODE — paste your own email")
    print("  (press Enter twice to submit, type 'quit' to exit)")
    print("=" * 65)

    while True:
        print("\nSubject: ", end="")
        subject = input().strip()
        if subject.lower() == "quit":
            break
        print("Body (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        body = " ".join(lines)

        if not body:
            print("No body entered, skipping.")
            continue

        result = predict_email(
            subject=subject, body=body,
            model=model, le=le, config=config,
            device=device, pipeline=pipeline,
        )
        _print_result(result)


if __name__ == "__main__":
    main()