"""
label_generator.py - Generate category labels using heuristic rules.

Assigns one of 5 labels to each email based on keyword patterns,
metadata signals, and structural features.

Labels: Urgent, Needs Reply, Informational, Scheduling, Spam/Low Priority
Priority: Urgent > Scheduling > Needs Reply > Informational > Spam/Low Priority

Usage:
    python src/label_generator.py

Input:  data/processed/cleaned_emails.csv
Output: data/processed/labeled_emails.csv
"""

import os
import pandas as pd
from tqdm import tqdm

INPUT_PATH = os.path.join("data", "processed", "cleaned_emails.csv")
OUTPUT_PATH = os.path.join("data", "processed", "labeled_emails.csv")

# Keyword dictionaries for each category
KEYWORDS = {
    "Urgent": [
        "urgent", "asap", "immediately", "deadline", "critical", "emergency",
        "action required", "time sensitive", "right away", "high priority",
        "must", "need this today", "eod", "end of day", "escalat", "crisis"
    ],
    "Scheduling": [
        "meeting", "schedule", "calendar", "available", "conference room",
        "agenda", "invite", "conference call", "reschedule", "appointment",
        "book", "slot", "availability", "lunch", "dinner", "call at",
        "monday", "tuesday", "wednesday", "thursday", "friday",
    ],
    "Informational": [
        "fyi", "for your information", "update", "announcement", "newsletter",
        "no action needed", "no response needed", "just letting you know",
        "please note", "heads up", "reminder", "be aware", "notice",
        "summary", "report attached", "see attached", "status update"
    ],
    "Spam/Low Priority": [
        "unsubscribe", "click here", "free", "offer", "promotion",
        "limited time", "act now", "congratulations", "winner",
        "discount", "deal", "subscribe", "opt out", "no obligation"
    ],
    "Needs Reply": [
        "please respond", "let me know", "your thoughts", "can you",
        "would you", "could you", "what do you think", "get back to me",
        "any update", "feedback", "input needed", "please advise",
        "please confirm", "awaiting your"
    ],
}


def count_matches(text, keywords):
    """Count keyword matches in text."""
    if not text or pd.isna(text):
        return 0
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def assign_label(row):
    """
    Assign a category label using heuristic rules.
    Priority: Urgent > Scheduling > Needs Reply > Informational > Spam/Low Priority
    """
    subject = str(row.get("subject_clean", "")).lower()
    body = str(row.get("body_clean", "")).lower()
    combined = f"{subject} {body}"
    
    to_str = str(row.get("to", ""))
    cc_str = str(row.get("cc", ""))
    is_reply = row.get("is_reply", False)
    is_forward = row.get("is_forward", False)
    
    # Count recipients
    num_to = len(to_str.split(",")) if to_str and to_str != "nan" else 0
    num_cc = len(cc_str.split(",")) if cc_str and cc_str != "nan" else 0
    num_recipients = num_to + num_cc
    
    question_marks = body.count("?")
    sender = str(row.get("sender", ""))
    external = "@enron.com" not in sender.lower()
    
    # Score each category
    scores = {cat: count_matches(combined, kws) for cat, kws in KEYWORDS.items()}
    
    # Metadata boosting
    if "urgent" in subject or "action required" in subject:
        scores["Urgent"] += 3
    if question_marks >= 2:
        scores["Needs Reply"] += 2
    elif question_marks >= 1:
        scores["Needs Reply"] += 1
    if is_reply and num_recipients <= 3:
        scores["Needs Reply"] += 1
    if is_forward or num_recipients > 10:
        scores["Informational"] += 2
    if external and scores["Spam/Low Priority"] >= 1:
        scores["Spam/Low Priority"] += 2
    
    max_score = max(scores.values())
    
    # If no signal, use defaults
    if max_score == 0:
        if question_marks > 0:
            return "Needs Reply"
        if is_forward or num_recipients > 5:
            return "Informational"
        if external:
            return "Spam/Low Priority"
        return "Informational"
    
    # Return highest-priority label with max score
    priority = ["Urgent", "Scheduling", "Needs Reply", "Informational", "Spam/Low Priority"]
    for label in priority:
        if scores[label] == max_score:
            return label
    return "Informational"


def generate_labels(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    """Apply heuristic labeling to all cleaned emails."""
    print(f"Loading from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} emails.")
    
    print("Generating labels...")
    df["label"] = [assign_label(row) for _, row in tqdm(df.iterrows(), total=len(df))]
    
    # Report distribution
    print(f"\n--- Label Distribution ---")
    for label, count in df["label"].value_counts().items():
        print(f"  {label:<20s}: {count:>7d} ({count/len(df)*100:5.1f}%)")
    print(f"  {'Total':<20s}: {len(df):>7d}")
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    return df


if __name__ == "__main__":
    generate_labels()
