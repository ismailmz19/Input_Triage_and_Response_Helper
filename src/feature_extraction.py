"""
feature_extraction.py - Extract features for email classification.

Three feature families:
1. TF-IDF text features (subject + body) - 10,000 dim sparse
2. Metadata features (recipients, flags, time) - 14 dim dense
3. Linguistic features (sentiment, greeting, caps) - 5 dim dense

Usage:
    python src/feature_extraction.py

Input:  data/processed/labeled_emails.csv
Output: data/processed/features.npz
        data/processed/labels.csv
        data/processed/feature_pipeline.joblib
"""

import os
import re
import math
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib

INPUT_PATH = os.path.join("data", "processed", "labeled_emails.csv")
OUTPUT_FEATURES = os.path.join("data", "processed", "features.npz")
OUTPUT_LABELS = os.path.join("data", "processed", "labels.csv")
OUTPUT_PIPELINE = os.path.join("data", "processed", "feature_pipeline.joblib")


def extract_metadata_features(df):
    """Extract 14 metadata features from email headers."""
    print("  Extracting metadata features...")
    features = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        to_str = str(row.get("to", ""))
        cc_str = str(row.get("cc", ""))
        num_to = len(to_str.split(",")) if to_str and to_str != "nan" else 0
        num_cc = len(cc_str.split(",")) if cc_str and cc_str != "nan" else 0
        
        body = str(row.get("body_clean", ""))
        subject = str(row.get("subject_clean", ""))
        sender = str(row.get("sender", ""))
        
        # Time features
        hour = 12
        is_biz = 1
        date_str = row.get("date", "")
        if date_str and str(date_str) != "nan":
            try:
                dt = pd.to_datetime(date_str)
                hour = dt.hour
                is_biz = 1 if (8 <= hour <= 18 and dt.weekday() < 5) else 0
            except Exception:
                pass
        
        features.append([
            num_to + num_cc,                              # num_recipients
            1 if num_cc > 0 else 0,                       # has_cc
            1 if row.get("is_reply", False) else 0,       # is_reply
            1 if row.get("is_forward", False) else 0,     # is_forward
            len(subject.split()),                          # subject_length
            math.log1p(row.get("word_count", len(body.split()))),  # body_length_log
            body.count("?"),                               # question_marks
            body.count("!"),                               # exclamation_marks
            math.sin(2 * math.pi * hour / 24),            # hour_sin
            math.cos(2 * math.pi * hour / 24),            # hour_cos
            is_biz,                                        # is_business_hours
            1 if "@enron.com" not in sender.lower() else 0,  # is_external
            1 if any(k in body.lower() for k in ["attached", "attachment"]) else 0,  # has_attachment_ref
            min(num_to + num_cc, 20),                      # recipients_capped
        ])
    return np.array(features, dtype=np.float32)


def extract_linguistic_features(df):
    """Extract 5 linguistic features from email text."""
    print("  Extracting linguistic features...")
    try:
        from textblob import TextBlob
        use_tb = True
    except ImportError:
        print("    TextBlob not available, sentiment will be 0.")
        use_tb = False
    
    features = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        body = str(row.get("body_clean", ""))
        
        # Sentiment
        sentiment = 0.0
        if use_tb and body:
            try:
                sentiment = TextBlob(body[:1000]).sentiment.polarity
            except Exception:
                pass
        
        # Greeting
        first_line = body.split("\n")[0].strip().lower() if body else ""
        has_greeting = 1 if any(first_line.startswith(g) for g in
                               ["hi ", "hello", "dear ", "hey ", "good morning"]) else 0
        
        # Closing
        has_closing = 1 if any(k in body.lower() for k in
                              ["thanks", "regards", "best,", "sincerely", "cheers"]) else 0
        
        # Caps ratio
        alpha = [c for c in body if c.isalpha()]
        caps_ratio = sum(1 for c in alpha if c.isupper()) / max(len(alpha), 1)
        
        # Avg sentence length
        sents = [s.strip() for s in re.split(r"[.!?]+", body) if s.strip()]
        avg_sent_len = np.mean([len(s.split()) for s in sents]) if sents else 0.0
        
        features.append([sentiment, has_greeting, has_closing, caps_ratio, avg_sent_len])
    return np.array(features, dtype=np.float32)


def extract_all_features(input_path=INPUT_PATH):
    """Extract and combine all three feature families."""
    print(f"Loading from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} emails.\n")
    
    # 1. TF-IDF
    print("1. TF-IDF features...")
    df["text_combined"] = df["subject_clean"].fillna("") + " " + df["body_clean"].fillna("")
    tfidf = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2),
        min_df=5, max_df=0.95, sublinear_tf=True, stop_words="english",
    )
    X_tfidf = tfidf.fit_transform(df["text_combined"])
    print(f"   Shape: {X_tfidf.shape}")
    
    # 2. Metadata
    print("2. Metadata features...")
    X_meta = extract_metadata_features(df)
    print(f"   Shape: {X_meta.shape}")
    
    # 3. Linguistic
    print("3. Linguistic features...")
    X_ling = extract_linguistic_features(df)
    print(f"   Shape: {X_ling.shape}")
    
    # Normalize dense features
    scaler_meta = StandardScaler()
    scaler_ling = StandardScaler()
    X_meta_s = scaler_meta.fit_transform(X_meta)
    X_ling_s = scaler_ling.fit_transform(X_ling)
    
    # Combine all features
    X = sparse.hstack([X_tfidf, sparse.csr_matrix(X_meta_s), sparse.csr_matrix(X_ling_s)])
    labels = df["label"].values
    
    print(f"\nCombined shape: {X.shape}")
    
    # Save outputs
    os.makedirs(os.path.dirname(OUTPUT_FEATURES), exist_ok=True)
    sparse.save_npz(OUTPUT_FEATURES, X)
    pd.DataFrame({"label": labels}).to_csv(OUTPUT_LABELS, index=False)
    joblib.dump({"tfidf": tfidf, "scaler_meta": scaler_meta, "scaler_ling": scaler_ling}, OUTPUT_PIPELINE)
    
    print(f"\nSaved: {OUTPUT_FEATURES}")
    print(f"Saved: {OUTPUT_LABELS}")
    print(f"Saved: {OUTPUT_PIPELINE}")
    
    # Label distribution
    print(f"\n--- Labels ---")
    unique, counts = np.unique(labels, return_counts=True)
    for l, c in zip(unique, counts):
        print(f"  {l:<20s}: {c:>7d} ({c/len(labels)*100:5.1f}%)")
    
    return X, labels


if __name__ == "__main__":
    extract_all_features()
