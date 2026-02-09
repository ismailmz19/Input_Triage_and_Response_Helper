"""
preprocessing.py - Clean and preprocess parsed Enron emails.

Handles: HTML removal, signature stripping, quoted text removal,
text normalization, duplicate removal, and short email filtering.

Usage:
    python src/preprocessing.py

Input:  data/processed/parsed_emails.csv
Output: data/processed/cleaned_emails.csv
"""

import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

INPUT_PATH = os.path.join("data", "processed", "parsed_emails.csv")
OUTPUT_PATH = os.path.join("data", "processed", "cleaned_emails.csv")

MIN_WORDS = 10   # Remove emails shorter than this
MAX_WORDS = 500  # Truncate emails longer than this


def remove_html(text):
    """Strip HTML tags from text."""
    if not text or pd.isna(text):
        return ""
    try:
        return BeautifulSoup(text, "html.parser").get_text(separator=" ")
    except Exception:
        return text


def remove_quoted_text(text):
    """Remove quoted/forwarded text, keep only original content."""
    if not text:
        return text
    lines = text.split("\n")
    clean = []
    for line in lines:
        if line.strip().startswith(">"):
            continue
        if re.match(r"^-+\s*(Original Message|Forwarded)\s*-+", line, re.IGNORECASE):
            break
        if re.match(r"^On .+ wrote:\s*$", line):
            break
        clean.append(line)
    return "\n".join(clean)


def remove_signature(text):
    """Remove email signatures."""
    if not text:
        return text
    lines = text.split("\n")
    clean = []
    for line in lines:
        if line.strip() in ["--", "---", "___"]:
            break
        if re.match(r"^(Sent from my|Get Outlook for|Sent via)", line, re.IGNORECASE):
            break
        if line.strip().startswith("X-"):
            continue
        clean.append(line)
    return "\n".join(clean)


def remove_urls_and_emails(text):
    """Remove URLs and email addresses."""
    if not text:
        return text
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    return text


def clean_whitespace(text):
    """Normalize whitespace."""
    if not text:
        return text
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def clean_email(row):
    """Apply full cleaning pipeline to one email row."""
    body = str(row.get("body", ""))
    subject = str(row.get("subject", ""))
    
    # Clean body
    body = remove_html(body)
    body = remove_quoted_text(body)
    body = remove_signature(body)
    body = remove_urls_and_emails(body)
    body = clean_whitespace(body)
    
    # Parse reply/forward flags from subject
    is_reply = bool(re.match(r"^re:", subject, re.IGNORECASE))
    is_forward = bool(re.match(r"^(fw|fwd):", subject, re.IGNORECASE))
    subject_clean = re.sub(r"^(re|fw|fwd):\s*", "", subject, flags=re.IGNORECASE).strip()
    
    # Word count (after cleaning)
    word_count = len(body.split()) if body else 0
    
    # Truncate long emails
    if word_count > MAX_WORDS:
        body = " ".join(body.split()[:MAX_WORDS])
        word_count = MAX_WORDS
    
    return {
        "subject_clean": subject_clean,
        "body_clean": body,
        "is_reply": is_reply,
        "is_forward": is_forward,
        "word_count": word_count,
    }


def preprocess_emails(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    """Clean all parsed emails and save result."""
    print(f"Loading from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} emails.")
    
    # Apply cleaning pipeline
    print("Cleaning emails...")
    cleaned = [clean_email(row) for _, row in tqdm(df.iterrows(), total=len(df))]
    df_cleaned = pd.DataFrame(cleaned)
    
    # Merge with original metadata
    df_result = pd.concat([
        df[["sender", "to", "cc", "subject", "body", "date"]].reset_index(drop=True),
        df_cleaned.reset_index(drop=True)
    ], axis=1)
    
    # Filter short emails
    before = len(df_result)
    df_result = df_result[df_result["word_count"] >= MIN_WORDS].reset_index(drop=True)
    print(f"Filtered short: {before} -> {len(df_result)} (removed {before - len(df_result)})")
    
    # Remove duplicates
    before = len(df_result)
    df_result = df_result.drop_duplicates(subset=["subject_clean", "body_clean"]).reset_index(drop=True)
    print(f"Deduplicated:   {before} -> {len(df_result)} (removed {before - len(df_result)})")
    
    # Stats
    print(f"\n--- Results ---")
    print(f"Final count:     {len(df_result)}")
    print(f"Mean words:      {df_result['word_count'].mean():.1f}")
    print(f"Median words:    {df_result['word_count'].median():.1f}")
    print(f"Replies:         {df_result['is_reply'].sum()} ({df_result['is_reply'].mean()*100:.1f}%)")
    print(f"Forwards:        {df_result['is_forward'].sum()} ({df_result['is_forward'].mean()*100:.1f}%)")
    
    df_result.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    return df_result


if __name__ == "__main__":
    preprocess_emails()
