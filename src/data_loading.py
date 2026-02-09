"""
data_loading.py - Load and parse the Enron Email Dataset from CSV.

Parses raw email messages into structured fields:
sender, recipients, cc, subject, body, date.

Usage:
    python src/data_loading.py

Input:  data/raw/emails.csv (download from Kaggle)
Output: data/processed/parsed_emails.csv
"""

import os
import sys
import email
import email.utils
import pandas as pd
import numpy as np
from tqdm import tqdm

# Paths
RAW_PATH = os.path.join("data", "raw", "emails.csv")
OUTPUT_PATH = os.path.join("data", "processed", "parsed_emails.csv")


def parse_email_message(raw_message):
    """
    Parse a raw email string into structured fields.
    
    Args:
        raw_message (str): Raw email message including headers and body.
        
    Returns:
        dict: Parsed fields - sender, to, cc, subject, body, date
    """
    try:
        msg = email.message_from_string(raw_message)
        
        # Extract header fields
        sender = msg.get("From", "")
        to = msg.get("To", "")
        cc = msg.get("Cc", "")
        subject = msg.get("Subject", "")
        date_str = msg.get("Date", "")
        
        # Extract body text
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        body = part.get_payload(decode=True).decode("utf-8", errors="replace")
                    except Exception:
                        body = str(part.get_payload())
                    break
        else:
            try:
                body = msg.get_payload(decode=True).decode("utf-8", errors="replace")
            except Exception:
                body = str(msg.get_payload())
        
        # Parse date into standard format
        date_parsed = None
        if date_str:
            try:
                date_parsed = email.utils.parsedate_to_datetime(date_str)
                date_parsed = date_parsed.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                date_parsed = None
        
        return {
            "sender": sender.strip() if sender else "",
            "to": to.strip() if to else "",
            "cc": cc.strip() if cc else "",
            "subject": subject.strip() if subject else "",
            "body": body.strip() if body else "",
            "date": date_parsed,
        }
    except Exception:
        return {"sender": "", "to": "", "cc": "", "subject": "", "body": "", "date": None}


def load_and_parse_enron(input_path=RAW_PATH, output_path=OUTPUT_PATH, sample_size=None):
    """
    Load Enron email CSV, parse all messages, save structured output.
    
    Args:
        input_path: Path to raw emails.csv
        output_path: Path for parsed output CSV
        sample_size: Optional limit for testing (None = all)
    """
    print(f"Loading raw emails from: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"\nERROR: {input_path} not found!")
        print("Please download the Enron dataset from Kaggle:")
        print("  https://www.kaggle.com/datasets/wcukierski/enron-email-dataset")
        print(f"Place emails.csv in: {os.path.dirname(input_path)}/")
        sys.exit(1)
    
    # Load CSV
    df_raw = pd.read_csv(input_path)
    print(f"Loaded {len(df_raw)} raw records.")
    
    if sample_size:
        df_raw = df_raw.head(sample_size)
        print(f"Using sample of {sample_size} emails.")
    
    # Parse each email message
    print("Parsing email messages...")
    parsed = []
    for idx, row in tqdm(df_raw.iterrows(), total=len(df_raw)):
        raw_msg = row.get("message", "")
        if pd.isna(raw_msg) or not raw_msg:
            continue
        result = parse_email_message(str(raw_msg))
        result["original_file"] = row.get("file", "")
        parsed.append(result)
    
    df = pd.DataFrame(parsed)
    
    # Report stats
    print(f"\n--- Parsing Results ---")
    print(f"Total parsed:        {len(df)}")
    print(f"With body:           {(df['body'] != '').sum()} ({(df['body'] != '').mean()*100:.1f}%)")
    print(f"With subject:        {(df['subject'] != '').sum()} ({(df['subject'] != '').mean()*100:.1f}%)")
    print(f"With date:           {df['date'].notna().sum()} ({df['date'].notna().mean()*100:.1f}%)")
    print(f"Unique senders:      {df['sender'].nunique()}")
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    return df


if __name__ == "__main__":
    load_and_parse_enron()
