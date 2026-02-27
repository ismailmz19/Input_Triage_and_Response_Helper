"""
response_templates.py — Response Template Suggestion
INSE 6450: AI in Systems Engineering — Winter 2026
Student: Ismail Mzouri (40335670)

What this module does:
  Given a predicted email label (from the MLP classifier), returns
  ranked response template suggestions appropriate for that category.

  Ranking is done by keyword similarity between the email body and
  each template — templates containing words that appear in the email
  are ranked higher, making suggestions contextually relevant.

Usage (standalone):
  python src/response_templates.py

Usage (as a module):
  from response_templates import suggest_responses
  suggestions = suggest_responses(label="Urgent", email_body="The server is down...")
  for s in suggestions:
      print(s["template"])
"""

import re
from collections import Counter


# ──────────────────────────────────────────────
# TEMPLATE LIBRARY
# ──────────────────────────────────────────────
TEMPLATES = {
    "Urgent": [
        {
            "id": "URG-1",
            "subject": "Re: [URGENT] Acknowledged",
            "template": (
                "Hi,\n\n"
                "I've received your urgent message and am addressing it immediately. "
                "I'll get back to you with an update within the next 30 minutes.\n\n"
                "Best regards"
            ),
            "keywords": ["urgent", "immediately", "asap", "critical", "emergency"],
        },
        {
            "id": "URG-2",
            "subject": "Re: On it now",
            "template": (
                "Hi,\n\n"
                "Understood — I'm on this right now. "
                "I'll keep you posted as the situation develops.\n\n"
                "Best regards"
            ),
            "keywords": ["fix", "issue", "problem", "broken", "down", "error"],
        },
        {
            "id": "URG-3",
            "subject": "Re: Escalating as requested",
            "template": (
                "Hi,\n\n"
                "I've flagged this as a priority and escalated to the relevant team. "
                "You should hear back shortly. Thank you for bringing this to my attention.\n\n"
                "Best regards"
            ),
            "keywords": ["escalate", "manager", "team", "priority", "deadline"],
        },
    ],

    "Needs Reply": [
        {
            "id": "NR-1",
            "subject": "Re: Your question",
            "template": (
                "Hi,\n\n"
                "Thank you for reaching out. Happy to help — here is my response:\n\n"
                "[Your answer here]\n\n"
                "Please let me know if you need any further clarification.\n\n"
                "Best regards"
            ),
            "keywords": ["question", "help", "wondering", "could you", "can you"],
        },
        {
            "id": "NR-2",
            "subject": "Re: Following up",
            "template": (
                "Hi,\n\n"
                "Thanks for your follow-up. I wanted to let you know that I'm looking into this "
                "and will have a full response for you by end of day.\n\n"
                "Best regards"
            ),
            "keywords": ["follow up", "following up", "checking in", "update", "status"],
        },
        {
            "id": "NR-3",
            "subject": "Re: Request received",
            "template": (
                "Hi,\n\n"
                "I've received your request and will process it shortly. "
                "I'll confirm once it has been completed.\n\n"
                "Best regards"
            ),
            "keywords": ["request", "please", "would like", "need", "require"],
        },
    ],

    "Informational": [
        {
            "id": "INF-1",
            "subject": "Re: Noted, thank you",
            "template": (
                "Hi,\n\n"
                "Thank you for the update — noted and appreciated.\n\n"
                "Best regards"
            ),
            "keywords": ["update", "fyi", "info", "note", "heads up"],
        },
        {
            "id": "INF-2",
            "subject": "Re: Thanks for sharing",
            "template": (
                "Hi,\n\n"
                "Thank you for sharing this. I'll review it and reach out if I have any questions.\n\n"
                "Best regards"
            ),
            "keywords": ["share", "document", "report", "attached", "newsletter"],
        },
        {
            "id": "INF-3",
            "subject": "Re: Acknowledged",
            "template": (
                "Hi,\n\n"
                "Acknowledged — thank you for keeping me in the loop.\n\n"
                "Best regards"
            ),
            "keywords": ["loop", "cc", "inform", "awareness", "notice"],
        },
    ],

    "Scheduling": [
        {
            "id": "SCH-1",
            "subject": "Re: Meeting confirmed",
            "template": (
                "Hi,\n\n"
                "The meeting works for me. I've added it to my calendar and look forward to connecting.\n\n"
                "Best regards"
            ),
            "keywords": ["meeting", "calendar", "invite", "confirmed", "schedule"],
        },
        {
            "id": "SCH-2",
            "subject": "Re: Availability",
            "template": (
                "Hi,\n\n"
                "I'm available on the following dates and times:\n\n"
                "- [Option 1: Day, Date, Time]\n"
                "- [Option 2: Day, Date, Time]\n"
                "- [Option 3: Day, Date, Time]\n\n"
                "Please let me know which works best for you.\n\n"
                "Best regards"
            ),
            "keywords": ["available", "availability", "time", "slot", "free", "when"],
        },
        {
            "id": "SCH-3",
            "subject": "Re: Rescheduling request",
            "template": (
                "Hi,\n\n"
                "Unfortunately I have a conflict at the proposed time. "
                "Could we reschedule to one of the following?\n\n"
                "- [Alternative 1]\n"
                "- [Alternative 2]\n\n"
                "Apologies for any inconvenience.\n\n"
                "Best regards"
            ),
            "keywords": ["reschedule", "conflict", "postpone", "move", "change"],
        },
    ],

    "Spam/Low Priority": [
        {
            "id": "SLP-1",
            "subject": "Re: Thank you",
            "template": (
                "Hi,\n\n"
                "Thank you for your message. "
                "I'll keep this on file for future reference.\n\n"
                "Best regards"
            ),
            "keywords": ["newsletter", "unsubscribe", "promotion", "offer"],
        },
        {
            "id": "SLP-2",
            "subject": "Re: No action required",
            "template": (
                "Hi,\n\n"
                "Thank you for reaching out. "
                "At this time, we don't require any further action on this matter.\n\n"
                "Best regards"
            ),
            "keywords": ["promo", "deal", "discount", "marketing", "advertisement"],
        },
    ],
}


# ──────────────────────────────────────────────
# KEYWORD SIMILARITY RANKING
# ──────────────────────────────────────────────
def _tokenize(text: str) -> Counter:
    """Lowercase and tokenize text into word counts."""
    words = re.findall(r'\b[a-z]{2,}\b', text.lower())
    return Counter(words)


def _keyword_score(email_tokens: Counter, template_keywords: list) -> float:
    """Score a template by how many of its keywords appear in the email."""
    if not template_keywords:
        return 0.0
    hits = sum(1 for kw in template_keywords if kw in email_tokens)
    return hits / len(template_keywords)


def suggest_responses(label: str, email_body: str = "", top_k: int = 3) -> list:
    """
    Given a predicted label and optional email body, return top_k ranked
    response template suggestions.

    Args:
        label:      Predicted label from the MLP classifier.
        email_body: Raw email body text (used for keyword ranking).
        top_k:      Number of suggestions to return.

    Returns:
        List of dicts with keys: rank, id, subject, template, score.
    """
    if label not in TEMPLATES:
        return [{
            "rank": 1,
            "id": "GEN-1",
            "subject": "Re: Your message",
            "template": "Hi,\n\nThank you for your message. I will get back to you shortly.\n\nBest regards",
            "score": 0.0,
        }]

    candidates = TEMPLATES[label]
    email_tokens = _tokenize(email_body) if email_body else Counter()

    scored = []
    for t in candidates:
        score = _keyword_score(email_tokens, t.get("keywords", []))
        scored.append({**t, "score": score})

    # Sort by score descending, then by original order for ties
    scored.sort(key=lambda x: x["score"], reverse=True)

    results = []
    for rank, t in enumerate(scored[:top_k], start=1):
        results.append({
            "rank":     rank,
            "id":       t["id"],
            "subject":  t["subject"],
            "template": t["template"],
            "score":    round(t["score"], 4),
        })

    return results


# ──────────────────────────────────────────────
# DEMO — run standalone to see suggestions
# ──────────────────────────────────────────────
if __name__ == "__main__":
    demo_cases = [
        {
            "label": "Urgent",
            "body": "The production server is down and we have a critical deadline in 2 hours. Please fix this asap."
        },
        {
            "label": "Scheduling",
            "body": "Hi, could we set up a meeting to discuss Q3 planning? Let me know your availability."
        },
        {
            "label": "Needs Reply",
            "body": "I was wondering if you could help me with the onboarding documentation? I have a few questions."
        },
        {
            "label": "Informational",
            "body": "FYI — the quarterly report has been published. Attaching it here for your awareness."
        },
        {
            "label": "Spam/Low Priority",
            "body": "Exclusive offer just for you! Get 50% off our premium newsletter subscription today."
        },
    ]

    print("=" * 65)
    print("  INBOX TRIAGE & RESPONSE HELPER — Template Suggestions Demo")
    print("=" * 65)

    for case in demo_cases:
        print(f"\n[Label: {case['label']}]")
        print(f"Email: {case['body'][:80]}...")
        print("-" * 65)
        suggestions = suggest_responses(case["label"], case["body"])
        for s in suggestions:
            print(f"  #{s['rank']} [{s['id']}] (keyword match: {s['score']:.0%})")
            print(f"  Subject: {s['subject']}")
            print(f"  {s['template'][:120].strip()}...")
            print()