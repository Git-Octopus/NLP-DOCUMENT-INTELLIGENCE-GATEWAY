"""
Rule-based Extraction module for NLP Document Intelligence Gateway.

Provides high-precision deterministic regex pattern extractors for emails,
phone numbers, currency/money, order & ticket IDs, dates, and evidence sentence mining.
"""

import re
from typing import List, Dict, Any, Optional
from app.nlp.preprocess import tokenize_sentences

# Compile high-precision regex patterns
REGEX_PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "PHONE": re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "MONEY": re.compile(
        r"(?:[\$€£₹¥]|USD\s?|EUR\s?|GBP\s?|INR\s?)\d+(?:,\d{3})*(?:\.\d{1,2})?\b|\b\d+(?:,\d{3})*(?:\.\d{1,2})?\s?(?:USD|EUR|GBP|INR|AUD|CAD|dollars|cents)\b",
        re.IGNORECASE,
    ),
    "ORDER_ID": re.compile(
        r"(?:#|ORD[-_]|ORDER[-_]?|TKT[-_]|TICKET[-_]?|ID:\s?)[A-Z0-9_-]{3,15}\b",
        re.IGNORECASE,
    ),
    "DATE": re.compile(
        r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*,?\s+\d{4}\b",
        re.IGNORECASE,
    ),
}


def extract_rule_entities(text: str) -> List[Dict[str, Any]]:
    """
    Runs high-precision deterministic regex pattern extractors:
    - Emails: RFC 5322 regex pattern
    - Phone Numbers: International and local formats
    - Money/Currency: Amounts like $49.99, USD 50, ₹500, €12.50
    - Order/Ticket IDs: Alpha-numeric patterns like #ORD-12345, TKT-9982, ID: 88392
    - Dates: Formats like YYYY-MM-DD, DD/MM/YYYY, Jan 15, 2026

    Returns list of dicts formatted as:
    [{"entity": str, "label": str, "source": "rule", "start": int, "end": int}]
    """
    if not text:
        return []

    entities = []

    for label, pattern in REGEX_PATTERNS.items():
        for match in pattern.finditer(text):
            entity_str = match.group().strip()
            start_idx = match.start()
            end_idx = match.end()

            # Specific label refined for tickets vs order ids
            final_label = label
            if label == "ORDER_ID" and (
                "TKT" in entity_str.upper() or "TICKET" in entity_str.upper()
            ):
                final_label = "TICKET_ID"

            entities.append(
                {
                    "entity": entity_str,
                    "text": entity_str,  # schema compatibility
                    "label": final_label,
                    "source": "rule",
                    "start": start_idx,
                    "end": end_idx,
                    "start_char": start_idx,  # schema compatibility
                    "end_char": end_idx,  # schema compatibility
                }
            )

    # Sort entities by start index
    entities.sort(key=lambda x: x["start"])
    return entities


def extract_regex_entities(text: str) -> List[Dict[str, Any]]:
    """
    Backwards-compatible wrapper for rule-based regex extraction.
    """
    return extract_rule_entities(text)


def extract_evidence_sentences(
    text: str, keywords: List[str], max_sentences: int = 3
) -> List[str]:
    """
    Extracts key justification sentences from the document containing high-density
    keywords or strong entity signals.
    """
    if not text or not text.strip():
        return []

    sentences = tokenize_sentences(text)
    if not sentences:
        return []

    if not keywords:
        return sentences[:max_sentences]

    keyword_set = set(k.lower() for k in keywords)

    scored_sentences = []
    for idx, sentence in enumerate(sentences):
        sent_lower = sentence.lower()
        # Count keyword hits in sentence
        score = sum(1 for kw in keyword_set if kw in sent_lower)
        scored_sentences.append((score, idx, sentence))

    # Sort by keyword score descending, maintaining original sentence order for ties
    scored_sentences.sort(key=lambda x: (-x[0], x[1]))

    top_sentences = [s[2] for s in scored_sentences[:max_sentences]]
    return top_sentences
