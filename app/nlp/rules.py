"""
Rule-based Extraction module for NLP Document Intelligence Gateway.

Provides high-precision deterministic regex pattern extractors for emails,
phone numbers, currency/money, order & ticket IDs, dates, and evidence sentence mining.
"""

import re
from typing import List, Dict, Any, Optional
from app.nlp.preprocess import tokenize_sentences

# Compile high-precision deterministic regex patterns
REGEX_PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "PHONE": re.compile(
        (
            r"(?:\+?91[\-\s]?)?[6-9]\d{4}[\-\s]?\d{5}\b|"
            r"(?:\+?91[\-\s]?)?[6-9]\d{9}\b|"
            r"(?:\+?\d{1,3}[\-\s.]?)?\(?\d{3}\)?[\-\s.]?\d{3}[\-\s.]?\d{4}\b"
        ),
        re.IGNORECASE,
    ),
    "MONEY": re.compile(
        (
            r"(?:[\$€£₹¥]|Rs\.?\s?|INR\s?|USD\s?|EUR\s?|GBP\s?)\d+(?:,\d{3})*(?:\.\d{1,2})?\b|"
            r"\b\d+(?:,\d{3})*(?:\.\d{1,2})?\s?(?:USD|EUR|GBP|INR|AUD|CAD|dollars|cents|rupees)\b"
        ),
        re.IGNORECASE,
    ),
    "ORDER_ID": re.compile(
        (
            r"(?:#|(?:ORD|ORDER|INV|INVOICE|TXN|TRANSACTION|TKT|TICKET|CASE|TRK|TRACKING|ERR|REF|ID)[-_: ]\s?)"
            r"[A-Z0-9_-]{3,20}\b"
        ),
        re.IGNORECASE,
    ),
    "DATE": re.compile(
        (
            r"\b\d{4}-\d{2}-\d{2}\b|"
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b|"
            r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*,?\s+\d{4}\b"
        ),
        re.IGNORECASE,
    ),
}


def _determine_identifier_label(entity_str: str) -> str:
    """Classifies identifier string into specific domain entity type."""
    upper = entity_str.upper()
    if "INV" in upper or "INVOICE" in upper:
        return "INVOICE_ID"
    if "TXN" in upper or "TRANS" in upper:
        return "TRANSACTION_ID"
    if "TKT" in upper or "TICKET" in upper:
        return "TICKET_ID"
    if "TRK" in upper or "TRACK" in upper:
        return "TRACKING_NUMBER"
    if "ERR" in upper:
        return "ERROR_CODE"
    if "CASE" in upper:
        return "CASE_ID"
    return "ORDER_ID"


def _detect_currency(entity_str: str) -> Optional[str]:
    """Detects currency symbol or ISO code from money entity string."""
    upper = entity_str.upper()
    if "₹" in entity_str or "RS" in upper or "INR" in upper or "RUPEES" in upper:
        return "INR"
    if "$" in entity_str or "USD" in upper or "DOLLARS" in upper:
        return "USD"
    if "€" in entity_str or "EUR" in upper:
        return "EUR"
    if "£" in entity_str or "GBP" in upper:
        return "GBP"
    if "¥" in entity_str:
        return "JPY"
    return None


def extract_rule_entities(text: str) -> List[Dict[str, Any]]:
    """
    Runs high-precision deterministic regex pattern extractors:
    - Emails: RFC 5322 regex pattern
    - Phone Numbers: International and Indian formats (+91, 10-digit mobile)
    - Money/Currency: Amounts like ₹1,500, Rs. 1500, $49.99, USD 50, €12.50
    - Identifiers: ORD-2026-1042, INV-2026-0042, TXN-10293, TKT-9982, TRK-88219, ERR-502
    - Dates: Formats like YYYY-MM-DD, DD/MM/YYYY, 12 September 2026

    Returns list of dicts with character offsets, confidence (1.0), and source ('rule').
    """
    if not text:
        return []

    entities = []

    for label, pattern in REGEX_PATTERNS.items():
        for match in pattern.finditer(text):
            entity_str = match.group().strip()
            start_idx = match.start()
            end_idx = match.end()

            final_label = label
            if label == "ORDER_ID":
                final_label = _determine_identifier_label(entity_str)

            ent_dict: Dict[str, Any] = {
                "entity": entity_str,
                "text": entity_str,
                "label": final_label,
                "type": final_label,
                "source": "rule",
                "confidence": 1.0,
                "start": start_idx,
                "end": end_idx,
                "start_char": start_idx,
                "end_char": end_idx,
            }

            if label == "MONEY":
                currency = _detect_currency(entity_str)
                if currency:
                    ent_dict["currency"] = currency

            entities.append(ent_dict)

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
