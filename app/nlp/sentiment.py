"""
Local Sentiment Analysis module for NLP Document Intelligence Gateway.

Provides high-performance, deterministic, zero-cost local sentiment analysis
tailored for customer support communications with domain-specific lexicons,
negation window parsing, and intensifier weighting.
"""

import re
from typing import Dict, Any, List, Tuple

# Customer support domain polarity lexicon
POSITIVE_LEXICON = {
    "thank": 1.0, "thanks": 1.0, "thank you": 1.2, "appreciate": 1.2, "appreciated": 1.2,
    "great": 1.5, "excellent": 1.8, "awesome": 1.8, "fantastic": 1.8, "wonderful": 1.5,
    "good": 1.0, "helpful": 1.2, "fast": 1.0, "quick": 1.0, "resolved": 1.5,
    "working": 0.8, "perfect": 1.5, "pleased": 1.2, "happy": 1.2, "satisfied": 1.5,
    "love": 1.5, "fixed": 1.2, "impressed": 1.5, "seamless": 1.2, "smooth": 1.0,
    "convenient": 1.0, "prompt": 1.2, "efficient": 1.2, "easy": 0.8, "clear": 0.8
}

NEGATIVE_LEXICON = {
    "fail": 1.2, "failed": 1.4, "failure": 1.4, "error": 1.2, "issue": 1.0,
    "problem": 1.0, "broken": 1.5, "crash": 1.5, "crashes": 1.6, "crashed": 1.6,
    "bug": 1.0, "terrible": 1.8, "horrible": 2.0, "awful": 1.8, "worst": 2.0,
    "bad": 1.2, "poor": 1.2, "unacceptable": 1.8, "frustrated": 1.5, "frustrating": 1.5,
    "annoyed": 1.2, "angry": 1.6, "upset": 1.4, "unhappy": 1.4, "disappointed": 1.5,
    "disaster": 1.8, "scam": 2.0, "fraud": 2.0, "stolen": 1.8, "charge twice": 1.8,
    "charged twice": 1.8, "double charged": 1.8, "delayed": 1.2, "late": 1.0,
    "missing": 1.2, "damaged": 1.6, "lost": 1.2, "useless": 1.6, "defective": 1.5,
    "unauthorized": 1.8, "locked": 1.4, "refused": 1.4, "denied": 1.2, "waste": 1.5,
    "freeze": 1.2, "freezes": 1.2, "frozen": 1.2, "slow": 0.8, "unresponsive": 1.4
}

NEGATIONS = {
    "not", "no", "never", "none", "neither", "nor", "cannot", "can't", "cant",
    "won't", "wont", "don't", "dont", "didn't", "didnt", "hardly", "barely",
    "scarcely", "without", "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt"
}

INTENSIFIERS = {
    "extremely": 1.8, "very": 1.4, "really": 1.3, "highly": 1.4, "super": 1.4,
    "absolutely": 1.6, "completely": 1.5, "totally": 1.5, "deeply": 1.4,
    "severely": 1.8, "incredibly": 1.6, "hugely": 1.4, "badly": 1.4
}


def analyze_sentiment_local(text: str) -> Dict[str, Any]:
    """
    Computes sentiment polarity and score (-1.0 to +1.0) using rule-based
    lexicon scoring with negation windows and intensifiers.

    Returns:
    {
        "sentiment": "positive" | "neutral" | "negative",
        "score": float  # -1.0 to +1.0
    }
    """
    if not text or not text.strip():
        return {"sentiment": "neutral", "score": 0.0}

    cleaned = text.lower()
    
    # Tokenize words while preserving phrases
    words = re.findall(r"\b\w+(?:'\w+)?\b", cleaned)
    if not words:
        return {"sentiment": "neutral", "score": 0.0}

    total_score = 0.0
    sentiment_signals = 0

    # Multi-word checks first
    for phrase, weight in POSITIVE_LEXICON.items():
        if " " in phrase and phrase in cleaned:
            total_score += weight
            sentiment_signals += 1

    for phrase, weight in NEGATIVE_LEXICON.items():
        if " " in phrase and phrase in cleaned:
            total_score -= weight
            sentiment_signals += 1

    # Word-by-word with 3-token lookback window for negation and intensifiers
    for i, word in enumerate(words):
        base_val = 0.0
        if word in POSITIVE_LEXICON and " " not in word:
            base_val = POSITIVE_LEXICON[word]
        elif word in NEGATIVE_LEXICON and " " not in word:
            base_val = -NEGATIVE_LEXICON[word]

        if base_val != 0.0:
            sentiment_signals += 1
            # Look back up to 3 tokens for intensifiers and negations
            window = words[max(0, i - 3):i]
            
            is_negated = any(w in NEGATIONS for w in window)
            multiplier = 1.0
            
            for w in window:
                if w in INTENSIFIERS:
                    multiplier *= INTENSIFIERS[w]

            if is_negated:
                base_val = -base_val * 0.8  # Invert and slightly dampen

            base_val *= multiplier
            total_score += base_val

    # Normalize score between -1.0 and +1.0 using hyperbolic tangent / bounded scaling
    if sentiment_signals == 0:
        normalized_score = 0.0
    else:
        # Scale score
        normalized_score = max(-1.0, min(1.0, total_score / (sentiment_signals + 1.5)))

    # Determine label
    if normalized_score >= 0.15:
        sentiment = "positive"
    elif normalized_score <= -0.15:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {
        "sentiment": sentiment,
        "score": round(float(normalized_score), 2)
    }
