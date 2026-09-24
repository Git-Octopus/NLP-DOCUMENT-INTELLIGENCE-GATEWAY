"""
Local Sentiment Analysis module for NLP Document Intelligence Gateway.

Provides high-performance, deterministic, zero-cost local sentiment analysis
tailored for customer support communications with domain-specific lexicons,
negation window parsing, intensifier weighting, and calibrated VADER fallback.
"""

import re
from typing import Dict, Any, Optional

# Customer support domain polarity lexicon
POSITIVE_LEXICON = {
    "thank": 1.0,
    "thanks": 1.0,
    "thank you": 1.2,
    "appreciate": 1.2,
    "appreciated": 1.2,
    "great": 1.5,
    "excellent": 1.8,
    "awesome": 1.8,
    "fantastic": 1.8,
    "wonderful": 1.5,
    "good": 1.0,
    "helpful": 1.2,
    "fast": 1.0,
    "quick": 1.0,
    "resolved": 1.5,
    "working": 0.8,
    "perfect": 1.5,
    "pleased": 1.2,
    "happy": 1.2,
    "satisfied": 1.5,
    "love": 1.5,
    "fixed": 1.2,
    "impressed": 1.5,
    "seamless": 1.2,
    "smooth": 1.0,
    "convenient": 1.0,
    "prompt": 1.2,
    "efficient": 1.2,
    "easy": 0.8,
    "clear": 0.8,
}

NEGATIVE_LEXICON = {
    "fail": 1.2,
    "failed": 1.4,
    "failure": 1.4,
    "error": 1.2,
    "issue": 1.0,
    "problem": 1.0,
    "broken": 1.5,
    "crash": 1.5,
    "crashes": 1.6,
    "crashed": 1.6,
    "bug": 1.0,
    "terrible": 1.8,
    "horrible": 2.0,
    "awful": 1.8,
    "worst": 2.0,
    "bad": 1.2,
    "poor": 1.2,
    "unacceptable": 1.8,
    "frustrated": 1.5,
    "frustrating": 1.5,
    "annoyed": 1.2,
    "angry": 1.6,
    "upset": 1.4,
    "unhappy": 1.4,
    "disappointed": 1.5,
    "disaster": 1.8,
    "scam": 2.0,
    "scammer": 2.0,
    "scammers": 2.0,
    "scamming": 2.0,
    "fraud": 2.0,
    "stolen": 1.8,
    "charge twice": 1.8,
    "charged twice": 1.8,
    "double charged": 1.8,
    "delayed": 1.2,
    "late": 1.0,
    "missing": 1.2,
    "damaged": 1.6,
    "lost": 1.2,
    "useless": 1.6,
    "defective": 1.5,
    "unauthorized": 1.8,
    "locked": 1.4,
    "refused": 1.4,
    "denied": 1.2,
    "waste": 1.5,
    "freeze": 1.2,
    "freezes": 1.2,
    "frozen": 1.2,
    "slow": 0.8,
    "unresponsive": 1.4,
    "unpaid": 1.4,
    "not received": 1.5,
    "haven't received": 1.5,
    # Profanities, insults, and vulgar frustration
    "motherfucker": 2.5,
    "motherfuckers": 2.5,
    "motherfucking": 2.5,
    "fuck": 2.2,
    "fucking": 2.2,
    "fucked": 2.2,
    "fucker": 2.2,
    "shit": 2.0,
    "shitty": 2.0,
    "bullshit": 2.2,
    "bitch": 2.0,
    "bastard": 2.0,
    "asshole": 2.0,
    "crap": 1.6,
    "crappy": 1.6,
    "damn": 1.4,
    "damned": 1.4,
    "hell": 1.4,
    "pissed": 1.8,
    "pissing": 1.8,
    "suck": 1.6,
    "sucks": 1.8,
    "sucking": 1.6,
    "idiot": 2.0,
    "idiots": 2.0,
    "idiotic": 2.0,
    "stupid": 1.8,
    "dumb": 1.6,
    "pathetic": 2.0,
    "garbage": 1.8,
    "trash": 1.8,
    "junk": 1.6,
    "rip off": 2.0,
    "ripoff": 2.0,
    "cheat": 2.0,
    "cheated": 2.0,
    "cheating": 2.0,
    "liar": 2.0,
    "liars": 2.0,
    "hate": 1.8,
    "hated": 1.8,
    "hating": 1.8,
    "disgusting": 2.0,
    "worthless": 2.0,
    "horrific": 2.0,
    "nonsense": 1.6,
    "ridiculous": 1.8,
    "annoying": 1.5,
    "infuriating": 2.0,
    "mad": 1.5,
    "outrageous": 1.8,
}

NEGATIONS = {
    "not",
    "no",
    "never",
    "none",
    "neither",
    "nor",
    "cannot",
    "can't",
    "cant",
    "won't",
    "wont",
    "don't",
    "dont",
    "didn't",
    "didnt",
    "hardly",
    "barely",
    "scarcely",
    "without",
    "isn't",
    "isnt",
    "aren't",
    "arent",
    "wasn't",
    "wasnt",
}

INTENSIFIERS = {
    "extremely": 1.8,
    "very": 1.4,
    "really": 1.3,
    "highly": 1.4,
    "super": 1.4,
    "absolutely": 1.6,
    "completely": 1.5,
    "totally": 1.5,
    "deeply": 1.4,
    "severely": 1.8,
    "incredibly": 1.6,
    "hugely": 1.4,
    "badly": 1.4,
}

_vader_instance: Optional[Any] = None
_vader_checked = False


def _get_vader_analyzer() -> Optional[Any]:
    """
    Lazily loads NLTK VADER sentiment analyzer calibrated for customer support.
    Neutralizes domain nouns ('support', 'ticket', 'order', etc.) so that
    operational questions remain truly neutral.
    """
    global _vader_instance, _vader_checked
    if _vader_checked:
        return _vader_instance

    _vader_checked = True
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        try:
            analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            analyzer = SentimentIntensityAnalyzer()

        # Neutralize common customer-support nouns that VADER treats as positive
        domain_neutral = [
            "support",
            "supports",
            "supported",
            "supporting",
            "order",
            "orders",
            "ticket",
            "tickets",
            "account",
            "service",
            "services",
        ]
        for term in domain_neutral:
            analyzer.lexicon[term] = 0.0

        # Enhance specific domain words
        analyzer.lexicon["resolved"] = 2.0
        analyzer.lexicon["helpful"] = 1.8
        analyzer.lexicon["crashed"] = -2.2
        analyzer.lexicon["unacceptable"] = -2.5

        _vader_instance = analyzer
    except Exception:
        _vader_instance = None

    return _vader_instance


def _analyze_rule_based_fallback(text: str) -> Dict[str, Any]:
    """Deterministic fallback lexicon scorer with negation and intensifier windows."""
    cleaned = text.lower()
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

    # Word-by-word with lookback window
    for i, word in enumerate(words):
        base_val = 0.0
        if word in POSITIVE_LEXICON and " " not in word:
            base_val = POSITIVE_LEXICON[word]
        elif word in NEGATIVE_LEXICON and " " not in word:
            base_val = -NEGATIVE_LEXICON[word]

        if base_val != 0.0:
            sentiment_signals += 1
            window = words[max(0, i - 3) : i]
            is_negated = any(w in NEGATIONS for w in window)
            multiplier = 1.0

            for w in window:
                if w in INTENSIFIERS:
                    multiplier *= INTENSIFIERS[w]

            if is_negated:
                base_val = -base_val * 0.8

            base_val *= multiplier
            total_score += base_val

    if sentiment_signals == 0:
        normalized_score = 0.0
    else:
        normalized_score = max(-1.0, min(1.0, total_score / (sentiment_signals + 1.5)))

    if normalized_score >= 0.15:
        sentiment = "positive"
    elif normalized_score <= -0.15:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {"sentiment": sentiment, "score": round(float(normalized_score), 2)}


def analyze_sentiment_local(text: str) -> Dict[str, Any]:
    """
    Computes sentiment polarity and score (-1.0 to +1.0).
    Uses domain-calibrated VADER when available, backed by customer support
    multi-word phrasing and local lexicon fallback.

    Returns:
    {
        "sentiment": "positive" | "neutral" | "negative",
        "score": float  # -1.0 to +1.0
    }
    """
    if not text or not text.strip():
        return {"sentiment": "neutral", "score": 0.0}

    vader = _get_vader_analyzer()
    if vader is not None:
        cleaned_lower = text.lower()
        # Add bias for domain-specific multi-word phrases
        phrase_bias = 0.0
        for phrase, weight in NEGATIVE_LEXICON.items():
            if " " in phrase and phrase in cleaned_lower:
                phrase_bias -= weight * 0.35
        for phrase, weight in POSITIVE_LEXICON.items():
            if " " in phrase and phrase in cleaned_lower:
                phrase_bias += weight * 0.35

        scores = vader.polarity_scores(text)
        compound = scores.get("compound", 0.0) + phrase_bias
        compound = max(-1.0, min(1.0, compound))

        if compound >= 0.15:
            sentiment = "positive"
        elif compound <= -0.15:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {"sentiment": sentiment, "score": round(float(compound), 2)}

    return _analyze_rule_based_fallback(text)
