"""
Linguistic Analysis module for NLP Document Intelligence Gateway.

Provides spaCy-driven linguistic processing including lemmatization,
Part-of-Speech (POS) tagging, dependency parsing, and noun chunk extraction.
Includes graceful fallback to blank English pipeline if pretrained model is unavailable.
"""

from typing import List, Tuple, Dict, Any, Optional
import spacy
import logging

logger = logging.getLogger(__name__)

_nlp_model = None


def get_spacy_model():
    """
    Loads en_core_web_sm model lazily as a cached singleton module.
    Falls back gracefully to blank English model if en_core_web_sm is unavailable.
    """
    global _nlp_model
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model 'en_core_web_sm' successfully.")
        except Exception as e:
            logger.warning(f"Could not load 'en_core_web_sm' directly ({e}). Trying package import.")
            try:
                import en_core_web_sm
                _nlp_model = en_core_web_sm.load()
            except Exception as e2:
                logger.warning(f"Falling back to spacy.blank('en') pipeline: {e2}")
                nlp = spacy.blank("en")
                if "sentencizer" not in nlp.pipe_names:
                    nlp.add_pipe("sentencizer")
                _nlp_model = nlp
    return _nlp_model


def analyze_linguistics(text: str) -> Dict[str, Any]:
    """
    Analyzes input text using spaCy and returns a comprehensive dictionary containing:
    - lemmas: List of lowercased lemmas (excluding punctuation/spaces).
    - pos_tags: List of tuples (token, coarse_pos, fine_pos).
    - dependencies: List of dicts {"token": ..., "dep": ..., "head": ...}.
    - noun_chunks: List of extracted noun phrase strings.
    """
    if not text or not text.strip():
        return {"lemmas": [], "pos_tags": [], "dependencies": [], "noun_chunks": []}

    nlp = get_spacy_model()
    doc = nlp(text)

    lemmas = [
        (token.lemma_ if token.lemma_ else token.text).lower()
        for token in doc
        if not token.is_punct and not token.is_space and any(c.isalnum() for c in token.text)
    ]

    pos_tags = [(token.text, token.pos_ or "NOUN", token.tag_ or "NN") for token in doc]

    dependencies = []
    if doc.has_annotation("DEP"):
        dependencies = [
            {"token": token.text, "dep": token.dep_, "head": token.head.text}
            for token in doc
        ]

    noun_chunks = []
    try:
        if hasattr(doc, "noun_chunks"):
            noun_chunks = [chunk.text.strip() for chunk in doc.noun_chunks]
    except Exception:
        noun_chunks = []

    return {
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "dependencies": dependencies,
        "noun_chunks": noun_chunks,
    }


def lemmatize_text(text: str) -> List[str]:
    """
    Returns list of lemmatized text tokens.
    """
    return analyze_linguistics(text)["lemmas"]


def pos_tag_text(text: str) -> List[Tuple[str, str, str]]:
    """
    Performs Part-of-Speech tagging. Returns list of tuples: (token, coarse_pos, fine_pos).
    """
    return analyze_linguistics(text)["pos_tags"]


def extract_noun_chunks(text: str) -> List[str]:
    """
    Extracts noun chunks / noun phrases.
    """
    return analyze_linguistics(text)["noun_chunks"]


def extract_dependency_features(text: str) -> List[Dict[str, Any]]:
    """
    Extracts dependency tree triples/dicts.
    """
    return analyze_linguistics(text)["dependencies"]
