"""
Linguistic Analysis module for NLP Document Intelligence Gateway.

Provides spaCy-driven linguistic processing including lemmatization,
Part-of-Speech (POS) tagging, dependency parsing, and noun chunk extraction.
"""

from typing import List, Tuple, Dict, Any
import spacy

_nlp_model = None


def get_spacy_model():
    """
    Loads en_core_web_sm model lazily as a cached singleton module.
    """
    global _nlp_model
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load("en_core_web_sm")
        except OSError:
            try:
                from spacy.cli import download

                download("en_core_web_sm")
                _nlp_model = spacy.load("en_core_web_sm")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load or download spaCy model 'en_core_web_sm': {e}"
                )
    return _nlp_model


def analyze_linguistics(text: str) -> Dict[str, Any]:
    """
    Analyzes input text using spaCy and returns a comprehensive dictionary containing:
    - lemmas: List of lowercased lemmas (excluding punctuation/spaces).
    - pos_tags: List of tuples (token, coarse_pos, fine_pos).
    - dependencies: List of dicts {"token": ..., "dep": ..., "head": ...}.
    - noun_chunks: List of extracted noun phrase strings.
    - tokens_info: List of dicts {"text": ..., "lemma": ..., "pos": ..., "tag": ..., "dep": ..., "head": ...}.
    """
    if not text or not text.strip():
        return {
            "lemmas": [],
            "pos_tags": [],
            "dependencies": [],
            "noun_chunks": [],
            "tokens_info": [],
        }

    try:
        nlp = get_spacy_model()
        doc = nlp(text)

        lemmas = [
            token.lemma_.lower()
            for token in doc
            if not token.is_punct and not token.is_space
        ]

        pos_tags = [(token.text, token.pos_, token.tag_) for token in doc]

        dependencies = [
            {"token": token.text, "dep": token.dep_, "head": token.head.text}
            for token in doc
        ]

        noun_chunks = [chunk.text.strip() for chunk in doc.noun_chunks]

        tokens_info = [
            {
                "text": token.text,
                "lemma": token.lemma_.lower(),
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
                "head": token.head.text,
            }
            for token in doc
        ]

        return {
            "lemmas": lemmas,
            "pos_tags": pos_tags,
            "dependencies": dependencies,
            "noun_chunks": noun_chunks,
            "tokens_info": tokens_info,
        }
    except Exception:
        # Fallback if spaCy processing encounters an unexpected issue
        from app.nlp.preprocess import tokenize_words

        raw_tokens = tokenize_words(text, remove_punct=True)
        return {
            "lemmas": [t.lower() for t in raw_tokens],
            "pos_tags": [(t, "NOUN", "NN") for t in raw_tokens],
            "dependencies": [
                {"token": t, "dep": "ROOT", "head": t} for t in raw_tokens
            ],
            "noun_chunks": [],
            "tokens_info": [],
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
