"""
Unified Orchestrator Pipeline for NLP Document Intelligence Gateway.

Provides a single entry-point function `run_local_nlp_pipeline` for FastAPI routes
and downstream modules to execute preprocessing, linguistic analysis, keyword extraction,
entity recognition, and local sentiment classification.
"""

from typing import Dict, Any
from app.nlp.preprocess import clean_text, tokenize_words, tokenize_sentences
from app.nlp.linguistic import analyze_linguistics
from app.nlp.tfidf import extract_keywords
from app.nlp.ner import extract_all_entities
from app.nlp.sentiment import analyze_sentiment_local


def run_local_nlp_pipeline(text: str) -> Dict[str, Any]:
    """
    Executes local text processing and traditional ML tasks:
    - Text cleaning & normalization
    - Sentence & word tokenization
    - spaCy linguistic analysis (lemmas, POS tags, noun chunks)
    - TF-IDF keyword extraction
    - Combined rule-based & spaCy entity recognition
    - Lexicon-based local sentiment analysis

    Output schema:
    {
        "cleaned_text": str,
        "tokens": list[str],
        "sentences": list[str],
        "lemmas": list[str],
        "pos_tags": list[dict], # [{"token": str, "coarse_pos": str, "fine_pos": str}]
        "noun_chunks": list[str],
        "keywords": list[dict], # [{"keyword": str, "score": float}]
        "entities": list[dict], # [{"entity": str, "label": str, "source": str, "start": int, "end": int}]
        "sentiment": dict       # {"sentiment": str, "score": float}
    }
    """
    if not text:
        return {
            "cleaned_text": "",
            "tokens": [],
            "sentences": [],
            "lemmas": [],
            "pos_tags": [],
            "noun_chunks": [],
            "keywords": [],
            "entities": [],
            "sentiment": {"sentiment": "neutral", "score": 0.0},
        }

    cleaned_text = clean_text(text)
    tokens = tokenize_words(cleaned_text, remove_punct=True)
    sentences = tokenize_sentences(cleaned_text)

    linguistic_res = analyze_linguistics(cleaned_text)
    lemmas = linguistic_res["lemmas"]

    formatted_pos_tags = [
        {"token": tag_tuple[0], "coarse_pos": tag_tuple[1], "fine_pos": tag_tuple[2]}
        for tag_tuple in linguistic_res["pos_tags"]
    ]

    noun_chunks = linguistic_res["noun_chunks"]
    keywords = extract_keywords(cleaned_text, top_n=5)
    entities = extract_all_entities(text)
    sentiment = analyze_sentiment_local(cleaned_text)

    return {
        "cleaned_text": cleaned_text,
        "tokens": tokens,
        "sentences": sentences,
        "lemmas": lemmas,
        "pos_tags": formatted_pos_tags,
        "noun_chunks": noun_chunks,
        "keywords": keywords,
        "entities": entities,
        "sentiment": sentiment,
    }
