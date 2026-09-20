"""
NLP Module Package for NLP Document Intelligence Gateway.

This package contains local NLP processing modules:
- preprocess: Text normalization, tokenization, stopword removal, stemming (NLTK & spaCy)
- linguistic: POS tagging, lemmatization, dependency parsing, noun chunks
- tfidf: Bag of Words, TF-IDF vectorization, N-grams, keyword extraction
- rules: Regex patterns for rule-based entity extraction and evidence sentence mining
- ner: spaCy Named Entity Recognition combined with rule-based extraction
- sentiment: Customer support local sentiment analysis engine
- pipeline: Unified orchestrator for Dev 1 FastAPI route integration
"""

from app.nlp.preprocess import (
    clean_text,
    normalize_text,
    tokenize_sentences,
    tokenize_sentences_spacy,
    tokenize_words,
    tokenize_words_spacy,
    remove_stopwords,
    remove_stopwords_spacy,
    apply_stemming,
    stem_tokens,
    preprocess_pipeline,
)
from app.nlp.linguistic import (
    analyze_linguistics,
    lemmatize_text,
    pos_tag_text,
    extract_noun_chunks,
    extract_dependency_features,
    get_spacy_model,
)
from app.nlp.tfidf import (
    generate_bow_and_tfidf,
    compute_tfidf,
    extract_keywords,
    extract_ngrams,
)
from app.nlp.rules import (
    extract_rule_entities,
    extract_regex_entities,
    extract_evidence_sentences,
)
from app.nlp.ner import (
    extract_spacy_entities,
    extract_all_entities,
)
from app.nlp.sentiment import (
    analyze_sentiment_local,
)
from app.nlp.pipeline import (
    run_local_nlp_pipeline,
)

__all__ = [
    # Preprocessing
    "clean_text",
    "normalize_text",
    "tokenize_sentences",
    "tokenize_sentences_spacy",
    "tokenize_words",
    "tokenize_words_spacy",
    "remove_stopwords",
    "remove_stopwords_spacy",
    "apply_stemming",
    "stem_tokens",
    "preprocess_pipeline",
    # Linguistic
    "analyze_linguistics",
    "lemmatize_text",
    "pos_tag_text",
    "extract_noun_chunks",
    "extract_dependency_features",
    "get_spacy_model",
    # TF-IDF & Keywords
    "generate_bow_and_tfidf",
    "compute_tfidf",
    "extract_keywords",
    "extract_ngrams",
    # Rules
    "extract_rule_entities",
    "extract_regex_entities",
    "extract_evidence_sentences",
    # NER
    "extract_spacy_entities",
    "extract_all_entities",
    # Sentiment
    "analyze_sentiment_local",
    # Orchestrator Pipeline
    "run_local_nlp_pipeline",
]
