"""
NLP Module Package for NLP Document Intelligence Gateway.

This package contains local NLP processing modules:
- preprocess: Text normalization, tokenization, stopword removal, stemming
- linguistic: POS tagging, lemmatization, dependency parsing, noun chunks
- tfidf: Bag of Words, TF-IDF vectorization, N-grams, keyword extraction
- rules: Regex patterns for rule-based entity extraction and evidence sentence mining
- ner: spaCy Named Entity Recognition combined with rule-based extraction
- embeddings: SentenceTransformers vector embedding generator
- vector_store: Qdrant vector DB client wrapper
- topics: Topic & intent matching logic
"""

from app.nlp.preprocess import preprocess_pipeline, normalize_text, tokenize_words, remove_stopwords, stem_tokens
from app.nlp.linguistic import lemmatize_text, pos_tag_text, extract_noun_chunks
from app.nlp.tfidf import extract_keywords
from app.nlp.rules import extract_regex_entities, extract_evidence_sentences
from app.nlp.ner import extract_all_entities
from app.nlp.embeddings import get_embedding
from app.nlp.vector_store import search_similar_topics, search_duplicate_message
from app.nlp.topics import match_topic, match_intent

__all__ = [
    "preprocess_pipeline",
    "normalize_text",
    "tokenize_words",
    "remove_stopwords",
    "stem_tokens",
    "lemmatize_text",
    "pos_tag_text",
    "extract_noun_chunks",
    "extract_keywords",
    "extract_regex_entities",
    "extract_evidence_sentences",
    "extract_all_entities",
    "get_embedding",
    "search_similar_topics",
    "search_duplicate_message",
    "match_topic",
    "match_intent",
]
