"""
Unified Orchestrator Pipeline for NLP Document Intelligence Gateway.

Provides a unified CoreNLPService orchestration class and entry-point function
`run_local_nlp_pipeline` for FastAPI routes and evaluation runners to execute
preprocessing, linguistic analysis, keyword extraction, entity recognition,
and local sentiment scoring.
"""

import logging
from typing import Dict, Any, List, Optional, Union

from app.config import settings
from app.nlp.preprocess import (
    clean_text,
    get_entity_preserving_text,
    tokenize_words,
    tokenize_sentences,
    preprocess_pipeline,
)
from app.nlp.linguistic import analyze_linguistics
from app.nlp.tfidf import (
    extract_keywords,
    compute_bow,
    compute_tfidf,
    extract_ngrams,
    extract_all_ngrams,
)
from app.nlp.rules import extract_rule_entities, extract_evidence_sentences
from app.nlp.ner import extract_spacy_entities, extract_all_entities
from app.nlp.sentiment import analyze_sentiment_local

logger = logging.getLogger("gateway.nlp")


class CoreNLPService:
    """
    Central orchestration service for local NLP operations:
    - Preprocessing & normalization
    - spaCy linguistic analysis (lemmas, POS tags, dependencies, noun chunks)
    - Statistical NLP (N-grams, Bag of Words, TF-IDF)
    - Hybrid keyword extraction
    - Rule-based deterministic entity extraction
    - spaCy NER with span priority deduplication
    - Domain-tailored local sentiment analysis
    - Evidence sentence mining
    """

    def __init__(
        self,
        max_keywords: int = 10,
        max_ngrams: int = 50,
        max_text_length: int = 50000,
    ):
        self.max_keywords = max_keywords
        self.max_ngrams = max_ngrams
        self.max_text_length = max_text_length

    def preprocess(self, text: str) -> Dict[str, Any]:
        """Runs full text preprocessing suite."""
        return preprocess_pipeline(text)

    def linguistic_analysis(self, text: str) -> Dict[str, Any]:
        """Runs spaCy linguistic analysis."""
        return analyze_linguistics(text)

    def ngrams(self, tokens: List[str], n: int = 2) -> List[str]:
        """Generates word n-grams from a token list."""
        return extract_ngrams(tokens, n=n)

    def all_ngrams(self, tokens: List[str], max_n: int = 3) -> Dict[str, List[str]]:
        """Generates structured unigrams, bigrams, and trigrams."""
        return extract_all_ngrams(tokens, max_n=max_n, max_per_n=self.max_ngrams)

    def bow(self, corpus: Union[str, List[str]]) -> Dict[str, Any]:
        """Computes Bag of Words representation."""
        return compute_bow(corpus)

    def tfidf(
        self, corpus: Union[str, List[str]], max_features: int = 1000
    ) -> Dict[str, Any]:
        """Computes TF-IDF representation and term ranking."""
        return compute_tfidf(corpus, max_features=max_features)

    def keywords(
        self,
        text: str,
        top_n: int = 5,
        reference_corpus: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Extracts top keywords using hybrid TF-IDF and noun chunk candidate ranking."""
        return extract_keywords(text, top_n=top_n, reference_corpus=reference_corpus)

    def rules(self, text: str) -> List[Dict[str, Any]]:
        """Runs deterministic rule extractors."""
        return extract_rule_entities(text)

    def ner(self, text: str) -> List[Dict[str, Any]]:
        """Runs spaCy statistical NER."""
        return extract_spacy_entities(text)

    def merge_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extracts combined rule and spaCy entities with deterministic priority."""
        return extract_all_entities(text)

    def sentiment(self, text: str) -> Dict[str, Any]:
        """Runs local customer support sentiment analysis."""
        return analyze_sentiment_local(text)

    def evidence(
        self, text: str, keywords: List[str], max_sentences: int = 3
    ) -> List[str]:
        """Extracts key justification sentences based on keyword density."""
        return extract_evidence_sentences(
            text, keywords=keywords, max_sentences=max_sentences
        )

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Executes the complete local NLP pipeline on the given input text.
        Guarantees non-crashing execution and schema compliance.
        """
        if not text or not text.strip():
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
                "evidence_sentences": [],
            }

        # Truncate overly long inputs to protect server resources
        original_length = len(text)
        if original_length > self.max_text_length:
            logger.warning(
                f"Input text length ({original_length}) exceeded max length ({self.max_text_length}). Truncating."
            )
            text = text[: self.max_text_length]

        raw_preserving_text = get_entity_preserving_text(text)
        cleaned_text = clean_text(raw_preserving_text)
        tokens = tokenize_words(cleaned_text, remove_punct=True)
        sentences = tokenize_sentences(cleaned_text)

        linguistic_res = self.linguistic_analysis(cleaned_text)
        lemmas = linguistic_res.get("lemmas", [])

        formatted_pos_tags = [
            {
                "token": tag_tuple[0],
                "coarse_pos": tag_tuple[1],
                "fine_pos": tag_tuple[2],
            }
            for tag_tuple in linguistic_res.get("pos_tags", [])
        ]

        noun_chunks = linguistic_res.get("noun_chunks", [])
        keyword_dicts = self.keywords(cleaned_text, top_n=self.max_keywords)
        entities = self.merge_entities(raw_preserving_text)
        sentiment_res = self.sentiment(raw_preserving_text)

        kw_list = [k["keyword"] for k in keyword_dicts]
        evidence = self.evidence(raw_preserving_text, keywords=kw_list, max_sentences=3)
        if not evidence and sentences:
            evidence = [sentences[0]]

        return {
            "cleaned_text": cleaned_text,
            "tokens": tokens,
            "sentences": sentences,
            "lemmas": lemmas,
            "pos_tags": formatted_pos_tags,
            "noun_chunks": noun_chunks,
            "keywords": keyword_dicts,
            "entities": entities,
            "sentiment": sentiment_res,
            "evidence_sentences": evidence,
        }


# Global CoreNLPService singleton instance
_core_nlp_service = CoreNLPService(
    max_keywords=getattr(settings, "MAX_KEYWORDS", 10),
    max_ngrams=getattr(settings, "MAX_NGRAMS", 50),
    max_text_length=getattr(settings, "MAX_TEXT_LENGTH", 50000),
)


def get_core_nlp_service() -> CoreNLPService:
    """Returns singleton CoreNLPService instance."""
    return _core_nlp_service


def run_local_nlp_pipeline(text: str) -> Dict[str, Any]:
    """
    Backwards-compatible top-level executor for the local NLP pipeline.
    Used by FastAPI endpoints, test suites, and evaluation runners.
    """
    return _core_nlp_service.analyze(text)
