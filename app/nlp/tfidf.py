"""
TF-IDF & Keyword Extraction module for NLP Document Intelligence Gateway.

Computes Bag-of-Words (BoW), N-gram representations, TF-IDF feature weights,
and extracts top keyword signals from customer support text using spaCy candidates
and scikit-learn statistical scoring.
"""

import re
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from app.nlp.preprocess import tokenize_sentences, clean_text
from app.nlp.linguistic import get_spacy_model


def generate_bow_and_tfidf(
    corpus: List[str], ngram_range: Tuple[int, int] = (1, 2)
) -> Dict[str, Any]:
    """
    Computes Bag of Words and TF-IDF matrices for a list of documents.
    Returns feature names and dense matrix representations.
    """
    if not corpus:
        return {"feature_names": [], "bow_matrix": [], "tfidf_matrix": []}

    cleaned_corpus = [clean_text(doc) for doc in corpus if doc and doc.strip()]
    if not cleaned_corpus:
        return {"feature_names": [], "bow_matrix": [], "tfidf_matrix": []}

    count_vec = CountVectorizer(ngram_range=ngram_range)
    bow_mat = count_vec.fit_transform(cleaned_corpus)

    tfidf_vec = TfidfVectorizer(ngram_range=ngram_range)
    tfidf_mat = tfidf_vec.fit_transform(cleaned_corpus)

    feature_names = count_vec.get_feature_names_out().tolist()

    return {
        "feature_names": feature_names,
        "bow_matrix": bow_mat.toarray().tolist(),
        "tfidf_matrix": tfidf_mat.toarray().tolist(),
    }


def compute_tfidf(
    corpus: List[str], max_features: int = 1000, ngram_range: Tuple[int, int] = (1, 2)
) -> Dict[str, Any]:
    """
    Backwards-compatible TF-IDF helper.
    """
    if not corpus:
        return {"feature_names": [], "tfidf_matrix": []}

    tfidf_vec = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    tfidf_mat = tfidf_vec.fit_transform(corpus)

    return {
        "feature_names": tfidf_vec.get_feature_names_out().tolist(),
        "tfidf_matrix": tfidf_mat.toarray().tolist(),
    }


def extract_keywords(text: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Extracts top N keywords and keyphrases from input text:
    1. Uses spaCy to extract candidate noun phrases and key single terms.
    2. Scores candidates using TfidfVectorizer fitted on a micro-corpus derived from text sentences.
    3. Returns top N keywords formatted as [{"keyword": str, "score": float}].
    """
    if not text or not text.strip():
        return []

    cleaned = clean_text(text)
    nlp = get_spacy_model()
    doc = nlp(cleaned)

    # 1. Extract candidates (noun phrases + key single terms: NOUN, PROPN, ADJ)
    candidates = set()
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip().lower()
        chunk_text = re.sub(
            r"^(the|a|an|my|our|your|this|that|these|those)\s+", "", chunk_text
        )
        if len(chunk_text) > 2 and not chunk_text.isdigit():
            candidates.add(chunk_text)

    for token in doc:
        if (
            token.pos_ in ("NOUN", "PROPN", "ADJ")
            and not token.is_stop
            and not token.is_punct
        ):
            tok_text = token.text.strip().lower()
            if len(tok_text) > 2 and not tok_text.isdigit():
                candidates.add(tok_text)

    if not candidates:
        return []

    # 2. Build micro-corpus from sentences
    sentences = tokenize_sentences(cleaned)
    if len(sentences) <= 1:
        # Create micro-corpus by splitting clauses or adding variations
        clauses = re.split(r"[,;:]", cleaned)
        micro_corpus = [c.strip() for c in clauses if c.strip()]
        if len(micro_corpus) <= 1:
            micro_corpus = [cleaned, cleaned.lower()]
    else:
        micro_corpus = sentences

    # 3. Fit TF-IDF vectorizer on micro-corpus
    try:
        tfidf_vec = TfidfVectorizer(ngram_range=(1, 3), stop_words="english")
        tfidf_matrix = tfidf_vec.fit_transform(micro_corpus)
        vocab = tfidf_vec.vocabulary_
        feature_names = tfidf_vec.get_feature_names_out()

        # Calculate mean TF-IDF score for each feature across micro-corpus
        mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
        feature_scores = {
            feature_names[i]: float(mean_tfidf[i]) for i in range(len(feature_names))
        }
    except Exception:
        feature_scores = {}

    # 4. Score candidate terms
    scored_candidates = []
    for cand in candidates:
        cand_words = cand.split()
        if cand in feature_scores:
            score = feature_scores[cand]
        else:
            # Aggregate score from constituent words
            scores = [feature_scores.get(w, 0.0) for w in cand_words]
            score = sum(scores) / len(scores) if scores else 0.0

        # Boost multi-word noun phrase keyphrases slightly
        if len(cand_words) > 1:
            score *= 1.25

        # Fallback term frequency score if TF-IDF gave 0
        if score == 0.0:
            count = len(re.findall(r"\b" + re.escape(cand) + r"\b", cleaned.lower()))
            score = count * 0.1

        scored_candidates.append({"keyword": cand, "score": round(float(score), 4)})

    # 5. Sort candidates by score descending
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Return top N
    return scored_candidates[:top_n]


def extract_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """
    Generates word n-grams from token sequence.
    """
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
