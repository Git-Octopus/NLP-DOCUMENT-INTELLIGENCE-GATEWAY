"""
TF-IDF & Keyword Extraction module for NLP Document Intelligence Gateway.

Computes Bag-of-Words (BoW), N-gram representations, TF-IDF feature weights,
and extracts top keyword signals from customer support text using spaCy candidates
and scikit-learn statistical scoring.
"""

import re
from typing import List, Dict, Any, Tuple, Union, Optional
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from app.nlp.preprocess import tokenize_sentences, clean_text
from app.nlp.linguistic import get_spacy_model


def generate_bow_and_tfidf(
    corpus: Union[str, List[str]], ngram_range: Tuple[int, int] = (1, 2)
) -> Dict[str, Any]:
    """
    Computes Bag of Words and TF-IDF matrices for a string or list of documents.
    Returns feature names, vocabulary, total counts, and dense matrix representations.
    """
    if isinstance(corpus, str):
        corpus = [corpus]

    if not corpus:
        return {
            "feature_names": [],
            "vocabulary": [],
            "counts": [],
            "bow_matrix": [],
            "tfidf_matrix": [],
        }

    cleaned_corpus = [clean_text(doc) for doc in corpus if doc and doc.strip()]
    if not cleaned_corpus:
        return {
            "feature_names": [],
            "vocabulary": [],
            "counts": [],
            "bow_matrix": [],
            "tfidf_matrix": [],
        }

    count_vec = CountVectorizer(ngram_range=ngram_range)
    bow_mat = count_vec.fit_transform(cleaned_corpus)

    tfidf_vec = TfidfVectorizer(ngram_range=ngram_range)
    tfidf_mat = tfidf_vec.fit_transform(cleaned_corpus)

    feature_names = count_vec.get_feature_names_out().tolist()
    counts = np.asarray(bow_mat.sum(axis=0)).flatten().tolist()

    return {
        "feature_names": feature_names,
        "vocabulary": feature_names,
        "counts": counts,
        "bow_matrix": bow_mat.toarray().tolist(),
        "tfidf_matrix": tfidf_mat.toarray().tolist(),
    }


def compute_bow(
    corpus: Union[str, List[str]], max_features: int = 1000
) -> Dict[str, Any]:
    """
    Computes Bag-of-Words representation using CountVectorizer.
    Returns vocabulary, counts, and feature names.
    """
    if isinstance(corpus, str):
        corpus = [corpus]

    cleaned_corpus = [clean_text(doc) for doc in corpus if doc and doc.strip()]
    if not cleaned_corpus:
        return {"feature_names": [], "vocabulary": [], "counts": [], "bow_matrix": []}

    vec = CountVectorizer(max_features=max_features)
    mat = vec.fit_transform(cleaned_corpus)
    feature_names = vec.get_feature_names_out().tolist()
    counts = np.asarray(mat.sum(axis=0)).flatten().tolist()

    return {
        "feature_names": feature_names,
        "vocabulary": feature_names,
        "counts": counts,
        "bow_matrix": mat.toarray().tolist(),
    }


def compute_tfidf(
    corpus: Union[str, List[str]],
    max_features: int = 1000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Dict[str, Any]:
    """
    Computes TF-IDF weights and ranked terms.
    Distinguishes document-level representation from corpus-trained TF-IDF.
    """
    if isinstance(corpus, str):
        corpus = [corpus]

    if not corpus:
        return {"feature_names": [], "tfidf_matrix": [], "ranked_terms": []}

    cleaned_corpus = [clean_text(doc) for doc in corpus if doc and doc.strip()]
    if not cleaned_corpus:
        return {"feature_names": [], "tfidf_matrix": [], "ranked_terms": []}

    tfidf_vec = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    tfidf_mat = tfidf_vec.fit_transform(cleaned_corpus)
    feature_names = tfidf_vec.get_feature_names_out().tolist()

    # Calculate mean score per feature across corpus
    mean_scores = np.asarray(tfidf_mat.mean(axis=0)).flatten()
    ranked_terms = [
        {"term": feature_names[i], "score": round(float(mean_scores[i]), 4)}
        for i in range(len(feature_names))
    ]
    ranked_terms.sort(key=lambda x: x["score"], reverse=True)

    return {
        "feature_names": feature_names,
        "tfidf_matrix": tfidf_mat.toarray().tolist(),
        "ranked_terms": ranked_terms,
    }


def extract_keywords(
    text: str, top_n: int = 5, reference_corpus: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Extracts top N keywords and keyphrases from input text using a hybrid approach:
    1. spaCy candidate extraction (noun chunks, NOUN, PROPN, ADJ+NOUN).
    2. Statistical scoring via TF-IDF (fitted on reference_corpus if provided,
       otherwise on a sentence micro-corpus derived from the input text).
    3. Returns top N ranked keywords formatted as:
       [{"keyword": str, "term": str, "score": float, "source": "noun_phrase"|"tfidf"}]
    """
    if not text or not text.strip():
        return []

    cleaned = clean_text(text)
    nlp = get_spacy_model()
    doc = nlp(cleaned)

    # 1. Extract candidates (noun phrases + single terms)
    noun_phrase_candidates = set()
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip().lower()
        chunk_text = re.sub(
            r"^(the|a|an|my|our|your|this|that|these|those)\s+", "", chunk_text
        )
        if len(chunk_text) > 2 and not chunk_text.isdigit():
            noun_phrase_candidates.add(chunk_text)

    token_candidates = set()
    for token in doc:
        if (
            token.pos_ in ("NOUN", "PROPN", "ADJ")
            and not token.is_stop
            and not token.is_punct
        ):
            tok_text = token.text.strip().lower()
            if len(tok_text) > 2 and not tok_text.isdigit():
                token_candidates.add(tok_text)

    candidates = noun_phrase_candidates.union(token_candidates)
    if not candidates:
        return []

    # 2. Build corpus for TF-IDF: use reference_corpus if supplied, else sentence micro-corpus
    if reference_corpus and len(reference_corpus) > 1:
        micro_corpus = [clean_text(d) for d in reference_corpus if d.strip()]
        if cleaned not in micro_corpus:
            micro_corpus.append(cleaned)
    else:
        sentences = tokenize_sentences(cleaned)
        if len(sentences) <= 1:
            clauses = re.split(r"[,;:]", cleaned)
            micro_corpus = [c.strip() for c in clauses if c.strip()]
            if len(micro_corpus) <= 1:
                micro_corpus = [cleaned, cleaned.lower()]
        else:
            micro_corpus = sentences

    # 3. Fit TF-IDF on the corpus
    try:
        tfidf_vec = TfidfVectorizer(ngram_range=(1, 3), stop_words="english")
        tfidf_matrix = tfidf_vec.fit_transform(micro_corpus)
        feature_names = tfidf_vec.get_feature_names_out()

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
            scores = [feature_scores.get(w, 0.0) for w in cand_words]
            score = sum(scores) / len(scores) if scores else 0.0

        source = "noun_phrase" if cand in noun_phrase_candidates else "tfidf"

        # Boost multi-word noun phrases slightly
        if len(cand_words) > 1:
            score *= 1.25

        # Fallback term frequency score if TF-IDF score is 0
        if score == 0.0:
            count = len(re.findall(r"\b" + re.escape(cand) + r"\b", cleaned.lower()))
            score = count * 0.1

        scored_candidates.append(
            {
                "keyword": cand,
                "term": cand,
                "score": round(float(score), 4),
                "source": source,
            }
        )

    # 5. Sort candidates descending by score
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)

    return scored_candidates[:top_n]


def extract_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """
    Generates word n-grams from a token sequence.
    Supports unigram (n=1), bigram (n=2), trigram (n=3), etc.
    """
    if not tokens or len(tokens) < n or n < 1:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def extract_all_ngrams(
    tokens: List[str], max_n: int = 3, max_per_n: int = 30
) -> Dict[str, List[str]]:
    """
    Generates structured unigrams, bigrams, and trigrams with reasonable limits.
    """
    names = {1: "unigrams", 2: "bigrams", 3: "trigrams"}
    res = {}
    for n in range(1, max_n + 1):
        label = names.get(n, f"{n}-grams")
        res[label] = extract_ngrams(tokens, n=n)[:max_per_n]
    return res
