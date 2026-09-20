"""
Pre-processing module for NLP Document Intelligence Gateway.

Provides text cleaning, normalization, sentence tokenization, word tokenization,
stopword removal, and stemming routines.
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Union
import nltk

# Auto-download required NLTK resources safely
for resource in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(
            f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}"
        )
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer

CONTRACTION_PATTERNS = [
    (r"\bcan't\b", "cannot"),
    (r"\bcan\t\b", "cannot"),
    (r"\bwon't\b", "will not"),
    (r"\bwon\t\b", "will not"),
    (r"\bshan't\b", "shall not"),
    (r"(\w+)'ve\b", r"\1 have"),
    (r"(\w+)'n't\b", r"\1 not"),
    (r"(\w+)n't\b", r"\1 not"),
    (r"(\w+)'re\b", r"\1 are"),
    (r"(\w+)'d\b", r"\1 would"),
    (r"(\w+)'ll\b", r"\1 will"),
    (r"(\b[Ii])'m\b", r"\1 am"),
    (r"(\b[Ii])'ve\b", r"\1 have"),
    (r"(\b[Ii])'ll\b", r"\1 will"),
    (r"(\b[Ii])'d\b", r"\1 would"),
    (r"(\b[Hh]e|[Ss]he|[It]t|[Tt]hat|[Ww]hat|[Hh]ow|[Ww]ho|[Ww]here)'s\b", r"\1 is"),
]


def expand_contractions(text: str) -> str:
    """
    Expands common English contractions using regex patterns.
    Standardizes smart apostrophes prior to matching.
    """
    if not text:
        return ""
    text = text.replace("’", "'")
    for pattern, replacement in CONTRACTION_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def clean_text(
    text: str, remove_accents: bool = True, expand_contraction_flag: bool = True
) -> str:
    """
    Normalizes whitespace, strips URLs/HTML tags using regex,
    expands contractions, normalizes Unicode accents, and standardizes punctuation
    while preserving sentence structure.
    """
    if not text:
        return ""

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Strip URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Standardize curly/smart apostrophes
    text = text.replace("’", "'")

    # Expand English contractions if requested
    if expand_contraction_flag:
        text = expand_contractions(text)

    # Remove accents/diacritics if requested
    if remove_accents:
        nfkd_form = unicodedata.normalize("NFKD", text)
        text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    # Standardize repeated noise symbols (e.g. "***", "===")
    text = re.sub(r"[\*=\+\~]{2,}", " ", text)

    # Standardize repeated punctuation (e.g., "!!!" -> "!", "???" -> "?")
    text = re.sub(r"(!)\1+", r"\1", text)
    text = re.sub(r"(\?)\1+", r"\1", text)
    text = re.sub(r"(\.)\1{2,}", "...", text)

    # Filter non-printable control characters while keeping standard ASCII & line breaks
    text = re.sub(r"[^\x20-\x7E\n\r\t]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(
    text: str,
    preserve_case: bool = False,
    remove_accents: bool = True,
    expand_contraction_flag: bool = True,
) -> str:
    """
    Normalizes text for downstream NLP analysis (casing, cleaning, contractions, accents).
    """
    cleaned = clean_text(
        text,
        remove_accents=remove_accents,
        expand_contraction_flag=expand_contraction_flag,
    )
    if not preserve_case:
        cleaned = cleaned.lower()
    return cleaned


def tokenize_sentences(text: str, backend: str = "nltk") -> List[str]:
    """
    Sentence segmentation using NLTK or spaCy with regex fallback.
    """
    if not text or not text.strip():
        return []

    if backend.lower() == "spacy":
        return tokenize_sentences_spacy(text)

    try:
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception:
        # Fallback regex split on sentence boundaries (. ! ?)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]


def tokenize_sentences_spacy(text: str) -> List[str]:
    """
    Sentence segmentation using spaCy.
    """
    if not text or not text.strip():
        return []
    try:
        from app.nlp.linguistic import get_spacy_model
        nlp = get_spacy_model()
        doc = nlp(text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]
    except Exception:
        return tokenize_sentences(text, backend="nltk")


def tokenize_words(
    text: str, remove_punct: bool = True, return_offsets: bool = False, backend: str = "nltk"
) -> Union[List[str], List[Dict[str, Any]]]:
    """
    Word tokenization using NLTK or spaCy with regex fallback.
    Optionally strips punctuation tokens and returns token character offsets.
    """
    if not text or not text.strip():
        return []

    if backend.lower() == "spacy" and not return_offsets:
        return tokenize_words_spacy(text, remove_punct=remove_punct)

    if return_offsets:
        pattern = re.compile(r"\w+|[^\w\s]")
        offset_tokens = []
        for match in pattern.finditer(text):
            token_str = match.group()
            if remove_punct and not any(c.isalnum() for c in token_str):
                continue
            offset_tokens.append(
                {
                    "text": token_str,
                    "start_char": match.start(),
                    "end_char": match.end(),
                }
            )
        return offset_tokens

    try:
        tokens = word_tokenize(text)
    except Exception:
        # Fallback regex tokenization
        tokens = re.findall(r"\b\w+\b|[^\w\s]", text)

    if remove_punct:
        tokens = [t for t in tokens if any(c.isalnum() for c in t)]

    return tokens


def tokenize_words_spacy(text: str, remove_punct: bool = True) -> List[str]:
    """
    Word tokenization using spaCy.
    """
    if not text or not text.strip():
        return []
    try:
        from app.nlp.linguistic import get_spacy_model
        nlp = get_spacy_model()
        doc = nlp(text)
        tokens = [
            t.text for t in doc
            if not (remove_punct and (t.is_punct or not any(c.isalnum() for c in t.text)))
            and not t.is_space
        ]
        return tokens
    except Exception:
        return tokenize_words(text, remove_punct=remove_punct, backend="nltk")


def remove_stopwords_spacy(
    tokens: List[str], custom_stopwords: Optional[List[str]] = None
) -> List[str]:
    """
    Removes standard English stopwords using spaCy's stopword list.
    """
    if not tokens:
        return []
    try:
        from app.nlp.linguistic import get_spacy_model
        nlp = get_spacy_model()
        stop_words = set(nlp.Defaults.stop_words)
    except Exception:
        stop_words = set()

    if custom_stopwords:
        stop_words.update(s.lower() for s in custom_stopwords)

    return [t for t in tokens if t.lower() not in stop_words]


def remove_stopwords(
    tokens: List[str], custom_stopwords: Optional[List[str]] = None, backend: str = "nltk"
) -> List[str]:
    """
    Removes standard English stopwords (NLTK or spaCy) with optional custom domain additions.
    """
    if not tokens:
        return []

    if backend.lower() == "spacy":
        return remove_stopwords_spacy(tokens, custom_stopwords=custom_stopwords)

    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        # Fallback standard English stopword list
        stop_words = {
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "having",
            "do",
            "does",
            "did",
            "doing",
            "a",
            "an",
            "the",
            "and",
            "but",
            "if",
            "or",
            "because",
            "as",
            "until",
            "while",
            "of",
            "at",
            "by",
            "for",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "to",
            "from",
            "up",
            "down",
            "in",
            "out",
            "on",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "s",
            "t",
            "can",
            "will",
            "just",
            "don",
            "should",
            "now",
        }

    if custom_stopwords:
        stop_words.update(s.lower() for s in custom_stopwords)

    return [t for t in tokens if t.lower() not in stop_words]


def apply_stemming(tokens: List[str], stemmer_type: str = "snowball") -> List[str]:
    """
    Applies Porter or Snowball stemming via NLTK.
    """
    if not tokens:
        return []

    if stemmer_type.lower() == "porter":
        stemmer = PorterStemmer()
    else:
        stemmer = SnowballStemmer("english")

    return [stemmer.stem(t) for t in tokens]


def stem_tokens(tokens: List[str], stemmer_type: str = "porter") -> List[str]:
    """
    Backwards-compatible alias for apply_stemming.
    """
    return apply_stemming(tokens, stemmer_type=stemmer_type)


def preprocess_pipeline(text: str) -> Dict[str, Any]:
    """
    Runs the complete preprocessing workflow on raw text and returns intermediate outputs.
    """
    cleaned = clean_text(text)
    sentences = tokenize_sentences(cleaned)
    words = tokenize_words(cleaned, remove_punct=True)
    filtered_words = remove_stopwords(words)
    stemmed = apply_stemming(filtered_words, stemmer_type="snowball")

    return {
        "cleaned_text": cleaned,
        "sentences": sentences,
        "raw_tokens": words,
        "filtered_tokens": filtered_words,
        "stemmed_tokens": stemmed,
    }
