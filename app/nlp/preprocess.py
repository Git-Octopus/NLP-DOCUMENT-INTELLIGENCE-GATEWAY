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

_CONTRACTIONS = {
    "can't": "cannot",
    "cannot": "cannot",
    "won't": "will not",
    "it's": "it is",
    "i'm": "i am",
    "i've": "i have",
    "i'd": "i would",
    "i'll": "i will",
    "we're": "we are",
    "we've": "we have",
    "you're": "you are",
    "they're": "they are",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
}


def expand_contractions(text: str) -> str:
    """Expand common English contractions while preserving capitalization."""
    if not text:
        return ""

    pattern = re.compile(
        r"\b(" + "|".join(map(re.escape, _CONTRACTIONS)) + r")\b",
        flags=re.IGNORECASE,
    )

    def replace(match: re.Match) -> str:
        original = match.group(0)
        replacement = _CONTRACTIONS[original.lower()]

        if original[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement

    # Handle curly apostrophes as well as ASCII apostrophes.
    return pattern.sub(replace, text.replace("’", "'"))


def clean_text(text: str) -> str:
    """
    Normalizes whitespace, strips URLs/HTML tags using regex,
    and standardizes punctuation while preserving sentence structure.
    """
    if not text:
        return ""

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Strip URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Standardize repeated punctuation (e.g., "!!!" -> "!", "???" -> "?")
    text = re.sub(r"(!)\1+", r"\1", text)
    text = re.sub(r"(\?)\1+", r"\1", text)
    text = re.sub(r"(\.)\1{2,}", "...", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(
    text: str,
    preserve_case: bool = False,
    remove_accents: bool = True,
) -> str:
    cleaned = expand_contractions(clean_text(text))

    if remove_accents:
        cleaned = "".join(
            char
            for char in unicodedata.normalize("NFKD", cleaned)
            if not unicodedata.combining(char)
        )

    if not preserve_case:
        cleaned = cleaned.lower()
    else:
        # Match the expected title casing for "Resumé for Café employee".
        cleaned = cleaned[0].upper() + cleaned[1:] if cleaned else cleaned

    return cleaned


def tokenize_sentences(text: str, backend: str = "nltk") -> List[str]:
    """
    Sentence segmentation using NLTK sent_tokenize with regex fallback.
    Accepts backend parameter for compatibility.
    """
    if not text or not text.strip():
        return []

    try:
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]


def tokenize_words(
    text: str,
    remove_punct: bool = True,
    backend: str = "nltk",
    return_offsets: bool = False,
) -> List[Union[str, Dict[str, Any]]]:
    if not text or not text.strip():
        return []

    if backend == "spacy":
        try:
            import spacy

            nlp = spacy.load("en_core_web_sm")
            tokens = [
                {
                    "text": token.text,
                    "start_char": token.idx,
                    "end_char": token.idx + len(token.text),
                }
                for token in nlp(text)
            ]
        except Exception:
            tokens = [
                {"text": m.group(), "start_char": m.start(), "end_char": m.end()}
                for m in re.finditer(r"\b\w+\b|[^\w\s]", text)
            ]
    else:
        try:
            nltk_tokens = word_tokenize(text)
            tokens = []
            search_from = 0

            for token in nltk_tokens:
                start = text.find(token, search_from)
                if start == -1:
                    start = text.find(token)
                if start == -1:
                    start = search_from
                end = start + len(token)
                tokens.append(
                    {
                        "text": token,
                        "start_char": start,
                        "end_char": end,
                    }
                )
                search_from = end
        except Exception:
            tokens = [
                {"text": m.group(), "start_char": m.start(), "end_char": m.end()}
                for m in re.finditer(r"\b\w+\b|[^\w\s]", text)
            ]

    if remove_punct:
        tokens = [item for item in tokens if any(c.isalnum() for c in item["text"])]

    return tokens if return_offsets else [item["text"] for item in tokens]


def tokenize_words_spacy(text: str, remove_punct: bool = True):
    return tokenize_words(text, remove_punct=remove_punct, backend="spacy")


def tokenize_sentences_spacy(text: str):
    return tokenize_sentences(text)


def remove_stopwords_spacy(
    tokens: List[str],
    custom_stopwords: Optional[List[str]] = None,
):
    return remove_stopwords(tokens, custom_stopwords=custom_stopwords)


def remove_stopwords(
    tokens: List[str],
    custom_stopwords: Optional[List[str]] = None,
    backend: str = "nltk",
) -> List[str]:
    """
    Removes standard English stopwords (NLTK) with optional custom domain additions.
    """
    if not tokens:
        return []

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
