"""
Pre-processing module for NLP Document Intelligence Gateway.

Provides text cleaning, normalization, sentence tokenization, word tokenization,
stopword removal, and stemming routines.
"""

import re
from typing import List, Dict, Any, Optional
import nltk

# Auto-download required NLTK resources safely
for resource in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer

def clean_text(text: str) -> str:
    """
    Normalizes whitespace, strips URLs/HTML tags using regex,
    and standardizes punctuation while preserving sentence structure.
    """
    if not text:
        return ""
    
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Strip URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Standardize repeated punctuation (e.g., "!!!" -> "!", "???" -> "?")
    text = re.sub(r'(!)\1+', r'\1', text)
    text = re.sub(r'(\?)\1+', r'\1', text)
    text = re.sub(r'(\.)\1{2,}', '...', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_text(text: str, preserve_case: bool = False, remove_accents: bool = True) -> str:
    """
    Backwards-compatible wrapper for text normalization.
    """
    cleaned = clean_text(text)
    if not preserve_case:
        cleaned = cleaned.lower()
    return cleaned

def tokenize_sentences(text: str) -> List[str]:
    """
    Sentence segmentation using NLTK sent_tokenize with regex fallback.
    """
    if not text or not text.strip():
        return []
    try:
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except Exception:
        # Simple regex split fallback on sentence boundaries (. ! ?)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

def tokenize_words(text: str, remove_punct: bool = True) -> List[str]:
    """
    Word tokenization using NLTK word_tokenize with regex fallback.
    Optionally strips punctuation tokens.
    """
    if not text or not text.strip():
        return []
    try:
        tokens = word_tokenize(text)
    except Exception:
        # Fallback regex tokenization
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
    
    if remove_punct:
        tokens = [t for t in tokens if any(c.isalnum() for c in t)]
    
    return tokens

def remove_stopwords(tokens: List[str], custom_stopwords: Optional[List[str]] = None) -> List[str]:
    """
    Removes standard English stopwords (NLTK) with optional custom domain additions.
    """
    if not tokens:
        return []
    
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        # Fallback standard English stopword list
        stop_words = {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
            "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
            "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
            "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
            "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
            "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
            "at", "by", "for", "with", "about", "against", "between", "into", "through",
            "during", "before", "after", "above", "below", "to", "from", "up", "down",
            "in", "out", "on", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "any",
            "both", "each", "few", "more", "most", "other", "some", "such", "no",
            "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
            "t", "can", "will", "just", "don", "should", "now"
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
        "stemmed_tokens": stemmed
    }
