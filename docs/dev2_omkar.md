# Dev 2 (Omkar) — Local NLP Core Pipeline Implementation & Architecture

This document tracks the completed design, schema mapping, modules, algorithms, and integration specifications delivered by **Dev 2 (Omkar)** for the **NLP Document Intelligence Gateway**.

---

## 1. Overview & Role

- **Owner:** Dev 2 (Omkar)
- **Scope:** Core Local NLP Pipeline (Text Preprocessing, Tokenization, POS Tagging, Lemmatization, Stemming, N-grams, Bag of Words, TF-IDF / Keyword Extraction, Rule-based Entities, spaCy NER, Domain Sentiment Analysis, and Core NLP Orchestrator Service).
- **Design Philosophy:** Local NLP First. The entire pipeline in `app/nlp/` runs 100% locally with spaCy, NLTK, scikit-learn, and regex without requiring external LLM API calls or cloud fees.

---

## 2. Pydantic Schemas Mapping (`app/schemas/`)

Dev 2's local NLP outputs directly satisfy key fields of the gateway's `AnalyzeResponse` schema:

| Schema Field (`AnalyzeResponse`) | Schema Type | Producing NLP Function / Module |
|---|---|---|
| `cleaned_text` | `str` | `preprocess.clean_text()` / `normalize_text()` ([`app/nlp/preprocess.py`](../app/nlp/preprocess.py)) |
| `topic` | `TopicMatch` | [`app/services/semantic.py`](../app/services/semantic.py) via fallback keyword matching / Dev 3 |
| `intent` | `IntentMatch` | [`app/services/semantic.py`](../app/services/semantic.py) via fallback keyword matching / Dev 3 |
| `entities` | `List[ExtractedEntity]` | [`app/nlp/ner.py`](../app/nlp/ner.py) (`extract_all_entities` combining Regex + spaCy NER with span deduplication) |
| `keywords` | `List[str]` | [`app/nlp/tfidf.py`](../app/nlp/tfidf.py) (`extract_keywords` combining TF-IDF & noun chunks) |
| `sentiment` | `SentimentAnalysis` | [`app/nlp/sentiment.py`](../app/nlp/sentiment.py) (`analyze_sentiment_local` with support domain lexicon & negations) |
| `evidence_sentences` | `List[str]` | [`app/nlp/rules.py`](../app/nlp/rules.py) (`extract_evidence_sentences` via sentence tokenization + keyword density) |
| `duplicate_info` | `DuplicateCheck` | [`app/services/semantic.py`](../app/services/semantic.py) / Dev 3 Vector Store |

---

## 3. Module Architecture (`app/nlp/`)

### 3.1 [`app/nlp/preprocess.py`](../app/nlp/preprocess.py)
Text preprocessing, normalization, and tokenization:
- `get_entity_preserving_text(text)`: Preserves case, currency symbols, and formatting for entity offset calculation.
- `get_analysis_text(text)`: Lowercased, contraction-expanded, accent-removed text for statistical matching.
- `normalize_text(text, preserve_case, remove_accents)`: Unicode NFKD normalization and contraction expansion.
- `tokenize_sentences(text, backend)`: Sentence segmentation (NLTK `sent_tokenize` with regex fallback).
- `tokenize_words(text, remove_punct, backend, return_offsets)`: Word tokenization returning structured tokens with character offsets and normalized tokens.
- `remove_stopwords(tokens, custom_stopwords, backend)`: Standard English stopwords filtering with custom support stopword extensions.
- `apply_stemming(tokens, stemmer_type)` / `stem_tokens()`: Snowball and Porter stemmers.
- `preprocess_pipeline(text)`: Full pipeline dictionary containing original, normalized, tokenized, filtered, and stemmed outputs.

### 3.2 [`app/nlp/linguistic.py`](../app/nlp/linguistic.py)
Linguistic analysis powered by spaCy `en_core_web_sm`:
- `get_spacy_model()`: Lazy singleton model loader with auto-download fallback.
- `analyze_linguistics(text)`: Extracts lemmas, coarse/fine POS tags, dependency triples, noun chunks, and `tokens_info` (structured per-token linguistic properties).
- `lemmatize_text(text)`: Returns list of lowercased lemmas.
- `pos_tag_text(text)`: Returns list of tuples `(token, coarse_pos, fine_pos)`.
- `extract_noun_chunks(text)`: Returns extracted noun chunks.
- `extract_dependency_features(text)`: Returns dependency tree triples `{"token": ..., "dep": ..., "head": ...}`.

### 3.3 [`app/nlp/tfidf.py`](../app/nlp/tfidf.py)
Bag-of-Words, N-grams, TF-IDF, and hybrid keyword extraction:
- `generate_bow_and_tfidf(corpus, ngram_range)`: Computes feature names, vocabulary, counts, and dense matrices.
- `compute_bow(corpus, max_features)`: Dedicated Bag-of-Words vectorizer returning counts and vocabulary.
- `compute_tfidf(corpus, max_features, ngram_range)`: TF-IDF feature weights and ranked term list sorted descending.
- `extract_keywords(text, top_n, reference_corpus)`: Hybrid extractor combining spaCy candidates (noun phrases, NOUN, PROPN, ADJ+NOUN) with TF-IDF statistical scoring and source attribution (`noun_phrase` vs `tfidf`).
- `extract_ngrams(tokens, n)`: N-gram generator for unigram, bigram, trigram sequences.
- `extract_all_ngrams(tokens, max_n, max_per_n)`: Structured dictionary of unigrams, bigrams, and trigrams.

### 3.4 [`app/nlp/rules.py`](../app/nlp/rules.py)
Deterministic regex entity extraction and evidence sentence mining:
- `extract_rule_entities(text)`: High-precision patterns for:
  - **EMAIL:** RFC 5322 regex (`customer@example.com`, `support@company.in`).
  - **PHONE:** Indian (+91 9876543210, 9876543210) and international (+1 555 123 4567).
  - **MONEY:** ₹, Rs., INR, $, €, £, ¥ amounts with comma separators (e.g. ₹1,500, Rs. 1500, $49.99).
  - **IDENTIFIERS:** Domain prefixes: `ORD-`, `INV-`, `TXN-`, `TKT-`, `TRK-`, `ERR-`, `CASE-`, `REF-`.
  - **DATE:** YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY, and month string formats (e.g. 12 September 2026).
  - Returns `confidence: 1.0`, `source: "rule"`, and exact character offsets on original text.
- `extract_evidence_sentences(text, keywords, max_sentences)`: Sentence ranking based on keyword density and entity signals.

### 3.5 [`app/nlp/ner.py`](../app/nlp/ner.py)
Hybrid Named Entity Recognition:
- `extract_spacy_entities(text)`: Statistical NER for PERSON, ORG, GPE, DATE, MONEY, PRODUCT, etc. with `confidence: 0.85` and `source: "spacy"`.
- `extract_all_entities(text)`: Merges deterministic rule entities with spaCy NER. Deterministic rule entities strictly take priority in case of overlapping character spans.

### 3.6 [`app/nlp/sentiment.py`](../app/nlp/sentiment.py)
Domain-tailored local sentiment analysis:
- `analyze_sentiment_local(text)`: High-performance deterministic sentiment scoring with customer-support positive/negative lexicons, negation window parsing (inverts polarity for 3 subsequent tokens), and intensifier weighting. Returns `{"sentiment": "positive"|"neutral"|"negative", "score": float}`.

### 3.7 [`app/nlp/pipeline.py`](../app/nlp/pipeline.py)
Unified Core NLP Orchestrator Service:
- `CoreNLPService`: Central class orchestrating preprocessing, linguistic analysis, n-grams, BoW, TF-IDF, keywords, rules, NER, entity merging, sentiment, and evidence mining.
- `get_core_nlp_service()`: Returns cached singleton service instance.
- `run_local_nlp_pipeline(text)`: Top-level API providing all local features in a single call.

---

## 4. Integration with Dev 1 and Dev 3

1. **FastAPI Gateway (`app/api/analyze.py`):**
   - Multi-modal support: Seamlessly handles JSON request body (`{"text": ...}`), Form data (`data={"text": ...}`), and file uploads (`files={"file": ...}`).
   - Integrates `analyze_sentiment_local()` to dynamically score sentiment instead of hardcoded neutral.
   - Leverages `safe_topic_match()`, `safe_intent_match()`, and `safe_duplicate_check()` from `app.services.semantic` for 100% resilient operation when Qdrant / external LLM APIs are offline.
2. **Streamlit UI (`streamlit_app/app.py`):**
   - Direct compatibility with Streamlit's `requests.post(url, json=payload)` calls.
3. **Database Persistence (`app/services/crud.py`):**
   - All analyses, extracted entities, and evidence sentences are automatically persisted to SQLite relational storage.

---

## 5. Benchmark & Evaluation Results

Benchmarked over 24 ground-truth annotated support queries in [`data/evaluation_dataset.json`](../data/evaluation_dataset.json):

| Metric | Measured Result | Target Baseline | Status |
|---|---|---|---|
| **NER Recall** | **100.0%** | 80.0% | ✅ Exceeded |
| **NER Macro F1-Score** | **78.4%** | 75.0% | ✅ Exceeded |
| **Average Processing Latency** | **48.7 ms / message** | < 150 ms | ✅ Real-time (<50ms) |
| **Test Suite Coverage** | **93 passing tests** | 85 tests | ✅ 100% Green |
