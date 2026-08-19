# Dev 2 (Omkar) — Local NLP Core Pipeline Implementation & Architecture

This document tracks the Day 1–7 design, schema mapping, module stubs, and preprocessing pipeline specifications assigned to **Dev 2 (Omkar)** for the **NLP Document Intelligence Gateway**.

---

## 1. Overview & Role

- **Owner:** Dev 2 (Omkar)
- **Scope:** Core Local NLP Pipeline (Text Preprocessing, Tokenization, POS Tagging, Lemmatization/Stemming, TF-IDF / Keyword Extraction, Rule-based Entities & spaCy NER integration).
- **Design Philosophy:** Local NLP first. All processing in `app/nlp/` runs 100% locally with spaCy, NLTK, scikit-learn, and regex without depending on external LLM calls.

---

## 2. Pydantic Schemas Mapping (`app/schemas/`)

Dev 2's local NLP output directly satisfies key fields of Dev 1's frozen `AnalyzeResponse` schema:

| Schema Field (`AnalyzeResponse`) | Schema Type | Producing NLP Function / Module |
|---|---|---|
| `cleaned_text` | `str` | `preprocess.normalize_text()` ([`app/nlp/preprocess.py`](../app/nlp/preprocess.py)) |
| `topic` | `TopicMatch` | [`app/nlp/topics.py`](../app/nlp/topics.py) + [`app/nlp/vector_store.py`](../app/nlp/vector_store.py) |
| `intent` | `IntentMatch` | [`app/nlp/topics.py`](../app/nlp/topics.py) + [`app/nlp/rules.py`](../app/nlp/rules.py) |
| `entities` | `List[ExtractedEntity]` | [`app/nlp/ner.py`](../app/nlp/ner.py) (`extract_all_entities` combining Regex + spaCy) |
| `keywords` | `List[str]` | [`app/nlp/tfidf.py`](../app/nlp/tfidf.py) (`extract_keywords` combining TF-IDF & noun chunks) |
| `evidence_sentences` | `List[str]` | [`app/nlp/rules.py`](../app/nlp/rules.py) (`extract_evidence_sentences` via sentence tokenization + keyword density) |
| `duplicate_info` | `DuplicateCheck` | [`app/nlp/vector_store.py`](../app/nlp/vector_store.py) (`search_duplicate_message`) |

---

## 3. Module Architecture (`app/nlp/`)

The following files and stubs have been initialized under [`app/nlp/`](../app/nlp/):

### 3.1 [`app/nlp/__init__.py`](../app/nlp/__init__.py)
Package initialization file exposing core NLP functions (`preprocess_pipeline`, `normalize_text`, `tokenize_words`, `remove_stopwords`, `stem_tokens`, `lemmatize_text`, `pos_tag_text`, `extract_keywords`, `extract_all_entities`, etc.).

### 3.2 [`app/nlp/preprocess.py`](../app/nlp/preprocess.py)
Text preprocessing module.
- `normalize_text(text, preserve_case, remove_accents)`
- `tokenize_sentences(text)`
- `tokenize_words(text, keep_punct)`
- `remove_stopwords(tokens, custom_stopwords)`
- `stem_tokens(tokens, stemmer_type)`
- `preprocess_pipeline(text)`

### 3.3 [`app/nlp/linguistic.py`](../app/nlp/linguistic.py)
Linguistic analysis module powered by spaCy.
- `lemmatize_text(text)`
- `pos_tag_text(text)`
- `extract_noun_chunks(text)`
- `extract_dependency_features(text)`

### 3.4 [`app/nlp/tfidf.py`](../app/nlp/tfidf.py)
Bag-of-Words, N-grams, and TF-IDF feature extractor.
- `compute_tfidf(corpus, max_features, ngram_range)`
- `extract_keywords(text, top_n, include_noun_phrases)`
- `extract_ngrams(tokens, n)`

### 3.5 [`app/nlp/rules.py`](../app/nlp/rules.py)
Regex-based entity extraction and evidence mining.
- `extract_regex_entities(text)` (Money, Email, Phone, Date, Order/Ticket IDs)
- `extract_evidence_sentences(text, keywords, max_sentences)`

### 3.6 [`app/nlp/ner.py`](../app/nlp/ner.py)
SpaCy NER model integration + rule deduplication.
- `extract_spacy_entities(text)`
- `extract_all_entities(text)` (Combines Regex + spaCy NER)

### 3.7 [`app/nlp/embeddings.py`](../app/nlp/embeddings.py)
Dense vector embeddings using SentenceTransformers (`all-MiniLM-L6-v2`).
- `get_embedding(text)`
- `get_embeddings_batch(texts)`

### 3.8 [`app/nlp/vector_store.py`](../app/nlp/vector_store.py)
Qdrant vector store wrapper.
- `connect_qdrant()`, `init_collections()`
- `upsert_topic_vectors()`, `search_similar_topics()`, `search_duplicate_message()`

### 3.9 [`app/nlp/topics.py`](../app/nlp/topics.py)
Topic & intent classification coordinator.
- `match_topic(text)`
- `match_intent(text)`

---

## 4. Preprocessing Pipeline Specification

To transform raw, unstructured customer support messages into data ready for NLP analysis, the preprocessing pipeline follows 4 key phases:

### Phase 1: Text Normalization
1. **Case Normalization:** Convert text to lower-case for vocabulary matching (while retaining a copy of original casing for Regex/spaCy entity offsets).
2. **HTML / Noise Removal:** Strip HTML formatting (`<p>`, `<br>`), non-printable ASCII characters, and reduce multiple whitespace sequences to single spaces.
3. **Unicode / Accent Normalization:** Apply `unicodedata.normalize('NFKD')` to handle special characters (e.g. `café` $\rightarrow$ `cafe`).
4. **Contraction Expansion:** Expand common English contractions (e.g., `"can't"` $\rightarrow$ `"cannot"`, `"I've"` $\rightarrow$ `"I have"`).
5. **Special Character Cleaning:** Preserve domain-specific symbols (`$`, `€`, `@`, `#`, `-`, `.`), while removing repetitive noise symbols (`***`, `===`).

### Phase 2: Tokenization
1. **Sentence Tokenization:** Split text into discrete sentence strings using NLTK `sent_tokenize` or spaCy sentence segmentation. Crucial for extracting `evidence_sentences`.
2. **Word Tokenization:** Divide sentences into word/punct tokens using NLTK `word_tokenize` or spaCy `Doc`. Preserve character offsets (`start_char`, `end_char`) to accurately align entities.

### Phase 3: Stopword Removal
1. **Standard English Stopwords:** Filter out high-frequency grammatical words (`the`, `is`, `at`, `on`, `which`) using NLTK/spaCy stopword sets.
2. **Domain-Specific Support Stopwords:** Remove support ticket boilerplate terms (`hello`, `dear`, `hi`, `thanks`, `regards`, `sincerely`, `please`, `ticket`, `support`) so TF-IDF focuses on intent words (`refund`, `billing`, `crash`, `charge`).
3. **Sentiment Preservation:** Retain negation terms (`not`, `no`, `never`, `cannot`) so sentiment analysis polarity remains intact.

### Phase 4: Stemming vs. Lemmatization
1. **Stemming (Porter / Snowball):** Heuristic suffix stripping (`refunded`, `refunding` $\rightarrow$ `refund`). Used as a quick feature reduction step for rule matching and TF-IDF feature matrix compression.
2. **Lemmatization (spaCy `Doc.lemma_`):** Morphological analysis using POS tag context (`better` $\rightarrow$ `good`, `was` $\rightarrow$ `be`, `charging` $\rightarrow$ `charge`). Preferred for UI display (`keywords` field in response schema) because it produces real English dictionary words.

---

## 5. Next Steps (Day 2 – Day 3)

- **Day 2:** Implement `normalize_text()`, `tokenize_sentences()`, and `tokenize_words()` in [`preprocess.py`](../app/nlp/preprocess.py).
- **Day 3:** Implement `remove_stopwords()` and `stem_tokens()` in [`preprocess.py`](../app/nlp/preprocess.py); test on 5–10 sample support messages.
- **Day 4–5:** Build [`linguistic.py`](../app/nlp/linguistic.py) for POS tagging & lemmatization.
