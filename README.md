# NLP Document Intelligence Gateway
### Customer Support Text Analysis API — College Mini Project (3–4 Weeks)

A FastAPI gateway that takes unstructured customer support text/documents and
returns structured JSON: topic, intent, entities, keywords, sentiment, and evidence.
Combines local NLP (spaCy, NLTK, scikit-learn) with an external LLM API (OpenAI/Gemini).

> **Project philosophy: NLP-first, LLM-as-fallback.**
> The academic core of this project is classical/local NLP — preprocessing,
> tokenization, POS tagging, NER, TF-IDF, keyword extraction, and semantic
> similarity via Qdrant. These should resolve the **majority** of requests on
> their own. The LLM (OpenAI/Gemini) is called **only** when local NLP is not
> confident enough — e.g. topic similarity score below threshold, or an
> entity type not covered by regex/spaCy rules. This keeps API cost low,
> keeps the project explainable in viva as "we built NLP, not a wrapper," and
> matches what the grader expects to see implemented, not outsourced.

---

## 1. Team & Roles

| Dev | Owns | Tech |
|---|---|---|
| **Dev 1** | Gateway, Schemas, DB, Deployment | FastAPI, Pydantic, SQLite, PyMuPDF, python-docx, Docker |
| **Dev 2** | Core NLP Pipeline (local) — **primary focus of the project** | spaCy, NLTK, scikit-learn, Regex |
| **Dev 3** | Semantic matching (Qdrant) + LLM fallback + UI | Sentence Transformers, Qdrant (vector DB), OpenAI/Gemini API (fallback only), Streamlit, YAML |

**Where the LLM is actually used (kept minimal, by design):**
- Topic/intent confirmation **only** when Qdrant similarity score falls below threshold (ambiguous message).
- Entity extraction **only** for entity types not caught by regex rules or spaCy NER.
- Optional summarization endpoint (`/summarize`) — separate, non-core feature.

Everything else — normalization, tokenization, stopwords, stemming, lemmatization,
POS tagging, dependency parsing, N-grams/BoW/TF-IDF, keyword extraction, rule-based
entity extraction, spaCy NER, and topic/duplicate matching via Qdrant — is 100% local
NLP with **no LLM call involved**.

**Golden rule:** Dev 1 freezes the Pydantic schemas (`app/schemas/`) on Day 1–2.
Dev 2 and Dev 3 build against those schemas independently — nobody waits for
anybody's full module to be "done" before starting their own.

---

## 2. Why NOT fully sequential (dev1 → dev2 → dev3)

Working one dev at a time means dev2 and dev3 sit idle for a week, and any
mistake in dev1's design is only discovered after everyone else is blocked.
Instead, **all 3 devs start Week 1** and work in parallel against a shared
contract (the schemas), with short syncs to catch mismatches early.

---

## 3. Folder Structure

```
nlp-gateway/
│
├── app/
│   ├── main.py
│   ├── config.py
│   ├── api/
│   │   ├── analyze.py
│   │   ├── similarity.py
│   │   └── extract.py
│   ├── nlp/
│   │   ├── preprocess.py
│   │   ├── linguistic.py
│   │   ├── tfidf.py
│   │   ├── ner.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py      # Qdrant client wrapper
│   │   ├── topics.py
│   │   └── rules.py
│   ├── providers/
│   │   ├── base.py
│   │   ├── openai_provider.py
│   │   └── gemini_provider.py
│   ├── schemas/
│   └── services/
│
├── config/
│   └── topics.yaml
├── tests/
├── streamlit_app/
├── .env.example
├── Dockerfile
├── docker-compose.yml        # runs API + Qdrant together
├── requirements.txt
└── README.md
```

> **Qdrant setup:** run it locally via Docker (`docker run -p 6333:6333 qdrant/qdrant`)
> or use [Qdrant Cloud](https://cloud.qdrant.io) free tier for the deployed version.
> Store the URL/API key in `.env` as `QDRANT_URL` and `QDRANT_API_KEY`.

---

## 4. Day-by-Day Plan

Legend: 🔴 = must sync with other devs today · 🟢 = independent work

### WEEK 1 — Foundation

| Day | Dev 1 | Dev 2 | Dev 3 |
|---|---|---|---|
| **Day 1** 🔴 | Init repo, create folder structure (`app/`, `api/`, `nlp/`, `providers/`, `schemas/`). Start writing Pydantic schemas (topic, intent, entity, keyword, sentiment, evidence). | Set up `app/nlp/` module files (empty stubs). Read Dev 1's draft schemas as they're written. List out preprocessing steps needed (normalization, tokenization, stopwords, stemming). | Create `config/topics.yaml` draft with 10 topic names + short descriptions. Decide embedding model (Sentence Transformers, e.g. `all-MiniLM-L6-v2`) that will feed vectors into **Qdrant**. |
| **Day 2** 🔴 | Finish + **freeze** Pydantic schemas. Push to repo. Build FastAPI skeleton: `main.py`, routes `/analyze`, `/similarity`, `/extract`, `/summarize` returning dummy JSON matching schema. | Pull frozen schemas. Start `preprocess.py`: text normalization (regex) + sentence/word tokenization (spaCy/NLTK). | Add 3–5 example phrases per topic in `topics.yaml`. Spin up **Qdrant locally via Docker** (`docker run -p 6333:6333 qdrant/qdrant`). Scaffold `providers/base.py` (abstract provider class: `get_intent()`, `extract_entities()`, `summarize()`). |
| **Day 3** | Add `.env.example`. Test that dummy `/analyze` endpoint runs locally and returns schema-valid JSON. | Continue `preprocess.py`: stopword removal + stemming (NLTK). Test on 5–10 sample support messages. | Finalize all 10 topics in `topics.yaml` with descriptions + examples. Decide primary LLM provider (OpenAI or Gemini) and get API key set up. |
| **Day 4** | Set up SQLite connection (`database.py`). Draft table schemas: `analysis`, `entities`, `evidence` (don't wire yet). | Start `linguistic.py`: lemmatization + POS tagging using spaCy pretrained model. | Install Sentence Transformers. Write `vector_store.py`: Qdrant client wrapper (`connect()`, `create_collection()`). Create a `topics` collection in Qdrant sized to the embedding model's dimension. |
| **Day 5** | Implement PDF text extraction using PyMuPDF (`extract_pdf(file)` function, tested standalone). | Continue `linguistic.py`: dependency parsing. Test lemmatizer + POS output on sample messages. | Embed all `topics.yaml` descriptions/examples and **upsert them into the Qdrant `topics` collection** (with topic label as payload). Write `embeddings.py`: embed an incoming message and query Qdrant for nearest topic. Test on 5 sample messages. |
| **Day 6** | Implement DOCX text extraction using python-docx (`extract_docx(file)`). Test both extractors with sample files. | Write unit tests for `preprocess.py` and `linguistic.py` outputs. Clean up code, add docstrings. | Refine Qdrant similarity-score threshold for topic matching. Create a second Qdrant collection (`messages`) to store past analyzed messages — this will power **duplicate detection**. Start `openai_provider.py` or `gemini_provider.py` skeleton (auth + basic call). |
| **Day 7** 🔴 | **Checkpoint:** demo schema + FastAPI skeleton + extraction functions to team. Confirm nothing needs to change in schema shape. | **Checkpoint:** demo preprocessing + linguistic pipeline output on 3 sample texts. | **Checkpoint:** demo topic similarity matching via Qdrant on 3 sample texts. Confirm LLM provider choice with team. |

### WEEK 2 — Core Build

| Day | Dev 1 | Dev 2 | Dev 3 |
|---|---|---|---|
| **Day 8** | Wire SQLite tables for real (`analysis`, `entities`, `evidence` — create + insert functions). | Start `tfidf.py`: N-grams + Bag of Words using scikit-learn. | Build the LLM provider as a **fallback function only**: `resolve_ambiguous(text, top_qdrant_matches)` — called when Qdrant confidence is low. |
| **Day 9** | Wire extraction functions (`extract_pdf`, `extract_docx`) into `/extract` route so it returns real extracted text. | Continue `tfidf.py`: TF-IDF vectorization + keyword extraction (TF-IDF + noun phrases). | Test fallback function on 10 deliberately ambiguous messages (low Qdrant confidence). Tune the confidence threshold that decides when to call the LLM at all. |
| **Day 10** | Test `/extract` route end-to-end with real PDF/DOCX file uploads. Fix bugs. | Finish `tfidf.py`. Write unit tests for keyword extraction. | Write fallback prompt for entity types not caught by regex/spaCy (only triggered on request, not always). |
| **Day 11** 🔴 | **Mid-sprint check:** confirm DB schema matches what entities/evidence data will look like from Dev 2/Dev 3 output. | Start `rules.py`: regex patterns for money, email, phone numbers, IDs, dates. | Test fallback entity extraction on 10 messages that regex/spaCy NER genuinely miss. Implement duplicate detection: embed incoming message, search `messages` collection in Qdrant for nearest neighbor above similarity threshold (no LLM needed here). |
| **Day 12** | Add error handling to `/extract` (bad file type, corrupted file, empty text). | Finish `rules.py`. Start `ner.py`: integrate spaCy pretrained NER model for baseline entity recognition. | Combine topic match + duplicate detection + LLM intent results into one clean function output that matches Dev 1's schema. On each `/analyze` call, upsert the new message embedding into the `messages` Qdrant collection so future duplicates can be caught. |
| **Day 13** | Write basic input validation across all routes (empty text, oversized file). | Finish `ner.py`. Merge rules-based + spaCy NER results into one entity list, deduplicate. | Add optional summarization prompt (for later `/summarize` route). Start `gemini_provider.py` (or OpenAI) as the optional second adapter — lower priority. |
| **Day 14** 🔴 | **Checkpoint:** demo DB writes + `/extract` route fully working with real files. | **Checkpoint:** demo full local NLP output (preprocess → linguistic → TF-IDF/keywords → entities) on a sample message, matching schema. | **Checkpoint:** demo topic match + duplicate detection (both via Qdrant) + intent/entity extraction from LLM provider on a sample message, matching schema. |

### WEEK 3 — Integration

| Day | Dev 1 | Dev 2 | Dev 3 |
|---|---|---|---|
| **Day 15** 🔴 | Start wiring `/analyze` route: call Dev 2's preprocessing + linguistic + TF-IDF functions in sequence. | Support Dev 1 — fix any function signature mismatches found during wiring. Start writing unit tests for `rules.py`/`ner.py`. | Support Dev 1 — help wire embeddings + LLM provider calls into `/analyze`. Start Streamlit app skeleton (`streamlit_app/app.py`) — basic text input box. |
| **Day 16** | Continue `/analyze` wiring: merge Dev 2's entities + Dev 3's topic/intent/LLM entities into one response object. | Fix bugs surfaced from integration. Add tests for `preprocess.py`/`linguistic.py` edge cases (empty text, special chars). | Streamlit: call `/analyze` endpoint from UI, display raw JSON response. |
| **Day 17** | Assemble final structured JSON: topic, intent, entities, keywords, sentiment, evidence — save result to SQLite after each `/analyze` call. | Add sentiment logic (simple rule-based or lightweight classifier) if not already covered by LLM output. | Streamlit: format JSON response into readable UI (topic badge, entity table, confidence scores). |
| **Day 18** | Wire `/similarity` route using Dev 3's embedding similarity function (compare text_a vs text_b). | Write integration tests: full pipeline on 10 sample messages, check schema compliance. | Finalize primary LLM provider end-to-end. Keep second provider adapter stubbed (not required to fully work). |
| **Day 19** 🔴 | **Sync:** run full `/analyze` flow live with team, fix any breaking issues found together. | **Sync:** confirm test coverage for `nlp/` modules is solid; report any pipeline bugs. | **Sync:** confirm Streamlit correctly displays every field from the schema. |
| **Day 20** | Add centralized error handling + logging across all routes. | Polish and re-run evaluation-style tests on the local NLP modules. | Add file upload support in Streamlit (PDF/DOCX → calls `/extract` then `/analyze`). |
| **Day 21** 🔴 | **Checkpoint:** full end-to-end test — upload a file in Streamlit → extract → analyze → structured JSON → saved in DB. All 3 devs test together. | | |

### WEEK 4 — Polish, Test, Deploy

| Day | Dev 1 | Dev 2 | Dev 3 |
|---|---|---|---|
| **Day 22** | Write Dockerfile + `docker-compose.yml` (API + Qdrant services) + finalize `.env.example` + `requirements.txt`. | Start building evaluation dataset: 300–600 realistic support messages across 10 topics (20–60 per topic). | Polish Streamlit UI styling — layout, labels, error states. |
| **Day 23** | Test full stack locally with `docker-compose up` (API + Qdrant together), fix issues. | Continue dataset creation, add paraphrased + deliberately tricky examples (typos, short/indirect messages). | Prepare architecture diagram (flow: text → preprocessing → topics/Qdrant → LLM → JSON). |
| **Day 24** | Deploy API to Render (or similar). Switch `QDRANT_URL`/`QDRANT_API_KEY` in prod `.env` to **Qdrant Cloud** free-tier instance. Confirm live URL works with a real request. | Annotate evaluation set (expected topic, entities, intent). Run pipeline against it, compute accuracy/F1 for topic detection & NER. | Draft final report sections: overview, architecture, tech stack. |
| **Day 25** | Write API documentation (routes, request/response examples). | Compute remaining metrics: similarity/duplicate detection precision-recall, sentiment accuracy. Write short evaluation report. | Record short demo video walkthrough of Streamlit app. |
| **Day 26** 🔴 | Bug bash with team — fix anything broken in deployed version. | Bug bash — verify metrics/report numbers are consistent with final code. | Bug bash — verify UI works against deployed API, not just localhost. |
| **Day 27** | Final review of README, Docker, deployment instructions. | Final review of evaluation report and dataset files. | Finish PPT + final report; polish demo video. |
| **Day 28** 🔴 | **Demo day / viva rehearsal** — all 3 devs run through the full demo together, assign who explains which part. | | |

---

## 5. Daily Sync Rule (to avoid overhead & waiting)

- **Week 1–2:** async check-ins are fine (message updates).
- **Week 3 (integration):** 10-min daily  — what's blocked, what's done.
---

## 6. Deliverables Checklist

- [ ] FastAPI gateway with all routes working
- [ ] Pydantic schemas
- [ ] Core NLP pipeline (preprocessing, POS, NER, TF-IDF, keywords)
- [ ] Topic config (`topics.yaml`) + semantic similarity via Qdrant
- [ ] Duplicate detection via Qdrant vector search
- [ ] OpenAI/Gemini provider integration
- [ ] PDF/DOCX extraction
- [ ] SQLite storage
- [ ] Streamlit demo UI
- [ ] Evaluation dataset + metrics report
- [ ] Dockerfile + deployment
- [ ] README + architecture diagram
- [ ] Final report, PPT, demo video

---

## 7. What NOT to Build (keep scope controlled)

No custom model/transformer training, no large data collection, no Redis/Kafka,
no microservices, no full CRM system, no generic chatbot as main interface.
