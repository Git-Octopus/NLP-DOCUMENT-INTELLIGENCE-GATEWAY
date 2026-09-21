# Remaining Work Plan — Developer 1 (Weeks 1–3 first, then Week 4)

**Audience:** Dev 1 (gateway, schemas, SQLite, extraction, route wiring).  
**Order:** Finish **100% of Weeks 1–3** in the numbered sequence below. **Do not start Week 4** until the Week 3 exit checklist is green.  
**Paths:** Every file is written as a repo-relative path (example: `app/api/extract.py`) so it is not confused with same-named files in other folders.

Legend in this document:

- **DEV 1** — you do this.
- **DEV 2 (other)** — Omkar / local NLP. Dev 1 cannot mark Week 3 done without this output existing or a documented temporary fallback.
- **DEV 3 (other)** — Qdrant, embeddings wiring, LLM fallback, Streamlit. Same rule.

---

## How to use this plan

Work **top to bottom**. Each step assumes the previous step is merged (or at least working locally). One GitHub PR per step is enough. Do not jump to Docker (`Dockerfile`, `docker-compose.yml`) until Section B.

---

# SECTION A — Weeks 1–3 to 100% (serial)

README Days 1–7 are largely done. Remaining Week 1–2 holes are **Steps 1–4** (they still block a honest “Week 2 complete”). Days 15–21 are **Steps 5–12**.

---

### Step 1 — Runtime packages (leftover Day 3 / Day 8)

**DEV 1**

1. Edit `requirements.txt`.
2. Add:
   - `sqlalchemy>=2.0.0`
   - `python-multipart>=0.0.9`
3. Keep existing FastAPI, Pydantic, PyMuPDF, python-docx lines. Do not remove spaCy/NLTK/qdrant lines owned by other devs.
4. Confirm `.gitignore` ignores `nlp_gateway.db` and `.env`.

**Done when:** `pip install -r requirements.txt` on a clean venv installs ORM + multipart; `POST /extract` no longer fails with “Form data requires python-multipart”.

**Other-dev dependency:** none.

---

### Step 2 — SQLite tables created on API start (leftover Day 4 / Day 8)

**DEV 1**

1. Edit `app/main.py` `lifespan()`.
2. Import models so metadata is registered: `app/models.py` (`AnalysisModel`, `EntityModel`, `EvidenceModel`) via `from app import models` or explicit imports.
3. Call `Base.metadata.create_all(bind=engine)` using `engine` from `app/services/database.py`.
4. Do not drop tables on shutdown.

**Done when:** `uvicorn app.main:app` then `POST /analyze` creates `nlp_gateway.db` without pytest.

**Other-dev dependency:** none.

---

### Step 3 — Settings for file size (leftover Day 13)

**DEV 1**

1. Edit `app/config.py`: add `MAX_FILE_SIZE_BYTES: int = 10485760` (10 MB).
2. Edit `app/api/extract.py`: read the limit from `settings`, delete the local-only `MAX_FILE_SIZE_BYTES` constant.
3. Edit `.env.example`: add `MAX_FILE_SIZE_BYTES=10485760` (optional override).

**Done when:** oversized upload to `/extract` still returns 413, driven by config.

**Other-dev dependency:** none.

---

### Step 4 — `/extract` cleanup + `/analyze` same file rules (Days 9–13 leftover)

**DEV 1**

1. Edit `app/api/extract.py`:
   - Remove unused `Depends(get_db)` **or** actually persist extract results (prefer remove — no extra tables).
   - Keep empty-file 400, bad type 400, `ExtractionError` handling already in this file.
2. Edit `app/api/analyze.py`:
   - After `await file.read()`, if `len(file_bytes) > settings.MAX_FILE_SIZE_BYTES` raise 413 (today only `app/api/extract.py` checks size).
3. Keep extractors in `app/services/extractors.py` (`extract_text_from_pdf`, `extract_text_from_docx`, `parse_document_stream`). Do not duplicate parsers in the API layer.
4. Export `DocumentMetadata` from `app/schemas/__init__.py` (it lives in `app/schemas/analysis.py` today).

**Done when:** PDF/DOCX work on both `/extract` and `/analyze`; 10 MB enforced on both; no unused DB dependency on extract.

**Other-dev dependency:** none.

---

### Step 5 — Persist file metadata in CRUD (Day 8 / Day 17 leftover)

**DEV 1**

1. Edit `app/services/crud.py` `create_analysis()`:
   - Copy `filename`, `file_type`, `word_count`, `char_count` from `analysis_data.doc_metadata` into `app/models.py` `AnalysisModel` columns (columns already exist).
2. Keep writing `entities` → `EntityModel` and `evidence` → `EvidenceModel`.

**Done when:** analyzing an uploaded DOCX stores filename in SQLite, not only in the JSON response.

**Other-dev dependency:** none.

---

### Step 6 — Frozen call contract for Dev 2 and Dev 3 (Day 11 / Day 15)

**DEV 1 owns the consumer.** Other devs own the implementations. Dev 1 does **not** rewrite `app/nlp/vector_store.py` or `app/nlp/ner.py` in this step except to agree signatures.

Put the adapter in a **new** Dev 1 file so PRs do not collide:

- Create `app/services/semantic.py` (DEV 1).
- Optionally create `app/services/sentiment.py` only if Dev 2 has no scorer yet (temporary; comment `# TEMP until Dev 2 sentiment`).

**Required return shapes (must match `app/schemas/analysis.py`):**

| Function | Path that should provide it | Return dict |
|---|---|---|
| `run_local_nlp_pipeline(text)` | `app/nlp/pipeline.py` | `cleaned_text`, `keywords`, `entities`, tokens/sentences |
| `extract_all_entities(text)` | `app/nlp/ner.py` | list of entity dicts |
| `extract_keywords(text, top_n)` | `app/nlp/tfidf.py` | list of `{"keyword", "score"}` |
| `extract_evidence_sentences(...)` | `app/nlp/rules.py` | `list[str]` |
| `analyze_linguistics(text)` | `app/nlp/linguistic.py` | lemmas, pos, chunks |
| `match_topic(text)` | `app/nlp/topics.py` | `{topic, confidence, resolved_by}` |
| `match_intent(text)` | `app/nlp/topics.py` | `{intent, confidence, resolved_by}` |
| `search_duplicate_message(text)` | must accept **str** (today stub in `app/nlp/vector_store.py` documents a **vector**) | `{is_duplicate, matched_message_id, similarity_score}` |
| `get_embedding(text)` | `app/nlp/embeddings.py` | `list[float]` |
| `compute_cosine_similarity(a, b)` | `app/nlp/embeddings.py` | `float` |
| sentiment helper | **DEV 2 (other)** — add e.g. `analyze_sentiment(text)` under `app/nlp/` | `{sentiment, score}` matching `SentimentAnalysis` |

**Broken contract to fix in `app/services/semantic.py` (DEV 1 wrapper, not a rewrite of Dev 3):**

- `app/api/analyze.py` currently calls `search_duplicate_message(raw_text)`.
- `app/nlp/vector_store.py` stub is `search_duplicate_message(text_vector, threshold=0.85)`.
- Wrapper: if Dev 3 only accepts a vector, call `get_embedding(text)` from `app/nlp/embeddings.py` first.

**DEV 2 (other) — required for Dev 1 Week 3 `/analyze` fields**

- Keep `app/nlp/preprocess.py` `clean_text` / `tokenize_sentences` working (already present).
- Keep `app/nlp/pipeline.py` `run_local_nlp_pipeline` aligned with entity/keyword keys (`text`/`entity`, `start`/`start_char`).
- **Add sentiment** (README Day 17). If missing, Dev 1 may ship a temporary lexicon in `app/services/sentiment.py` and replace it when Dev 2 lands. Comment the temp clearly.

**DEV 3 (other) — required for *real* topic/duplicate/LLM; not required to *start* Step 7 if fallback exists**

Must complete for Dev 1 to drop fallbacks before calling Week 3 “NLP-first + Qdrant” (still inside Week 3 if possible):

1. Fill `config/topics.yaml` (10 topics + example phrases). File is currently **empty**.
2. Implement `app/nlp/vector_store.py` (`connect_qdrant`, `init_collections`, `upsert_topic_vectors`, `search_similar_topics`, `search_duplicate_message`).
3. Implement `app/nlp/topics.py` `match_topic` / `match_intent` (stop leaving `pass`).
4. Implement LLM fallback in `app/providers/base.py`, `app/providers/gemini_provider.py` and/or `app/providers/openai_provider.py` (`resolve_ambiguous`, optional `summarize`). Files are currently **empty**.
5. After each successful analyze, upsert into Qdrant messages collection (Dev 3 helper called from Dev 1 `app/api/analyze.py`).
6. Streamlit in `streamlit_app/app.py` is **DEV 3 (other)** — needed for README Day 16/21 **UI**. Dev 1 Week 3 **API** checkpoint uses curl/`/docs` if this file stays empty.

**Done when:** team agrees the table above; Dev 3 changes the duplicate function to take `text: str` **or** Dev 1 wrapper is documented.

---

### Step 7 — Local fallbacks so `/analyze` is honest (Days 15–17, Dev 1)

**DEV 1** — implement in `app/services/semantic.py`:

1. Try Dev 3 functions from `app/nlp/topics.py` and `app/nlp/vector_store.py`.
2. If `None`, `pass`, or exception: **do not** fill `TopicMatch` with `confidence=0.50` and pretend Qdrant worked (that is current `app/api/analyze.py` behavior).
3. Fallback topic/intent via keyword rules using the same topic **names** as `config/topics.yaml` (if YAML still empty, Dev 1 may add a **minimal 10-topic seed** in `config/topics.yaml` and comment `# DEV 1 seed; Dev 3 owns phrases`).
4. Duplicate fallback if Qdrant down: `is_duplicate=False`, `similarity_score=0.0` (honest). Pairwise compare stays `app/api/similarity.py`.
5. `llm_fallback_used=True` only when a provider in `app/providers/` actually ran.

Log a warning: `semantic fallback: qdrant_unavailable`.

**Other-dev:** Step 6 items. Fallback lets Dev 1 continue serially without waiting.

---

### Step 8 — Wire `/analyze` for real (Days 15–17)

**DEV 1** — edit `app/api/analyze.py` only (do not edit Dev 2 NLP bodies):

1. Keep multipart `text` + `file` (tests in `tests/test_analyze_route.py` and `tests/test_pipeline.py` use form `data=`).
2. Document in the route description that JSON `AnalyzeRequest` in `app/schemas/analysis.py` is **not** the POST body unless you add a second JSON route. Prefer **one** contract: form/file. Adding JSON is optional extra, not required for Week 3 if OpenAPI text is clear.
3. Replace piecemeal imports with:
   - `run_local_nlp_pipeline` from `app/nlp/pipeline.py` for cleaned text, keywords, entities.
   - Evidence from `app/nlp/rules.py`.
   - Topic/intent/duplicate from `app/services/semantic.py`.
   - Sentiment from Dev 2 **or** `app/services/sentiment.py` temp.
4. Remove unused import of `analyze_linguistics` if still unused.
5. After `create_analysis` in `app/services/crud.py`, try `upsert_message_embedding` (Dev 3); swallow only that call, log it.
6. Optional: `GET /analyze/{request_id}` using `get_analysis_by_request_id` in `app/services/crud.py`.

**DEV 2 (other):** pipeline functions must not crash on empty/short text (`app/nlp/preprocess.py`, `app/nlp/pipeline.py`).

**DEV 3 (other):** when implementations exist, Step 7 automatically uses them; no second rewrite of `app/api/analyze.py`.

**Done when:** response matches `AnalyzeResponse` in `app/schemas/analysis.py`; SQLite has parent + children; sentiment is not hardcoded `neutral` unless the scorer returned that.

---

### Step 9 — `/similarity` finish (Day 18)

**DEV 1** — `app/api/similarity.py` is mostly done.

1. Keep `get_embedding` / `compute_cosine_similarity` from `app/nlp/embeddings.py`.
2. Keep TF-IDF fallback inside `app/api/similarity.py` if vectors are all zeros.
3. Threshold from `app/config.py` `DUPLICATE_SIMILARITY_THRESHOLD`.
4. Add/adjust tests (new or existing under `tests/`).

**DEV 3 (other):** `app/nlp/embeddings.py` must keep returning a 384-d list (already implemented). Dev 1 does not reimplement SentenceTransformers.

**Done when:** identical strings → score 1.0 / duplicate true; empty strings → 422.

---

### Step 10 — `/summarize` not dummy (Day 13 / Day 18)

**DEV 1** — edit `app/api/summarize.py`.

1. If Dev 3 implemented `summarize()` on a provider in `app/providers/`, call it.
2. Else **extractive** summary: `tokenize_sentences` from `app/nlp/preprocess.py` + first/keyword-ranked sentences; honor `max_length` from `app/schemas/summarize.py` `SummarizeRequest`.
3. Fill real `original_word_count` and `summary_word_count` in `SummarizeResponse`.

**DEV 3 (other):** optional LLM `summarize` in `app/providers/gemini_provider.py` / `app/providers/openai_provider.py`. Not a Week 3 blocker if extractive works.

**Done when:** changing input text changes the summary (current dummy paragraph in `app/api/summarize.py` must go).

---

### Step 11 — Central validation and errors (Day 20)

**DEV 1** — edit `app/main.py`:

1. Add `RequestValidationError` handler (same JSON shape as existing `HTTPException` / 500 handlers in `app/main.py`).
2. Keep request logging middleware.
3. Extend `GET /health` to report whether SQLite engine connects (optional Qdrant ping in try/except so missing Dev 3 does not fail health).

**Other-dev dependency:** none for HTTP handlers.

---

### Step 12 — Tests and Week 3 checkpoint (Days 19–21)

**DEV 1**

1. `tests/test_extractors.py`, `tests/test_extract_route.py` — still pass.
2. `tests/test_db_crud.py` — include file metadata fields if you persist them.
3. `tests/test_analyze_route.py` — oversized file 413; short text 422.
4. `tests/test_pipeline.py` — parent row in DB; prefer using `tests/conftest.py` `get_db` override instead of a second engine if tests flake.
5. Day 19: live `POST /analyze` with the team; one saved JSON sample.
6. Day 21 **Dev 1 bar:** upload via `/extract` (`app/api/extract.py`) then `/analyze` (`app/api/analyze.py`) then row in SQLite. **Do not wait** for `streamlit_app/app.py`.

**DEV 3 (other):** `streamlit_app/app.py` for the README “upload in Streamlit” Day 21 demo. Comment: **not required for Dev 1 to declare Weeks 1–3 API complete.**

**DEV 2 (other):** integration-style checks on `app/nlp/` if they fail pytest when `/analyze` calls the pipeline — Dev 2 fixes NLP, Dev 1 does not patch spaCy internals.

---

## Week 3 exit checklist (must all be true before Week 4)

- [ ] `requirements.txt` includes sqlalchemy + python-multipart
- [ ] `app/main.py` creates tables on startup
- [ ] `app/config.py` has `MAX_FILE_SIZE_BYTES`; `app/api/extract.py` and `app/api/analyze.py` enforce it
- [ ] `app/services/crud.py` stores file metadata + entities + evidence
- [ ] `app/api/analyze.py` uses pipeline + `app/services/semantic.py`; no fake Qdrant 0.50
- [ ] Sentiment not hardcoded unless scorer output is neutral
- [ ] `app/api/similarity.py` thresholded and tested
- [ ] `app/api/summarize.py` extractive or real LLM
- [ ] Validation handler in `app/main.py`
- [ ] Dev 1 pytest files green
- [ ] Handshake sent: Dev 2 sentiment; Dev 3 `topics.yaml`, `app/nlp/topics.py`, `app/nlp/vector_store.py`, providers

Until this list is green, **do not** edit `Dockerfile` or `docker-compose.yml` except to claim ownership in chat (“Dev 1 owns those files in Week 4”).

---

# SECTION B — Week 4 only (Days 22–28)

Start only after Section A checklist.

### Step 13 — Container files (Day 22) — DEV 1

1. Write `Dockerfile` (currently empty): Python 3.10, install `requirements.txt`, `python -m spacy download en_core_web_sm`, NLTK data, `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
2. Write `docker-compose.yml` (currently empty): service `api` + service `qdrant` (`qdrant/qdrant:latest`), `QDRANT_URL=http://qdrant:6333`.
3. Finalize `.env.example` and `requirements.txt` pins if compose revealed missing packages.

**DEV 3 (other):** Qdrant collection upsert on API start can live in `app/nlp/vector_store.py`; Dev 1 `app/main.py` lifespan may call it in try/except.

### Step 14 — Local compose test (Day 23) — DEV 1

`docker compose up --build`; hit `/health`, `/extract`, `/analyze`. If collections empty, Step 7 fallback still returns JSON.

### Step 15 — Deploy (Day 24) — DEV 1

Render (or similar). SQLite on free tier is ephemeral unless a disk is attached — document that in `README.md`.

**DEV 3 (other):** Qdrant Cloud URL/API key; Dev 1 only pastes into host env (`QDRANT_URL`, `QDRANT_API_KEY` from `.env.example`).

### Step 16 — API documentation (Day 25) — DEV 1

`README.md` run instructions + curl for `/analyze`, `/extract`, `/similarity`, `/summarize`. FastAPI UI remains `/docs`.

**DEV 2 (other):** evaluation dataset/metrics (not Dev 1).  
**DEV 3 (other):** architecture diagram, Streamlit polish, demo video.

### Step 17 — Bug bash and viva (Days 26–28) — all

**DEV 1:** deployed API regressions.  
**DEV 2 (other):** metrics vs final NLP.  
**DEV 3 (other):** UI against **deployed** API not only localhost.

---

# SECTION C — GitHub (serial with the steps)

1. PR Step 1–2 before anyone else rebases (`requirements.txt`, `app/main.py` conflict magnets).
2. Never resolve conflicts by deleting bodies in `app/nlp/preprocess.py`, `app/nlp/linguistic.py`, `app/nlp/tfidf.py`, `app/nlp/ner.py`, `app/nlp/rules.py`, `app/nlp/embeddings.py`, `app/nlp/vector_store.py`, `app/nlp/topics.py`.
3. Dev 1 must not implement Qdrant inside `app/nlp/vector_store.py` unless the team assigns it; use `app/services/semantic.py`.
4. Rebase onto `main` (or `dev`) before each push. CI: `.github/workflows/ci.yml`.

---

# SECTION D — Summary table

Rows = README **day** whose work must still be finished (or verified).  
Columns = **Dev 1 / Dev 2 / Dev 3**.  
`DONE` = already satisfied in the repo.  
`(other)` = not Dev 1’s implementation, but Dev 1 Week 3 needs the output.

| Day | Dev 1 | Dev 2 (other) | Dev 3 (other) |
| ---: | --- | --- | --- |
| 1 | DONE: repo layout, start `app/schemas/` | DONE: `app/nlp/` stubs exist | **Remaining:** draft `config/topics.yaml` (file empty) |
| 2 | DONE: freeze `app/schemas/analysis.py`, `app/schemas/extract.py`, `app/schemas/similarity.py`, `app/schemas/summarize.py`; skeleton routes | DONE: start `app/nlp/preprocess.py` | **Remaining:** topic examples in `config/topics.yaml`; `app/providers/base.py` still empty |
| 3 | **Remaining:** `requirements.txt` missing sqlalchemy / python-multipart; `.env.example` exists | Verify `app/nlp/preprocess.py` stopwords/stem on samples | **Remaining:** 10 topics in `config/topics.yaml`; LLM key in their env (not committed) |
| 4 | **Remaining:** `app/main.py` must `create_all` using `app/services/database.py` + `app/models.py` | Continue `app/nlp/linguistic.py` (largely present) | **Remaining:** implement `app/nlp/vector_store.py` (stubs) |
| 5 | DONE: PDF in `app/services/extractors.py` | Linguistic tests as they own them | **Remaining:** upsert topics into Qdrant; `app/nlp/embeddings.py` already usable |
| 6 | DONE: DOCX in `app/services/extractors.py` | Unit tests for preprocess/linguistic | **Remaining:** messages collection + provider skeleton `app/providers/openai_provider.py` / `app/providers/gemini_provider.py` (empty) |
| 7 | Verify checkpoint: schemas + `app/api/*` + extractors | Checkpoint NLP samples | Checkpoint Qdrant (blocked until `app/nlp/vector_store.py` is real) |
| 8 | **Remaining:** `app/services/crud.py` persist `doc_metadata`; startup tables (Step 2) | `app/nlp/tfidf.py` (present) | **Remaining:** `resolve_ambiguous` in `app/providers/` |
| 9 | **Remaining:** unused `get_db` in `app/api/extract.py`; size limit via `app/config.py` | Continue TF-IDF | **Remaining:** tune threshold; test low-confidence messages |
| 10 | Re-test `app/api/extract.py` e2e after Step 4 | Keyword tests | **Remaining:** fallback entity prompt in providers |
| 11 | Confirm `app/models.py` matches `app/schemas/analysis.py` entities/evidence | `app/nlp/rules.py` (present) | **Remaining:** duplicate search real in `app/nlp/vector_store.py`; **fix signature vs** `app/api/analyze.py` |
| 12 | **Remaining:** 10 MB on `app/api/analyze.py` file path | `app/nlp/ner.py` (present) | **Remaining:** combine topic+duplicate+LLM; upsert messages |
| 13 | **Remaining:** `RequestValidationError` in `app/main.py`; replace dummy `app/api/summarize.py` | Finish NER merge | Optional second provider |
| 14 | Demo `/extract` + DB writes after Steps 1–5 | Demo local NLP matching schema | Demo Qdrant+LLM matching schema (blocked on stubs) |
| 15 | **Remaining:** `app/api/analyze.py` call `app/nlp/pipeline.py` + `app/services/semantic.py` | Support signature fixes in `app/nlp/*` | Support embeddings; **Remaining:** `streamlit_app/app.py` skeleton (empty) |
| 16 | Merge entities + topic/intent into `AnalyzeResponse` | Bugfixes from integration | Streamlit calls `/analyze` (`streamlit_app/app.py`) |
| 17 | **Remaining:** save full result in `app/services/crud.py`; sentiment field | **Remaining:** sentiment function in `app/nlp/` | Streamlit formatted UI |
| 18 | **Remaining:** prove `app/api/similarity.py`; extractive `app/api/summarize.py` | Integration tests on 10 messages | Finalize primary provider in `app/providers/` |
| 19 | Live `/analyze` sync; fix schema breaks | NLP test coverage | Streamlit shows every `app/schemas/analysis.py` field |
| 20 | **Remaining:** logging already in `app/main.py`; add validation handler + health/DB | Polish `app/nlp/` eval tests | File upload in `streamlit_app/app.py` → `/extract` then `/analyze` |
| 21 | **Dev 1 Week 3 done:** file → `app/api/extract.py` → `app/api/analyze.py` → SQLite | Assist pipeline bugs | **(UI)** Streamlit e2e; API e2e does not wait |
| 22 | **Week 4:** write `Dockerfile`, `docker-compose.yml`; freeze `requirements.txt`, `.env.example` | Eval dataset (other) | Polish `streamlit_app/app.py` |
| 23 | **Week 4:** `docker compose up`; fix API+Qdrant networking | Tricky examples dataset | Architecture diagram |
| 24 | **Week 4:** deploy API; env `QDRANT_URL` | Annotate eval / F1 | Qdrant Cloud instance (other) |
| 25 | **Week 4:** `README.md` API examples | Metrics report (other) | Demo video (other) |
| 26 | Bug bash deployed gateway | Check metrics vs code | UI vs deployed API |
| 27 | Final `README.md` + Docker instructions | Eval report review | PPT / report (other) |
| 28 | Demo / viva — explain `app/main.py`, `app/schemas/`, `app/services/`, `app/api/` | Explain `app/nlp/` | Explain Qdrant + `streamlit_app/app.py` |

**How to read the table:** Dev 1 completes every **Remaining** cell in column Dev 1 for Days **1–21** before any Day **22+** cell. Grey-area Days 15–21 Dev 3 Streamlit/Qdrant are **(other)**; Dev 1 uses Steps 6–7 fallbacks so Week 3 API can still close.
