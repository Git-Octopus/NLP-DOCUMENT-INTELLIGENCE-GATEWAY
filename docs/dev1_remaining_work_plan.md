# Developer 1 Remaining Work Plan (Shreya)

**Owner:** Dev 1 (Gateway, Schemas, DB, Extraction, Route wiring, Deployment)  
**Rule:** Finish **100% of Weeks 1–3** before starting Week 4.  
**Date:** 2026-09-21  
**Source of truth:** `README.md` day plan + current repo (not `docs/dev1_shreya.md`, which overstates Docker/`create_all`).

---

## 0. Goal definition: “Week 3 100%” for Dev 1

Week 3 is done for Dev 1 when **all of these are true without Docker/Render**:

1. Local `uvicorn` starts, SQLite tables exist, `/health` returns 200.
2. `/extract` works on real PDF/DOCX, rejects bad/empty/oversized files.
3. `/analyze` returns **schema-valid** JSON: topic, intent, entities, keywords, sentiment, evidence, duplicate_info, `llm_fallback_used`.
4. Every `/analyze` success **writes** `analysis` + `entities` + `evidence` (and file metadata when a file was used).
5. `/similarity` compares two texts with embeddings (or documented TF-IDF fallback) vs `DUPLICATE_SIMILARITY_THRESHOLD`.
6. `/summarize` is **not dummy copy**; it uses local extractive summary until Dev 3’s LLM `summarize()` exists.
7. Central logging + HTTP/validation/500 handlers; empty text and oversized files fail with stable status codes.
8. Tests for extract, analyze, CRUD, and pipeline pass locally.
9. Topic / intent / duplicate **never silently pretend** Qdrant succeeded. If Dev 3 is missing, Dev 1 uses an **explicit local fallback** (`resolved_by: "rules"` or `"tfidf"`) so the gateway still demos.

Week 4 (Docker, compose, Render, Qdrant Cloud, README deploy section) starts **only after** the list above.

---

## 1. Current status (honest)

| Area | Status | Gap |
|---|---|---|
| Schemas | ~95% | `DocumentMetadata` not exported from `schemas/__init__.py`; `/analyze` does not use JSON `AnalyzeRequest` |
| FastAPI skeleton + CORS + logging | Done | No `RequestValidationError` handler |
| SQLite models + CRUD | ~80% | No `create_all` on app startup; file metadata columns unused |
| PDF/DOCX + `/extract` | ~90% | Size limit hardcoded; unused `db` dependency |
| `/analyze` wiring | ~55% | Silent `except`; hardcoded sentiment; signature mismatch with Dev 3 stubs |
| `/similarity` | ~80% | Works if embeddings load; else TF-IDF |
| `/summarize` | Dummy | Must replace for Week 3 completeness |
| `requirements.txt` | Incomplete | Missing `sqlalchemy`, `python-multipart` |
| Docker / deploy | **0%** | Empty `Dockerfile` / `docker-compose.yml` — **Week 4 only** |

**Dev 3 (blocker for “real” Qdrant/LLM/UI):** `topics.yaml` empty; `vector_store.py` / `topics.py` are `pass`; providers empty; Streamlit empty. Embeddings module is usable.

---

## 2. Phase order (do not skip)

```
PHASE W3-A  Unblock runtime (deps, DB create, config)
PHASE W3-B  Freeze Dev 3 contract + Dev-1-owned fallbacks
PHASE W3-C  Finish /extract + validation
PHASE W3-D  Finish /analyze merge + persist
PHASE W3-E  Finish /similarity + /summarize (local)
PHASE W3-F  Tests + Day 19/21 checkpoint (API-side)
──────────  WEEK 3 COMPLETE  ──────────
PHASE W4-A  Docker + compose
PHASE W4-B  Deploy + prod Qdrant URL
PHASE W4-C  Docs + bug bash
```

Work in **small GitHub PRs** (one phase ≈ one PR). See Section 6.

---

## 3. WEEK 3 — Phase W3-A: Unblock runtime

**Files:** `requirements.txt`, `app/main.py`, `app/config.py`, `app/models.py`

### Tasks

1. Add missing packages to `requirements.txt`:
   - `sqlalchemy>=2.0.0`
   - `python-multipart>=0.0.9` (UploadFile / form `/analyze`)
   - Keep `pytest` out of runtime file or add `requirements-dev.txt` if CI should stay explicit.
2. Import ORM models inside FastAPI `lifespan` and call `Base.metadata.create_all(bind=engine)` so first `/analyze` does not fail with “no such table”.
3. Add `MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024` to `Settings`; stop duplicating the constant only in `extract.py`.
4. Confirm `DATABASE_URL` still defaults to `sqlite:///./nlp_gateway.db`. Add `nlp_gateway.db` to `.gitignore` if not already ignored.

### Done when

- Fresh clone: `pip install -r requirements.txt`, `uvicorn app.main:app`, POST `/analyze` with text creates `nlp_gateway.db` and a row.

### PR risk

- `requirements.txt` and `main.py` are **high-conflict** files. Merge this PR first; tell Dev 2/3 to rebase.

---

## 4. WEEK 3 — Phase W3-B: Dev 3 dependency from Dev 1’s point of view

Dev 1 does **not** own Qdrant, Streamlit, or LLM prompts. Dev 1 **does** own a gateway that must still return valid JSON in Week 3. Treat missing Dev 3 as a **fallback adapter**, not a blocker.

### 4.1 Handshake Dev 1 must send Dev 3 (copy into a team chat)

Freeze these signatures. If Dev 3 already stubbed a different shape, **Dev 1’s `/analyze` is the consumer** — either Dev 3 changes stubs or Dev 1 wraps them. Prefer changing the stub to match the gateway (one string in, dict out).

```python
# REQUIRED BY /analyze (app/api/analyze.py)

def match_topic(text: str, confidence_threshold: float = 0.70) -> dict:
    # {"topic": str, "confidence": float, "resolved_by": "qdrant"|"llm_fallback"|"rules"}

def match_intent(text: str, confidence_threshold: float = 0.70) -> dict:
    # {"intent": str, "confidence": float, "resolved_by": "rules"|"qdrant"|"llm_fallback"}

def search_duplicate_message(text: str, threshold: float = 0.85) -> dict:
    # {"is_duplicate": bool, "matched_message_id": str|None, "similarity_score": float}

def upsert_message_embedding(message_id: str, text: str) -> None:
    # Called AFTER successful analyze; indexes text into Qdrant "messages"
```

**Known bug to fix in the same PR as the adapter:**  
Today `analyze.py` calls `search_duplicate_message(raw_text)` but `vector_store.py` documents `search_duplicate_message(text_vector, ...)`. Dev 1 should call a **gateway helper** that always takes `text: str`. If Dev 3 only implements vector-in, Dev 1 embeds first with `get_embedding(text)` then searches.

Optional for `/summarize`:

```python
def summarize(text: str, max_length: int = 150) -> dict:
    # {"summary": str, "key_points": list[str]}
```

### 4.2 What Dev 1 implements if Dev 3 is still stubs (required for Week 3 100%)

Create **`app/services/semantic.py`** (Dev 1 file — not `app/nlp/`, to avoid merge wars with Dev 3).

Behavior:

1. Try Dev 3: `match_topic` / `match_intent` / `search_duplicate_message`.
2. If result is `None`, empty, or raises, **do not** use fake confidence `0.50` labeled as success.
3. Run **Dev-1 local fallback**:
   - **Topic:** keyword map aligned with the 10 names Dev 3 *should* put in `topics.yaml` (billing, login, shipping, cancellation, etc.). Set `resolved_by="rules"`.
   - **Intent:** simple keyword/rules (`request_refund`, `report_bug`, `general_query`).
   - **Duplicate:** reuse `compute_cosine_similarity` on embeddings of current text vs… if Qdrant is down, return `is_duplicate=False`, `similarity_score=0.0` (honest). Pairwise duplicate remains `/similarity`.
4. Set `llm_fallback_used=True` **only** when Dev 3 provider actually ran.

### 4.3 Minimal `topics.yaml` seed (Dev 1 may add if file stays empty)

If `config/topics.yaml` is still empty when you start W3-D, Dev 1 adds a **10-topic stub file** (name + 2–3 example phrases each) so fallbacks and later Qdrant upsert have the same labels. Tell Dev 3: “edit this file, do not replace the keys.” That prevents schema/topic-name drift.

### 4.4 What Dev 1 must **not** build (still Dev 3)

- Real Qdrant client (`connect`, collections, upsert topics).
- Gemini/OpenAI `resolve_ambiguous()`.
- Streamlit UI (Day 16–21 UI). For Day 21 checkpoint, Dev 1 demos via `/docs` + curl until Streamlit exists.

### 4.5 Sentiment (Dev 2 field, Dev 1 merge)

README Day 17: Dev 2 adds sentiment; Dev 1 puts it on the response.  
Today `/analyze` hardcodes `neutral`. Dev 1 plan:

1. Ask Dev 2 for `analyze_sentiment(text) -> {sentiment, score}` (or use theirs if already in `linguistic.py` / `rules.py`).
2. If missing, Dev 1 adds a **tiny rule-based scorer** in `app/services/sentiment.py` (negation + word lists) so Week 3 JSON is not fake-neutral on “please refund immediately”. Replace when Dev 2 lands.

### Done when

- `/analyze` always fills topic/intent/duplicate/sentiment with honest `resolved_by`.
- Logs: `WARNING semantic fallback used: qdrant_unavailable` when Dev 3 stubs/fail.

---

## 5. WEEK 3 — Phases W3-C to W3-F (gateway completeness)

### W3-C `/extract` polish

**Files:** `app/api/extract.py`, `app/services/extractors.py`, `tests/test_extract_route.py`

1. Use `settings.MAX_FILE_SIZE_BYTES`.
2. Remove unused `Depends(get_db)` **or** insert an extract audit row (prefer remove — less schema churn).
3. Keep 400 empty, 413 oversized, 400 bad extension, `ExtractionError` → response `error` field (already there).
4. Apply the same size/type checks to `/analyze` file upload (today analyze has no 10 MB check).

### W3-D `/analyze` merge + SQLite

**Files:** `app/api/analyze.py`, `app/services/crud.py`, `app/schemas/__init__.py`, `tests/test_analyze_route.py`, `tests/test_pipeline.py`

1. Input:
   - Keep multipart `text` + `file` (tests already use form data).
   - Optionally add JSON body route or dual endpoint; if time is short, **document form-only** in OpenAPI description so Streamlit does not POST JSON and get 422.
2. Pipeline:
   - Prefer `run_local_nlp_pipeline(raw_text)` for cleaned_text, keywords, entities.
   - Evidence from `extract_evidence_sentences`.
   - Topic/intent/duplicate from `app/services/semantic.py`.
   - Sentiment from Dev 2 or Dev 1 rules.
3. Map entity dict keys (`text`/`entity`, `start`/`start_char`) in one helper — already started; keep it.
4. `create_analysis`: write `filename`, `file_type`, `word_count`, `char_count` from `doc_metadata`.
5. After successful commit, call `upsert_message_embedding(request_id, raw_text)` inside try/except (no-op if Dev 3 stub).
6. Export `DocumentMetadata` from `schemas/__init__.py`.
7. Optional but Week-3-nice: `GET /analyze/{request_id}` using `get_analysis_by_request_id` so you can prove DB round-trip without SQL tools.

### W3-E `/similarity` and `/summarize`

**Similarity:** already close. Ensure empty strings 422; clamp score `[0, 1]`; `is_duplicate` uses settings threshold. Add one test: identical texts → 1.0 / true; unrelated → below threshold.

**Summarize (Dev 1 local until provider exists):**

1. Try `from app.providers...` summarize if implemented.
2. Else extractive: `tokenize_sentences` + rank by keyword overlap / first N sentences, respect `max_length`.
3. Fill `original_word_count` / `summary_word_count` from real strings (not dummy paragraph).

### W3-F Tests and team checkpoint (Days 19–21, API only)

1. Fix tests that assume dummy topic names if you change defaults.
2. Add test: `/analyze` persists entities + evidence child rows, not only parent.
3. Add test: oversized file on `/analyze` → 413.
4. Day 19: run live `/analyze` with team; paste one JSON sample into Discord/WhatsApp.
5. Day 21: **Dev 1 pass** = curl/Postman: upload DOCX → `/extract` → paste text `/analyze` → row in SQLite. Streamlit is Dev 3; do not wait.

### Week 3 definition-of-done checklist

- [ ] `sqlalchemy` + `python-multipart` installed; app creates tables
- [ ] `MAX_FILE_SIZE_BYTES` in settings; both extract and analyze enforce it
- [ ] CRUD saves file metadata
- [ ] Semantic adapter + honest fallbacks; no silent 0.50 Qdrant
- [ ] Sentiment not hardcoded unless scorer really returned neutral
- [ ] `/summarize` extractive (or real LLM)
- [ ] Validation + 500 handlers; request logging
- [ ] pytest green for Dev 1 tests
- [ ] Handshake message sent to Dev 3; signature mismatch fixed

---

## 6. GitHub pull requests — conflicts and process

Assume shared `main`/`dev`, CI in `.github/workflows/ci.yml` (install requirements, spaCy model, flake8, pytest).

### 6.1 Branch and PR rules for Dev 1

| Rule | Why |
|---|---|
| One phase per PR | Conflict surface stays small |
| Branch names: `dev1/w3-db-startup`, `dev1/w3-semantic-adapter`, `dev1/w3-analyze-persist`, `dev1/w4-docker` | Easy to revert |
| Never mix Docker (W4) with analyze wiring (W3) | Review noise + conflict with empty Docker files Dev 3 might also fill |
| Do not commit `.env`, `nlp_gateway.db`, `__pycache__`, `.venv` | Secret/noise conflicts |
| Do not “complete” `app/nlp/vector_store.py` in a Dev 1 PR unless agreed | Guaranteed conflict with Dev 3 |
| Freeze `app/schemas/**` unless a field is truly wrong | Dev 2/3 build against schemas |

### 6.2 Files most likely to conflict

| File | Who touches it | How Dev 1 avoids pain |
|---|---|---|
| `app/api/analyze.py` | Dev 1 + Dev 3 wiring | Dev 1 only imports `semantic.py`; Dev 3 does not edit this file after handshake |
| `app/nlp/embeddings.py` | Dev 3 (already implemented) | Dev 1 only **calls** `get_embedding` / `compute_cosine_similarity` |
| `app/nlp/topics.py`, `vector_store.py` | Dev 3 | Dev 1 wrapper in `app/services/semantic.py` |
| `app/main.py` | Dev 1 | Small diffs: lifespan `create_all`, one extra handler |
| `requirements.txt` | All | Dev 1 PR first for sqlalchemy/multipart; others rebase |
| `app/config.py` | All | Only add `MAX_FILE_SIZE_BYTES`; do not rename Qdrant keys |
| `app/schemas/analysis.py` | Frozen | Avoid edits; if needed, extra optional field with default |
| `config/topics.yaml` | Dev 3 | If Dev 1 seeds it, PR **only** that YAML + tell Dev 3 to pull |
| `Dockerfile`, `docker-compose.yml` | Dev 1 Week 4 | Currently empty — first writer wins; coordinate “Dev 1 owns Docker” |

### 6.3 Before every PR

```text
git fetch origin
git checkout <your-branch>
git rebase origin/main          # or origin/dev — match the default branch
pytest                          # at least tests/test_extract*.py tests/test_analyze*.py tests/test_db_crud.py
git push --force-with-lease     # only if you rebased a branch you own
```

If rebase stops: fix **your** hunks; never delete Dev 2 NLP function bodies to “make it compile.”

### 6.4 Conflict playbook (when GitHub shows conflicts)

1. **Do not** use “Resolve by discarding theirs” on `app/nlp/*`.
2. `analyze.py`: keep Dev 1 route/validation/DB; keep Dev 3 function **calls** if they match the handshake; route calls through `semantic.py`.
3. `requirements.txt`: union of packages, sorted sections (gateway / NLP / embeddings / extract / LLM / UI). Duplicate pins → take the higher compatible version, then `pip install` once.
4. `main.py`: keep both routers **and** both middlewares; merge `lifespan` so `create_all` and any Dev 3 Qdrant `init_collections()` both run (Qdrant init in try/except so API still boots).
5. After resolve: run pytest; if spaCy/NLTK missing in CI, that is CI setup not a logic conflict.

### 6.5 CI failures Dev 1 should expect

| Failure | Likely cause | Dev 1 fix |
|---|---|---|
| `ModuleNotFoundError: sqlalchemy` | Missing in requirements | W3-A PR |
| `Form data requires python-multipart` | Same | W3-A |
| Import error `match_topic` returns None → Pydantic | No fallback | W3-B adapter |
| Flake8 E9/F82 | Unused imports (`analyze_linguistics`) | Delete unused |
| Tests hit real SQLite file | `test_pipeline.py` uses app engine not fixture | Align with `conftest.py` override |
| spaCy download timeout | CI network | Already in workflow; don’t add extra downloads in Docker until W4 |

### 6.6 Suggested merge order (tell the team)

1. Dev 1 W3-A (deps + create_all) → `main`
2. Dev 2 NLP PRs (preprocess/linguistic/tfidf/ner) → `main`
3. Dev 1 W3-B/C/D (adapter + extract + analyze) → `main`
4. Dev 3 Qdrant/topics (swap fallbacks for real search without changing `analyze.py` if adapter already calls their functions)
5. Dev 1 W3-E summarize + tests
6. Dev 3 Streamlit
7. **Only then** Dev 1 Week 4 Docker PR

If Dev 3 merges empty `Dockerfile` with a comment, Dev 1 Week 4 PR replaces contents — announce ownership: **“Dev 1 owns Dockerfile and docker-compose.yml.”**

### 6.7 Review comments to expect on Dev 1 PRs

- Dual mount `/analyze` and `/api/v1/analyze` — keep; Streamlit should use one base URL.
- `sys.path.insert` in services — leftover; remove in W3-A to avoid import-order bugs.
- Swallowing all exceptions in `/analyze` — replace with adapter + logging (W3-B).

---

## 7. WEEK 4 — only after Week 3 checklist is green

### W4-A Docker

1. `Dockerfile`: Python 3.10-slim; gcc for spaCy/PyMuPDF; `pip install -r requirements.txt`; `python -m spacy download en_core_web_sm`; NLTK punkt/stopwords; `CMD uvicorn app.main:app --host 0.0.0.0 --port 8000`.
2. `docker-compose.yml`:
   - `qdrant` image `qdrant/qdrant`, ports 6333
   - `api` build `.`, depends_on qdrant, env `QDRANT_URL=http://qdrant:6333`, volume for SQLite
3. Local: `docker compose up --build`; `/health`; `/extract`; `/analyze`.
4. If Qdrant collection missing, API still returns rules fallback (Week 3 adapter). Dev 3 adds upsert-on-start later.

### W4-B Deploy

1. Render (or similar) web service from Dockerfile or `uvicorn` start command.
2. Disk or ephemeral SQLite: document that SQLite resets on free-tier unless a disk is attached.
3. Prod `.env`: `QDRANT_URL` / `QDRANT_API_KEY` for Qdrant Cloud — **Dev 3 creates the cloud collection**; Dev 1 only plugs URLs.
4. CORS already `allow_origins=["*"]` — tighten later if Streamlit has a fixed URL.

### W4-C Docs and demo

1. README: install, `.env.example`, `docker compose`, curl examples for four routes, “NLP first / LLM fallback.”
2. OpenAPI is already `/docs` — add 3 copy-paste examples.
3. Bug bash: deployed `/analyze` + `/extract`; if Streamlit exists, point it at deployed API.
4. Do not rewrite `docs/dev1_shreya.md` as if Week 4 were done until compose actually runs.

---

## 8. Day-by-day execution (compressed, Weeks 3 then 4)

Use this if the calendar is still on README days 15–28.

| Block | Dev 1 focus | Depends on Dev 3? |
|---|---|---|
| Days 15–16 | W3-A + start W3-B adapter; call Dev 2 `run_local_nlp_pipeline` | No |
| Day 17 | Persist full JSON + sentiment merge + file metadata | No (fallback topic OK) |
| Day 18 | `/similarity` tests; extractive `/summarize` | Embeddings exist; LLM optional |
| Day 19 | Team live `/analyze`; fix schema mismatches | Prefer Dev 3 topic fn; fallback OK |
| Day 20 | Validation handler, logging, health mentions DB | No |
| Day 21 | File → extract → analyze → SQLite (curl). Streamlit optional | Streamlit is Dev 3 |
| Days 22–23 | W4 Docker + compose | Qdrant container; collections can be empty |
| Day 24 | Deploy; env to Qdrant Cloud if Dev 3 ready | Cloud instance is Dev 3 |
| Days 25–28 | README, bug bash, viva | UI/video Dev 3 |

---

## 9. Communication scripts (Dev 1)

**To Dev 3 (blocker):**  
“Please implement `match_topic(text)`, `match_intent(text)`, `search_duplicate_message(text)` returning dicts matching `TopicMatch` / `IntentMatch` / `DuplicateCheck`. Change the vector-only stub — `/analyze` passes a string. Fill `config/topics.yaml` (10 topics). I will keep a rules fallback so the API demos if Qdrant is down. Do not edit `app/api/analyze.py`; I will import your functions from `app/services/semantic.py`. I own `Dockerfile`.”

**To Dev 2:**  
“Need `analyze_sentiment` or I ship a temporary lexicon scorer. Please keep `extract_keywords` returning `{"keyword", "score"}` and entities with `text`/`label`/`start_char`/`end_char`. I will call `run_local_nlp_pipeline`.”

---

## 10. Out of scope for this plan

- Training custom models  
- Redis/Kafka  
- Fully replacing Dev 3 Streamlit or Qdrant client  
- Force-pushing `main`

When Week 3 checklist is checked, start Section 7. Until then, ignore empty Docker files except to claim ownership in the team chat.
