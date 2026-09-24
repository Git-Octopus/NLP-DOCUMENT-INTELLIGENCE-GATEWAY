# ⚡ NLP Document Intelligence Gateway

> **High-Performance Multi-Tier NLP API & Analytics Dashboard for Customer Support Intelligence**  
> Combines deterministic local NLP (spaCy, NLTK, scikit-learn) with dense vector search (Qdrant) and adaptive local/cloud LLM fallbacks (Ollama & Google Gemini).

---

## 📌 Executive Summary & Philosophy

The **NLP Document Intelligence Gateway** processes raw customer support communications (emails, tickets, chats, and uploaded documents) and transforms them into structured intelligence:
- **Controlled 10-Topic Taxonomy Classification** (Qdrant semantic search)
- **Intent Detection** (Regex rules + Qdrant similarity + LLM fallback)
- **Dual-Tier Sentiment Analysis** (Calibrated VADER + support domain lexicon + LLM fallback)
- **Named Entity Recognition (NER)** (spaCy pretrained models + high-precision regex extractors)
- **TF-IDF Keyword Extraction & Evidence Sentence Mining**
- **Vector-Based Duplicate Message Detection** (Cosine thresholding in Qdrant)
- **Multi-Format Document Text Extraction** (PDF via PyMuPDF, DOCX via python-docx)

### 💡 Why This Project?
- **Avoid Costly LLM Overuse**: Sending every support ticket to cloud LLMs creates high latency, large monthly API bills, and privacy risks.
- **Explainable & Fast**: Classical NLP (spaCy/NLTK) resolves **>85% of queries locally in sub-milliseconds** at zero operational cost.
- **Best of Both Worlds**: Pairs high-speed local processing with local/cloud LLMs (`Ollama`/`Gemini`) as a safety net only when queries are ambiguous.

### ⚙️ How It Works (In Short)
1. **Ingest & Clean**: Customer messages or uploaded files (`.pdf`/`.docx`) are normalized, stripped of noise, and tokenized locally.
2. **Local Analysis First**: Extracts key entities (money, emails, orders, dates), TF-IDF keywords, evidence sentences, and sentiment (calibrated VADER + support lexicon) in real-time.
3. **Semantic Matching**: Embeds the text with Sentence Transformers and finds the nearest topic in the **Qdrant Vector DB** while checking for duplicate tickets.
4. **Adaptive Fallback**: If topic similarity is low (`< 0.58`) or sentiment is ambiguous, the gateway automatically asks **Ollama** (`qwen2.5:7b`) or **Gemini** to resolve the nuance.

---

## 🏛️ System Architecture

```mermaid
graph TD
    User([Customer Ticket / Uploaded PDF / DOCX]) --> Gateway[FastAPI Gateway :8000]

    subgraph Tier 1: Local Classical NLP Engine
        Gateway --> Preprocess[Text Normalization & Tokenization]
        Preprocess --> Linguistic[Lemmatization & POS Tagging]
        Preprocess --> NER[Hybrid NER: spaCy + Regex Extractors]
        Preprocess --> TFIDF[TF-IDF N-Gram Keywords & Evidence Sentences]
        Preprocess --> Sentiment[Domain-Calibrated VADER & Support Lexicon]
    end

    subgraph Tier 2: Dense Semantic Vector Store
        Preprocess --> Embed[Sentence Transformers: all-MiniLM-L6-v2 / Ollama nomic-embed-text]
        Embed --> Qdrant[(Qdrant Vector DB :6333)]
        Qdrant -->|Cosine >= 0.58| TopicMatch[Classified Topic & Intent]
        Qdrant -->|Deduplication| DuplicateCheck[Duplicate Check Flag]
    end

    subgraph Tier 3: Adaptive LLM Fallback Brain
        Qdrant -.->|Confidence < 0.58| LLM[Host Ollama: qwen2.5:7b / Google Gemini]
        Sentiment -.->|Ambiguous / Neutral| LLM
        LLM --> AmbiguityResolver[Nuanced Topic, Intent & Sentiment Resolution]
    end

    Tier 1 --> Aggregator[Unified Structured Output Schema]
    TopicMatch --> Aggregator
    DuplicateCheck --> Aggregator
    AmbiguityResolver --> Aggregator

    Aggregator --> SQLite[(SQLite Audit DB)]
    Aggregator --> Client([Structured JSON / Streamlit UI :8501])
```

---

## 📂 Project Structure

```
NLP-DOCUMENT-INTELLIGENCE-GATEWAY/
├── app/
│   ├── main.py                      # FastAPI application entrypoint & lifespan
│   ├── config.py                    # Pydantic v2 settings & environment configuration
│   ├── api/
│   │   ├── analyze.py               # POST /analyze (unified ticket intelligence)
│   │   ├── extract.py               # POST /extract (PDF/DOCX parsing)
│   │   ├── similarity.py            # POST /similarity (semantic text comparison)
│   │   ├── summarize.py            # POST /summarize (extractive & LLM summarization)
│   │   └── models.py                # GET /models & POST /models/config (runtime model manager)
│   ├── nlp/
│   │   ├── preprocess.py            # Regex cleaning, stopwords, sentence/word tokenizers
│   │   ├── linguistic.py            # spaCy lemmatization, POS tags, dependency parsing
│   │   ├── ner.py                   # Combined spaCy NER + regex rule extractors
│   │   ├── tfidf.py                 # scikit-learn TF-IDF n-grams & keyword scorer
│   │   ├── rules.py                 # Evidence sentence extraction & patterns
│   │   ├── sentiment.py             # Calibrated VADER + support polarity lexicons
│   │   ├── embeddings.py            # SentenceTransformer & Ollama embedding adapter
│   │   ├── vector_store.py          # Qdrant client wrapper & collection managers
│   │   └── topics.py                # Taxonomy loader & similarity matching
│   ├── providers/
│   │   ├── base.py                  # Abstract BaseLLMProvider interface
│   │   ├── ollama_provider.py       # Host Ollama integration (qwen2.5, gemma3, etc.)
│   │   ├── gemini_provider.py       # Google Gemini Flash/Pro LLM adapter
│   │   └── openai_provider.py       # OpenAI GPT adapter (optional stub)
│   ├── schemas/                     # Pydantic data validation schemas
│   │   ├── analysis.py              # AnalyzeRequest, AnalyzeResponse, TopicMatch, etc.
│   │   ├── extract.py               # DocumentExtractionResponse
│   │   ├── models.py                # ModelConfigRequest, AvailableModelsResponse
│   │   └── summarize.py             # SummarizeRequest, SummarizeResponse
│   └── services/
│       ├── crud.py                  # SQLAlchemy persistence helpers
│       ├── database.py              # SQLite engine & Base model
│       ├── extractors.py            # PyMuPDF (PDF) & python-docx byte stream parsers
│       ├── model_manager.py         # Runtime dynamic model catalog & discovery
│       └── semantic.py              # Safe fallback adapters for Qdrant/LLM
├── config/
│   └── topics.yaml                  # Controlled 10-Topic Taxonomy & anchor sentences
├── streamlit_app/
│   └── app.py                       # Interactive Streamlit analytics dashboard
├── tests/                           # 100+ pytest unit and integration tests
├── Dockerfile                       # Optimized lightweight CPU-only container image
├── docker-compose.yml               # Multi-container orchestration (Gateway, UI, Qdrant)
├── requirements.txt                 # Production dependencies
├── requirements-dev.txt             # Testing & linting tools (flake8, black, pytest)
└── README.md
```

---

## 🏷️ Controlled 10-Topic Taxonomy

Defined in [`config/topics.yaml`](config/topics.yaml), each topic contains curated intent tags and anchor sentences embedded into Qdrant for semantic similarity classification:

| # | Topic Label | Focus Area | Example Intent Tags |
|---|---|---|---|
| 1 | **Payment** | Card declines, failed checkouts, gateway timeouts | `duplicate_payment`, `failed_transaction` |
| 2 | **Billing & Refunds** | Money reversals, refund status, double charges | `request_refund`, `delayed_refund` |
| 3 | **Invoice & Billing** | Invoice discrepancies, tax/GST details, receipts | `request_invoice`, `incorrect_amount` |
| 4 | **Order Tracking & Shipping** | Tracking numbers, delivery delays, missing parcels | `track_order`, `delayed_delivery` |
| 5 | **Returns & Exchanges** | Defective items, replacements, RMA labels | `return_item`, `report_damaged_item` |
| 6 | **Account & Login** | Password resets, account lockouts, 2FA issues | `reset_password`, `account_locked` |
| 7 | **Subscription & Plans** | Plan cancellations, auto-renewals, tier upgrades | `cancel_subscription`, `upgrade_plan` |
| 8 | **Technical Issue & Bugs** | App crashes, UI freezes, error codes | `report_bug`, `app_crash` |
| 9 | **General Inquiry** | Operating hours, store policies, contact channels | `general_info`, `business_hours` |
| 10 | **Feedback & Complaints** | Service feedback, agent escalations, compliments | `file_complaint`, `give_feedback` |

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+** (Tested on Python 3.12)
- **Docker Desktop** (for containerized deployment)
- *(Optional)* **Ollama** installed on your host machine for local LLM inference

---

### Method 1: Docker Compose (Recommended)

The project includes an optimized Docker setup. The Dockerfile uses lightweight CPU-only PyTorch wheels (~180MB instead of 2.5GB+ CUDA bundles) to ensure fast, minimal-footprint container builds.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Git-Octopus/NLP-DOCUMENT-INTELLIGENCE-GATEWAY.git
   cd NLP-DOCUMENT-INTELLIGENCE-GATEWAY
   ```

2. **Configure Environment Variables:**
   Copy the template and provide keys if using cloud providers:
   ```bash
   cp .env.example .env
   ```
   *(Optional: Add `GEMINI_API_KEY=your_key` in `.env` if you want Google Gemini fallback enabled).*

3. **Build and Launch Containers:**
   ```bash
   docker compose up --build
   ```

4. **Access the Services:**
   - 🌐 **Interactive Streamlit Dashboard**: [http://localhost:8501](http://localhost:8501)
   - ⚡ **FastAPI Swagger API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - 📦 **Qdrant Vector Database Web UI**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

---

### Method 2: Native Local Installation (Development)

1. **Create and Activate Virtual Environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate   # On Linux/macOS: source venv/bin/activate
   ```

2. **Install Dependencies & Download NLP Assets:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('vader_lexicon', quiet=True)"
   ```

3. **Start Qdrant Vector DB (Docker):**
   ```bash
   docker run -d -p 6333:6333 -p 6334:6334 --name nlp_qdrant qdrant/qdrant:latest
   ```

4. **Launch the FastAPI Gateway:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Launch the Streamlit Frontend (In a separate terminal):**
   ```bash
   streamlit run streamlit_app/app.py
   ```

---

## 🦙 Host Ollama Auto-Discovery

The gateway natively auto-detects and connects to Ollama running on your host machine (`127.0.0.1:11434` natively or `host.docker.internal:11434` inside Docker Compose).

1. Ensure Ollama is running on your machine:
   ```bash
   ollama list
   ```
2. Pull recommended embedding and reasoning models:
   ```bash
   ollama pull nomic-embed-text:latest
   ollama pull qwen2.5:7b
   ```
3. Open the **Streamlit Dashboard** (`http://localhost:8501`). The sidebar will automatically show `Ollama: 🟢 Online` and populate dropdowns with your installed host models!

---

## 🔌 API Endpoints Reference

### 1. `POST /analyze` — Full Document Intelligence
Accepts either JSON text input or uploaded document (`.pdf` / `.docx`).

**Request Example (JSON):**
```json
{
  "text": "My Visa card was charged $49.99 twice for order #ORD-8812. Please issue a refund.",
  "metadata": {"channel": "email"}
}
```

**Response Example:**
```json
{
  "request_id": "req-9b81f1e2",
  "original_text": "My Visa card was charged $49.99 twice for order #ORD-8812. Please issue a refund.",
  "cleaned_text": "my visa card was charged 49.99 twice for order ord 8812 please issue a refund",
  "topic": {
    "topic": "Billing & Refunds",
    "confidence": 0.924,
    "resolved_by": "qdrant"
  },
  "intent": {
    "intent": "request_refund",
    "confidence": 0.92,
    "resolved_by": "rules"
  },
  "sentiment": {
    "sentiment": "negative",
    "score": -0.65,
    "resolved_by": "local_nlp"
  },
  "entities": [
    {"text": "$49.99", "label": "MONEY", "source": "regex"},
    {"text": "ORD-8812", "label": "ORDER_ID", "source": "regex"},
    {"text": "Visa", "label": "ORG", "source": "spacy"}
  ],
  "keywords": ["charged twice", "visa card", "refund", "order"],
  "evidence_sentences": [
    "My Visa card was charged $49.99 twice for order #ORD-8812."
  ],
  "duplicate_info": {
    "is_duplicate": false,
    "similarity_score": 0.0
  }
}
```

### 2. `POST /extract` — Document Text Extraction
Extracts plain text and token metadata from uploaded PDF or Word documents.

### 3. `POST /similarity` — Semantic Text Similarity
Computes cosine similarity between two text snippets using the active dense embedding model.

### 4. `POST /summarize` — Support Ticket Summarization
Generates concise executive summaries and bulleted key points using local extractive rankers or LLM fallback.

### 5. `GET /models` & `POST /models/config` — Runtime Model Selection
Enables dynamic, zero-restart switching of embedding providers (Sentence Transformer vs. Ollama) and reasoning providers (Ollama vs. Gemini).

---

## 🧪 Testing & Code Quality

The repository enforces strict linting, formatting, and unit/integration testing across all modules:

```powershell
# 1. Check code formatting with Black
python -m black --check .

# 2. Verify PEP8 compliance with Flake8
python -m flake8 .

# 3. Run complete test suite (100+ tests)
python -m pytest
```

---

## 👥 Academic Team Roles & Responsibilities

| Contributor | Area of Ownership | Technology Stack |
|---|---|---|
| **Dev 1** | Gateway API, Pydantic Schemas, DB Persistence, Docker Containers | FastAPI, Pydantic v2, SQLite, SQLAlchemy, PyMuPDF, python-docx, Docker |
| **Dev 2** | Core Local NLP Pipeline, Tokenization, TF-IDF, NER, Sentiment Engine | spaCy, NLTK, scikit-learn, VADER, Regular Expressions |
| **Dev 3** | Semantic Matching, Vector Store, LLM Fallback Integration, Streamlit UI | Sentence Transformers, Qdrant Vector DB, Ollama, Gemini API, Streamlit |

---

## 📜 License
Distributed under the MIT License for academic and educational evaluation.