# 📄 AskMyDocs — Multi-Document RAG Chatbot

An AI-powered document Q&A platform built using Python, Flask, LangChain, ChromaDB, and Groq. The application lets users upload multiple documents and ask natural-language questions, getting answers that are strictly grounded in the uploaded content — with page-level source citations and zero hallucination.

---

## 🚀 Features

### 📁 Document Intelligence
- Upload multiple PDF, DOCX, or TXT files at once
- Automatic chunking and embedding of all uploaded content
- Semantic search across all documents simultaneously
- Source citations with filename, page number, and matched excerpt on every answer

### 💬 Conversational Chat
- Natural follow-up questions ("What about medical leave?" after asking about leave policy)
- Session-based conversation memory — remembers context across the chat
- Live token-by-token streaming responses, just like ChatGPT
- "New Chat" to reset conversation while keeping documents loaded

### 🚫 Grounded, No-Hallucination Answers
- Answers generated strictly from retrieved document chunks
- Explicitly refuses to answer questions outside the uploaded content
- `temperature=0` for deterministic, factual responses every time

### 🩺 Production-Minded Engineering
- `check_groq_model.py` — verifies the configured LLM hasn't been deprecated before deploying
- Cached embedding model (singleton pattern) to control memory usage on free-tier hosting
- Locally bundled embedding model (no runtime download, no Hub rate limits)
- Tuned Gunicorn worker config (`--preload`, `--max-requests`) to avoid memory creep
- Pinned dependency versions for reproducible builds across machines

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Groq — `openai/gpt-oss-20b` |
| **RAG Framework** | LangChain 0.3.27 (LCEL + `ConversationalRetrievalChain`) |
| **Embeddings** | sentence-transformers — `all-MiniLM-L6-v2` (bundled locally, CPU-only torch) |
| **Vector Store** | ChromaDB — persistent, HNSW index |
| **Document Parsing** | PyPDFLoader · Docx2txtLoader · TextLoader |
| **Backend** | Flask + Gunicorn (SSE streaming, session memory) |
| **Frontend** | Vanilla HTML / CSS / JS — separated files, no framework |

---

## 🏗 Architecture

```
User uploads PDF / DOCX / TXT
         │
         ▼
┌─────────────────────┐
│  LangChain Loaders  │  PyPDFLoader · Docx2txtLoader · TextLoader
└────────┬────────────┘
         │ raw text
         ▼
┌─────────────────────┐
│   Text Splitter     │  RecursiveCharacterTextSplitter
│  chunk_size = 1000  │  chunk_overlap = 200
└────────┬────────────┘
         │ chunks (Document objects)
         ▼
┌─────────────────────┐
│  HuggingFace        │  all-MiniLM-L6-v2 (loaded from local_models/,
│  Embeddings         │  cached singleton, no runtime download)
└────────┬────────────┘
         │ vectors
         ▼
┌─────────────────────┐
│  ChromaDB             │  persistent vector store
│  Vector Store         │  HNSW index · cosine similarity
└────────┬────────────┘
         │
    User asks question
         │
         ▼
┌─────────────────────┐
│  Question Condenser  │  ConversationalRetrievalChain
│  (memory-aware)      │  rewrites follow-ups as standalone questions
└────────┬────────────┘
         │ condensed question
         ▼
┌─────────────────────┐
│  ChromaDB Search     │  top-4 chunks by cosine similarity
└────────┬────────────┘
         │ chunks + metadata
         ▼
┌─────────────────────┐
│  RAG Prompt Template │  "Answer ONLY from the context below..."
│  + Groq LLM           │  openai/gpt-oss-20b · temperature=0
└────────┬────────────┘
         │ streamed tokens + sources
         ▼
┌─────────────────────┐
│  Flask SSE Stream     │  GET /stream → text/event-stream
│  + Chat UI             │  live answer + page citations
└─────────────────────┘
```

---

## 📁 Project Structure

```
rag-chatbot/
├── app.py                    # Flask app — routes for upload, chat, stream, reset
├── requirements.txt          # Pinned dependencies (CPU-only torch)
├── render.yaml                # Render deployment blueprint
├── .python-version             # Pins Python 3.11.9 on Render
├── .env.example                # Environment variable template
├── .gitignore
├── check_groq_model.py         # Verifies configured Groq model is still active
│
├── local_models/
│   └── all-MiniLM-L6-v2/       # Embedding model bundled locally — no runtime download
│
├── rag/
│   ├── __init__.py
│   ├── loader.py              # Document loading + chunking
│   ├── embedder.py            # ChromaDB store, search, cached embeddings
│   └── chain.py                # ConversationalRetrievalChain + memory + streaming
│
├── templates/
│   └── index.html              # HTML structure only
│
├── static/
│   ├── css/
│   │   └── style.css           # All styling
│   └── js/
│       └── chat.js              # Upload, SSE streaming, citations logic
│
├── documents/                  # Sample documents for testing
│   └── company_policy_sample.pdf
│
└── test_loader.py, test_embedder.py, test_chain.py, test_setup.py
```

---

## ⚙️ Getting Started (Local)

### Prerequisites
- Python 3.11+
- Free Groq API key from [console.groq.com](https://console.groq.com)

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot

python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt

cp .env.example .env
# add your GROQ_API_KEY to .env
```

### Run

```bash
python app.py
```
Visit `http://localhost:5000`

### Health check before deploying

```bash
python check_groq_model.py
```
Verifies your configured Groq model hasn't been deprecated — Groq retires models with little notice, and this catches it before a production crash.

---

## 🔑 Key Design Decisions

**RAG over fine-tuning** — retrieves fresh information at query time instead of baking knowledge into model weights. No retraining needed when documents change.

**Groq over OpenAI** — fast LPU-based token streaming with a usable free tier; the LangChain API is close enough to OpenAI's that switching providers later is a one-line change.

**Pinned `langchain==0.3.27`** — LangChain's v1.0 release restructured core imports (`langchain.chains`, `langchain.memory`). Pinning to the last stable 0.3.x line keeps the project reproducible on any machine.

**`check_groq_model.py`** — added after the originally-used model (`llama3-8b-8192`) was decommissioned mid-project. Hits Groq's `/models` endpoint to confirm the configured model is still active before deploying.

**`.python-version` over `runtime.txt`** — `runtime.txt` is deprecated and silently ignored by Render. Discovered this after Render defaulted to Python 3.14, which broke LangChain's pydantic-based class definitions.

**Bundled embedding model locally** — Render's free tier (512MB) couldn't reliably download `all-MiniLM-L6-v2` from HuggingFace Hub at runtime without hitting memory limits or rate limits. Shipping the model files directly in the repo removes the runtime download entirely.

**ChromaDB over Pinecone** — zero-config local vector store, ideal for a portfolio project. `CHROMA_DIR` points to `/tmp/chroma_db` on Render since the filesystem resets between deploys.

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | required | From console.groq.com |
| `SECRET_KEY` | `dev-secret` | Flask session encryption |
| `CHROMA_DIR` | `chroma_db` | Vector DB path (`/tmp/chroma_db` on Render) |

| Constant | Location | Default |
|----------|----------|---------|
| `chunk_size` | `rag/loader.py` | 1000 |
| `chunk_overlap` | `rag/loader.py` | 200 |
| `TOP_K` | `rag/embedder.py` | 4 |
| `model` | `rag/chain.py` | `openai/gpt-oss-20b` |

---



## 📄 License

MIT License
