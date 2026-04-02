# Voice RAG — Phase 1 (Stable)

## What changed from the starter

| Problem | Fix |
|---|---|
| Server restart wiped all indexed documents | FAISS index + metadata saved to `data/` on every upload |
| Two simultaneous uploads corrupted the index | `asyncio.Lock()` wraps every read/write of shared state |
| Bad PDF crashed the whole server | `try/except` around all I/O — returns HTTP 422 with a clear message |
| Ollama hanging forever if unreachable | `request_timeout=60` on the LLM; WebSocket has `asyncio.wait_for` 30s timeout |
| `langchain-ollama` missing from requirements | Added + all deps pinned |
| `ingestion.py` loaded a duplicate embedding model | Removed — `main.py` owns the single embedder instance |
| Committed `__pycache__` bytecode | Added to `.gitignore` |
| No file size or type validation | 20 MB hard cap + allowlist of pdf/docx/txt |
| `/health` endpoint missing | Added — returns chunk count, useful for uptime checks |

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure Ollama is running with the right model
ollama serve                    # in one terminal
ollama pull llama3.2:3b         # first time only

# 3. Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

## Project structure

```
voice_rag/
├── main.py               ← FastAPI app, all routes
├── rag/
│   ├── __init__.py
│   ├── retrieval.py      ← FAISS search
│   └── generation.py     ← Ollama LLM call
├── static/
│   └── index.html        ← Frontend (unchanged)
├── data/                 ← Auto-created at first upload
│   ├── index.faiss       ← Persisted vector index
│   └── metadata.pkl      ← Chunk texts + doc names
├── requirements.txt
└── .gitignore
```

## Key environment notes

- The `data/` folder is created automatically on first upload.
- Deleting `data/` resets the index completely (clean slate).
- To switch LLM models, change the `model=` line in `rag/generation.py`.
- Phase 2 will replace Ollama with the Claude API for much better answers.
