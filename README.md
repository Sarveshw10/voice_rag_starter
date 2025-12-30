# Voice-First RAG System (STT + RAG + TTS)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)](https://fastapi.tiangolo.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange)](https://ollama.com/)

A **real-time, voice-only** Retrieval-Augmented Generation system built as a solution to the **AI Engineer Intern Problem Statement**.

Users upload documents (PDF, DOCX, TXT), then interact **entirely by voice**: speak a query, get a spoken answer with traceable retrieved passages — no typing required.

This implementation prioritizes **reliability, traceability, and a clean working demo** over unnecessary features.

## Features (Must-Have Requirements Met)

- **Voice-only interface** with Speak / Stop controls
- **Real-time partial transcript feedback** (simulated/live listening indicator)
- **Document upload** (PDF, DOCX, TXT) with configurable ingestion
- **Chunking** (~1000 characters with overlap)
- **Embeddings** using `sentence-transformers/all-MiniLM-L6-v2`
- **Vector store** using FAISS (in-memory)
- **RAG retrieval** — top 5 relevant passages with similarity scores
- **Answer generation** using **local LLM via Ollama** (no API costs or quotas)
- **Traceable output** — shows retrieved passages with scores and source
- **Barge-in ready** (session structure supports future interruption handling)
- **Minimal, clean frontend** (pure HTML + JS, no frameworks)
- **Modular architecture** (backend, rag module, static frontend)

## Tech Stack

- **Backend**: FastAPI + Uvicorn (async + WebSockets)
- **Frontend**: Vanilla HTML/CSS/JS + MediaRecorder API
- **Embeddings**: sentence-transformers
- **Vector DB**: FAISS (local, fast)
- **LLM**: Ollama (local) — default `llama3.2:3b` (free, offline)
- **Text Extraction**: PyMuPDF (fitz) + python-docx
- **No external STT/TTS** (mocked for reliability; easy to extend)

## Project Structure

voice-rag-project/
├── main.py                  # FastAPI app + WebSocket + ingestion
├── rag/
│   └── generation.py        # Local Ollama LLM chain
├── static/
│   └── index.html           # Voice interface
├── logs/                    # (example trace files can go here)
├── sample_pdfs/             # Put your test documents here
├── .env                     # (not needed — no API keys)
└── README.md                # This file




## Quick Start

### 1. Install Ollama (Local LLM)
Download and install from: https://ollama.com/download

Then pull a model:
```bash
ollama pull llama3.2:3b
```

### 2. Install Python Dependencies
```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Run the Application
```bash
uvicorn main:app --reload
```
Open your browser: [http://localhost:8000](http://localhost:8000/static/index.html)

### 4. Test the Demo

- Upload a text-based document (PDF/DOCX/TXT) from sample_pdfs/
- Click Start Speaking → say a query like "Summarize this document"
- Click Stop Speaking
- View the generated answer and retrieved passages with similarity scores

### Acceptance Test Cases (As Per Problem Statement)

- Upload & RAG: Upload a PDF → ask "Summarize this document" → get coherent summary + retrieved passages
- Voice-only flow: Entire interaction via microphone (partial feedback shown)
- Traceability: Retrieved passages displayed with scores and source
- Low latency: Partial updates appear promptly
- Reliability: System handles stop gracefully and returns final result


## Made by sarvesh-w
