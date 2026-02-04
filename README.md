# Voice-First RAG System (Browser STT + RAG + TTS)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)](https://fastapi.tiangolo.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange)](https://ollama.com/)

A **real-time, voice-only** Retrieval-Augmented Generation system built as a solution to the **AI Engineer Intern Problem Statement**.

Users upload documents (PDF, DOCX, TXT), then interact **entirely by voice**: speak a query, get a spoken answer with traceable retrieved passages — no typing required.

This implementation prioritizes **reliability, traceability, and a clean working demo** over unnecessary features.

## Features (Must-Have Requirements Met)

- **Voice-only interface** — Speak / Stop buttons + live transcript feedback
- **Browser-based real-time STT** — uses Web Speech API (partial + final transcripts shown live)
- **Document upload** — PDF, DOCX, TXT (text extraction + chunking)
- **Chunking** — ~1000 characters with 20% overlap
- **Embeddings** — `sentence-transformers/all-MiniLM-L6-v2`
- **Vector store** — FAISS (in-memory)
- **RAG retrieval** — top 5 relevant passages with cosine similarity scores + source
- **Answer generation** — local LLM via **Ollama** (no API costs, no quotas)
- **Traceable output** — shows retrieved passages with scores and doc/chunk info
- **Barge-in ready** — session structure supports future interruption handling
- **Minimal, clean frontend** — pure HTML + JS (no frameworks)
- **Modular architecture** — backend, rag module, static frontend

## Tech Stack

- **Backend** — FastAPI + Uvicorn (async + WebSockets)
- **Frontend** — Vanilla HTML/CSS/JS + **Web Speech API** (browser STT)
- **Embeddings** — sentence-transformers
- **Vector DB** — FAISS (local, fast)
- **LLM** — Ollama (local) — default `llama3.2:3b` (free, offline)
- **Text Extraction** — PyMuPDF (fitz) + python-docx
- **No external STT/TTS** — browser STT + text-only output (TTS easy to add later)

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
