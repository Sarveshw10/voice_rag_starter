# main.py - Phase 4: Streaming, confidence threshold, persistent session ID
import os
import uuid
import json
import asyncio
import logging
import pickle
import time
from collections import defaultdict
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from sentence_transformers import SentenceTransformer
import fitz
from docx import Document
import io
import faiss
import numpy as np

from rag.retrieval import retrieve_chunks
from rag.generation import generate_answer_stream, rewrite_query

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DIMENSION             = 384
PERSIST_DIR           = Path("data")
FAISS_PATH            = PERSIST_DIR / "index.faiss"
META_PATH             = PERSIST_DIR / "metadata.pkl"
MAX_FILE_BYTES        = 20 * 1024 * 1024
ALLOWED_EXTS          = {"pdf", "docx", "txt"}
MAX_HISTORY_TURNS     = 4
CONFIDENCE_THRESHOLD  = 0.45   # below this → refuse to answer instead of hallucinating

MAGIC_BYTES = {
    "pdf":  [(0, b"%PDF")],
    "docx": [(0, b"PK\x03\x04")],
    "txt":  [],
}

# ── Rate limiter ──────────────────────────────────────────────────────────────
_rate_store: dict[str, list[float]] = defaultdict(list)
_rate_lock  = asyncio.Lock()
RATE_LIMITS = {"upload": (5, 60), "ask": (10, 60)}

async def _check_rate_limit(ip: str, action: str) -> None:
    max_calls, window = RATE_LIMITS[action]
    now = time.monotonic()
    async with _rate_lock:
        key = f"{ip}:{action}"
        _rate_store[key] = [t for t in _rate_store[key] if now - t < window]
        if len(_rate_store[key]) >= max_calls:
            raise HTTPException(status_code=429,
                detail=f"Too many {action} requests. Please wait a moment.")
        _rate_store[key].append(now)

# ── Global state ──────────────────────────────────────────────────────────────
store_lock = asyncio.Lock()
index:    faiss.IndexFlatL2   = None
chunks:   list[str]           = []
metadata: list[dict]          = []
embedder: SentenceTransformer = None

# session_id → list of {"query": str, "answer": str}
conversation_history: dict[str, list[dict]] = defaultdict(list)

# ── Persistence ───────────────────────────────────────────────────────────────
def _save_store() -> None:
    PERSIST_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(FAISS_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "metadata": metadata}, f)
    log.info("Store saved — %d chunks.", len(chunks))

def _load_store() -> None:
    global index, chunks, metadata
    if FAISS_PATH.exists() and META_PATH.exists():
        index = faiss.read_index(str(FAISS_PATH))
        with open(META_PATH, "rb") as f:
            data = pickle.load(f)
        chunks   = data.get("chunks", [])
        metadata = data.get("metadata", [])
        log.info("Loaded store — %d chunks.", len(chunks))
    else:
        index    = faiss.IndexFlatL2(DIMENSION)
        chunks   = []
        metadata = []
        log.info("Fresh store created.")

# ── Magic bytes ───────────────────────────────────────────────────────────────
def _validate_magic_bytes(content: bytes, ext: str) -> bool:
    checks = MAGIC_BYTES.get(ext, [])
    if not checks:
        return True
    return any(content[o: o + len(m)] == m for o, m in checks)

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder
    log.info("Loading embedding model…")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    log.info("Embedding model loaded.")
    _load_store()
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

@app.get("/health")
async def health():
    return {"status": "ok", "chunks_indexed": len(chunks)}

# ── Upload ────────────────────────────────────────────────────────────────────
@app.post("/upload_doc")
async def upload_doc(request: Request, file: UploadFile = File(...)):
    global index, chunks, metadata

    client_ip = request.client.host
    await _check_rate_limit(client_ip, "upload")

    filename = file.filename or "unknown"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400,
            detail=f"Unsupported file type '.{ext}'. Allowed: {ALLOWED_EXTS}")

    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413,
            detail=f"File too large. Max is {MAX_FILE_BYTES//1024//1024} MB.")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if not _validate_magic_bytes(content, ext):
        raise HTTPException(status_code=400,
            detail=f"File content does not match '.{ext}'. Upload rejected.")

    log.info("[UPLOAD] %s (%.1f KB) from %s", filename, len(content)/1024, client_ip)

    text = ""
    try:
        if ext == "pdf":
            doc = fitz.open(stream=content, filetype="pdf")
            for page in doc: text += page.get_text()
            doc.close()
        elif ext == "docx":
            doc = Document(io.BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        elif ext == "txt":
            text = content.decode("utf-8", errors="ignore")
    except Exception as exc:
        log.exception("[INGEST] Extraction failed for %s", filename)
        raise HTTPException(status_code=422, detail=f"Could not parse file: {exc}")

    if not text.strip():
        return {"status": "warning", "chunks_added": 0,
                "note": "No text found. Image-based PDFs are not supported yet."}

    chunk_size, overlap = 800, 150
    new_chunks = [text[i: i+chunk_size].strip()
                  for i in range(0, len(text), chunk_size - overlap)
                  if len(text[i: i+chunk_size].strip()) > 60]

    if not new_chunks:
        return {"status": "warning", "chunks_added": 0, "note": "Document too short."}

    try:
        embeddings_np = np.array(
            embedder.encode(new_chunks, show_progress_bar=False), dtype="float32")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding error: {exc}")

    async with store_lock:
        start_idx = len(chunks)
        index.add(embeddings_np)
        for j, ct in enumerate(new_chunks):
            chunks.append(ct)
            metadata.append({"doc": filename, "chunk_idx": start_idx + j})
        _save_store()

    return {"status": "success", "chunks_added": len(new_chunks),
            "total_chunks": len(chunks), "filename": filename}

# ── WebSocket Q&A ─────────────────────────────────────────────────────────────
@app.websocket("/ask")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Phase 4: read session_id from query param so history survives page refresh
    # Frontend sends: ws://host/ask?session_id=abc123
    # Falls back to a new UUID if not provided
    params     = websocket.query_params
    session_id = params.get("session_id") or str(uuid.uuid4())[:8]
    client_ip  = websocket.client.host
    log.info("[WS:%s] Connected from %s.", session_id, client_ip)

    final_query = ""

    try:
        while True:
            data     = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            stripped = data.strip()
            if stripped == "__DONE__":
                break
            if stripped:
                final_query += stripped + " "
                await _safe_send(websocket, {"type": "partial", "text": f"Received: {stripped}"})
    except asyncio.TimeoutError:
        await _safe_send(websocket, {"type": "error", "message": "Timed out. Please try again."})
        return
    except WebSocketDisconnect:
        return

    final_query = final_query.strip()
    if not final_query:
        await _safe_send(websocket, {"type": "error", "message": "No speech detected."})
        return

    try:
        await _check_rate_limit(client_ip, "ask")
    except HTTPException:
        await _safe_send(websocket, {"type": "error", "message": "Too many questions. Slow down."})
        return

    # Query rewriting
    try:
        clean_query = rewrite_query(final_query)
        log.info("[WS:%s] Rewritten: '%s'", session_id, clean_query)
    except Exception:
        clean_query = final_query

    # Store snapshot
    async with store_lock:
        current_total  = index.ntotal
        store_snapshot = {"index": index, "chunks": list(chunks), "metadatas": list(metadata)}

    if current_total == 0:
        await _safe_send(websocket, {"type": "final", "query": final_query,
            "answer": "No document uploaded yet.", "passages": [], "low_confidence": False})
        return

    # Retrieval
    try:
        retrieved = retrieve_chunks(clean_query, vector_store=store_snapshot,
                                    embedder=embedder, top_k=5)
    except Exception as exc:
        await _safe_send(websocket, {"type": "error", "message": f"Retrieval error: {exc}"})
        return

    # ── Confidence threshold ──────────────────────────────────────────────────
    # If the best passage is below threshold, don't even call the LLM
    best_score = retrieved[0][1] if retrieved else 0.0
    if best_score < CONFIDENCE_THRESHOLD:
        log.info("[WS:%s] Low confidence (%.2f) — refusing to answer.", session_id, best_score)
        await _safe_send(websocket, {
            "type":           "final",
            "query":          final_query,
            "answer":         "I couldn't find relevant information about that in the uploaded document. Try rephrasing or ask about something covered in the document.",
            "passages":       [],
            "low_confidence": True,
        })
        return

    history = conversation_history[session_id]

    # ── Streaming generation ──────────────────────────────────────────────────
    # Send tokens one by one as they arrive from Groq
    full_answer = ""
    try:
        async for token in generate_answer_stream(clean_query, retrieved, history=history):
            full_answer += token
            await _safe_send(websocket, {"type": "stream", "token": token})
    except Exception as exc:
        log.exception("[WS:%s] Generation failed.", session_id)
        await _safe_send(websocket, {"type": "error",
            "message": f"AI model unavailable: {exc}"})
        return

    # Save turn to memory
    conversation_history[session_id].append({"query": final_query, "answer": full_answer})
    if len(conversation_history[session_id]) > MAX_HISTORY_TURNS:
        conversation_history[session_id] = conversation_history[session_id][-MAX_HISTORY_TURNS:]

    # Send final metadata (passages, scores) after streaming completes
    await _safe_send(websocket, {
        "type":           "final",
        "query":          final_query,
        "answer":         full_answer,
        "low_confidence": False,
        "passages": [
            {"text": chunk, "score": float(round(score, 3)),
             "source": f"{meta['doc']} (chunk {meta['chunk_idx']})"}
            for chunk, score, meta in retrieved
        ],
    })
    log.info("[WS:%s] Streaming complete (%d chars).", session_id, len(full_answer))

# ── Utility ───────────────────────────────────────────────────────────────────
async def _safe_send(websocket: WebSocket, payload: dict) -> None:
    try:
        await websocket.send_text(json.dumps(payload))
    except Exception as exc:
        log.warning("_safe_send failed: %s", exc)
