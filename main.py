# main.py - Phase 1: Persistent index, concurrency lock, full error handling
import os
import uuid
import json
import asyncio
import logging
import pickle
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from docx import Document
import io
import faiss
import numpy as np

from rag.retrieval import retrieve_chunks
from rag.generation import generate_answer

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DIMENSION       = 384
PERSIST_DIR     = Path("data")          # all index files live here
FAISS_PATH      = PERSIST_DIR / "index.faiss"
META_PATH       = PERSIST_DIR / "metadata.pkl"
MAX_FILE_BYTES  = 20 * 1024 * 1024     # 20 MB hard limit
ALLOWED_EXTS    = {"pdf", "docx", "txt"}

# ── Global state ──────────────────────────────────────────────────────────────
# Protected by `store_lock` — never mutate index/chunks/metadata without acquiring it.
store_lock = asyncio.Lock()
index:    faiss.IndexFlatL2 = None
chunks:   list[str]         = []
metadata: list[dict]        = []
embedder: SentenceTransformer = None


# ── Persistence helpers ───────────────────────────────────────────────────────
def _save_store() -> None:
    """Write FAISS index + metadata to disk. Call inside store_lock."""
    PERSIST_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(FAISS_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "metadata": metadata}, f)
    log.info("Store saved — %d chunks on disk.", len(chunks))


def _load_store() -> None:
    """Load persisted index from disk, or create a fresh one."""
    global index, chunks, metadata
    if FAISS_PATH.exists() and META_PATH.exists():
        index = faiss.read_index(str(FAISS_PATH))
        with open(META_PATH, "rb") as f:
            data = pickle.load(f)
        chunks   = data.get("chunks", [])
        metadata = data.get("metadata", [])
        log.info("Loaded existing store — %d chunks.", len(chunks))
    else:
        index    = faiss.IndexFlatL2(DIMENSION)
        chunks   = []
        metadata = []
        log.info("No existing store found — starting fresh.")


# ── App lifecycle ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder
    log.info("Loading embedding model…")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    log.info("Embedding model loaded.")
    _load_store()
    yield
    # (shutdown hook — nothing to clean up for now)


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "chunks_indexed": len(chunks)}


# ── Upload endpoint ───────────────────────────────────────────────────────────
@app.post("/upload_doc")
async def upload_doc(file: UploadFile = File(...)):
    global index, chunks, metadata

    # ── Validate filename / extension ─────────────────────────────────────────
    filename = file.filename or "unknown"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Allowed: {ALLOWED_EXTS}"
        )

    # ── Read with size guard ──────────────────────────────────────────────────
    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content)//1024} KB). Max is {MAX_FILE_BYTES//1024//1024} MB."
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    log.info("[UPLOAD] %s  (%.1f KB)", filename, len(content) / 1024)

    # ── Text extraction ───────────────────────────────────────────────────────
    text = ""
    try:
        if ext == "pdf":
            doc = fitz.open(stream=content, filetype="pdf")
            for page in doc:
                text += page.get_text()
            doc.close()

        elif ext == "docx":
            doc = Document(io.BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

        elif ext == "txt":
            text = content.decode("utf-8", errors="ignore")

    except Exception as exc:
        log.exception("[INGEST] Text extraction failed for %s", filename)
        raise HTTPException(
            status_code=422,
            detail=f"Could not parse file: {exc}"
        )

    if not text.strip():
        log.warning("[INGEST] No text extracted from %s — possibly image-based PDF.", filename)
        return {
            "status": "warning",
            "chunks_added": 0,
            "note": "No text found. Image-based PDFs are not supported yet."
        }

    log.info("[INGEST] Extracted %d chars from %s.", len(text), filename)

    # ── Chunking (sentence-aware, 800 chars, 150 overlap) ────────────────────
    chunk_size = 800
    overlap    = 150
    new_chunks: list[str] = []

    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i : i + chunk_size].strip()
        if len(chunk) > 60:           # drop tiny trailing fragments
            new_chunks.append(chunk)

    if not new_chunks:
        return {"status": "warning", "chunks_added": 0, "note": "Document too short to index."}

    log.info("[INGEST] Created %d chunks.", len(new_chunks))

    # ── Embed + index (inside lock) ───────────────────────────────────────────
    try:
        embeddings = embedder.encode(new_chunks, show_progress_bar=False)
        embeddings_np = np.array(embeddings, dtype="float32")
    except Exception as exc:
        log.exception("[EMBED] Embedding failed.")
        raise HTTPException(status_code=500, detail=f"Embedding error: {exc}")

    async with store_lock:
        start_idx = len(chunks)
        index.add(embeddings_np)
        for j, chunk_text in enumerate(new_chunks):
            chunks.append(chunk_text)
            metadata.append({"doc": filename, "chunk_idx": start_idx + j})
        _save_store()   # persist immediately after every successful upload

    log.info("[INGEST] Done. Total chunks in store: %d", len(chunks))
    return {
        "status": "success",
        "chunks_added": len(new_chunks),
        "total_chunks": len(chunks),
        "filename": filename,
    }


# ── WebSocket Q&A endpoint ────────────────────────────────────────────────────
@app.websocket("/ask")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    log.info("[WS:%s] Connected.", session_id)

    final_query = ""

    # ── Receive transcript chunks until __DONE__ ──────────────────────────────
    try:
        while True:
            data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            stripped = data.strip()

            if stripped == "__DONE__":
                log.info("[WS:%s] Stop signal received.", session_id)
                break

            if stripped:
                final_query += stripped + " "
                log.info("[WS:%s] Partial: '%s'", session_id, stripped)
                try:
                    await websocket.send_text(json.dumps({
                        "type": "partial",
                        "text": f"Received: {stripped}"
                    }))
                except Exception:
                    log.warning("[WS:%s] Could not send partial ack.", session_id)

    except asyncio.TimeoutError:
        log.warning("[WS:%s] Timed out waiting for transcript.", session_id)
        await _safe_send(websocket, {"type": "error", "message": "Timed out. Please try again."})
        return

    except WebSocketDisconnect:
        log.info("[WS:%s] Client disconnected during transcript.", session_id)
        return

    # ── Process query ─────────────────────────────────────────────────────────
    final_query = final_query.strip()
    if not final_query:
        await _safe_send(websocket, {"type": "error", "message": "No speech detected. Please try again."})
        return

    log.info("[WS:%s] Query: '%s'", session_id, final_query)

    # Snapshot store under lock so we don't read while another upload is writing
    async with store_lock:
        current_total = index.ntotal
        store_snapshot = {
            "index":     index,
            "chunks":    list(chunks),
            "metadatas": list(metadata),
        }

    if current_total == 0:
        await _safe_send(websocket, {
            "type":    "final",
            "query":   final_query,
            "answer":  "No document uploaded yet. Please upload a file first.",
            "passages": [],
        })
        return

    # ── Retrieval ─────────────────────────────────────────────────────────────
    try:
        retrieved = retrieve_chunks(
            final_query,
            vector_store=store_snapshot,
            embedder=embedder,
            top_k=5,
        )
        log.info("[WS:%s] Retrieved %d passages.", session_id, len(retrieved))
    except Exception as exc:
        log.exception("[WS:%s] Retrieval failed.", session_id)
        await _safe_send(websocket, {"type": "error", "message": f"Retrieval error: {exc}"})
        return

    # ── Generation ────────────────────────────────────────────────────────────
    try:
        answer = generate_answer(final_query, retrieved)
        log.info("[WS:%s] Answer generated (%d chars).", session_id, len(answer))
    except Exception as exc:
        log.exception("[WS:%s] Generation failed.", session_id)
        await _safe_send(websocket, {
            "type":    "error",
            "message": f"The AI model is unavailable right now. Error: {exc}",
        })
        return

    # ── Send final response ───────────────────────────────────────────────────
    await _safe_send(websocket, {
        "type":   "final",
        "query":  final_query,
        "answer": answer,
        "passages": [
            {
                "text":   chunk,
                "score":  float(round(score, 3)),
                "source": f"{meta['doc']} (chunk {meta['chunk_idx']})",
            }
            for chunk, score, meta in retrieved
        ],
    })
    log.info("[WS:%s] Response sent.", session_id)


# ── Utility ───────────────────────────────────────────────────────────────────
async def _safe_send(websocket: WebSocket, payload: dict) -> None:
    """Send JSON without raising if the client already disconnected."""
    try:
        await websocket.send_text(json.dumps(payload))
    except Exception as exc:
        log.warning("_safe_send failed: %s", exc)
