# this is main file with open ai api integration
import os
import uuid
import json
import asyncio
import io
import faiss
import numpy as np
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from docx import Document
import nltk

load_dotenv()
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
print("Embedding model loaded!")

# Global in-memory vector store
index = faiss.IndexFlatL2(dimension)
chunks = []
metadata = []

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

@app.post("/upload_doc")
async def upload_doc(file: UploadFile = File(...)):
    global index, chunks, metadata
    
    content = await file.read()
    filename = file.filename
    ext = filename.lower().split('.')[-1]

    print(f"\n[UPLOAD] File: {filename} ({len(content)/1024:.1f} KB)")

    text = ""
    if ext == "pdf":
        print("[INGEST] Extracting text from PDF...")
        doc = fitz.open(stream=content, filetype="pdf")
        for page in doc:
            text += page.get_text()
    elif ext == "docx":
        print("[INGEST] Extracting from DOCX...")
        doc = Document(io.BytesIO(content))
        text = "\n".join([p.text for p in doc.paragraphs if p.text])
    elif ext == "txt":
        print("[INGEST] Reading TXT...")
        text = content.decode("utf-8", errors="ignore")
    else:
        return {"error": "Unsupported"}

    print(f"[INGEST] Extracted {len(text)} characters")

    if len(text.strip()) == 0:
        print("[INGEST] No text found — likely image-based PDF")
        return {"status": "warning", "chunks_added": 0, "note": "Image-based PDF — text not extracted"}

    # Chunking
    chunk_size = 1000
    overlap = 200
    new_chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk.strip()) > 50:
            new_chunks.append(chunk)

    print(f"[INGEST] Created {len(new_chunks)} chunks")

    if new_chunks:
        print("[INGEST] Generating embeddings...")
        embeddings = embedder.encode(new_chunks)
        embeddings_np = np.array(embeddings).astype('float32')
        index.add(embeddings_np)

        start_idx = len(chunks)
        for j, chunk_text in enumerate(new_chunks):
            chunks.append(chunk_text)
            metadata.append((filename, start_idx + j))

    print(f"[INGEST] Success! Total chunks in DB: {len(chunks)}\n")
    return {"status": "success", "chunks_added": len(new_chunks)}

@app.websocket("/ask")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    print(f"\n[WEBSOCKET] Session started: {session_id}")

    audio_buffer = b""
    partial_counter = 0

    try:
        while True:
            data = await websocket.receive_bytes()

            if len(data) == 0:
                print("[WEBSOCKET] Empty blob received → client stopped speaking")
                break

            audio_buffer += data
            partial_counter += 1

            if partial_counter % 4 == 0:
                print(f"[STREAM] Received {len(audio_buffer)/1024:.1f} KB audio")

            await websocket.send_text(json.dumps({
                "type": "partial",
                "text": f"Hearing you{'.' * (partial_counter % 8)}"
            }))

    except WebSocketDisconnect:
        print("[WEBSOCKET] Client disconnected abruptly")
    except Exception as e:
        print(f"[WEBSOCKET] Error: {e}")
    finally:
        print("[PROCESSING] User finished speaking → starting RAG")

        # Simple intent detection from audio (mock STT)
        sample = audio_buffer.decode(errors="ignore").lower()
        if any(word in sample for word in ["summarize", "summary", "overview"]):
            query = "Summarize the document"
        elif any(word in sample for word in ["what", "about", "content"]):
            query = "What is this document about?"
        else:
            query = "Tell me about the uploaded document"

        print(f"[STT] Detected query: {query}")

        if index.ntotal == 0:
            answer = "No document uploaded yet. Please upload a PDF, DOCX, or TXT file first."
            retrieved = []
        else:
            print(f"[RETRIEVAL] Searching among {index.ntotal} chunks...")
            q_emb = embedder.encode([query])
            q_np = np.array(q_emb).astype('float32')
            D, I = index.search(q_np, k=5)

            retrieved = []
            for dist, idx in zip(D[0], I[0]):
                if idx < len(chunks):
                    score = float(1 / (1 + dist))
                    text = chunks[idx]
                    source = metadata[idx][0]
                    retrieved.append((text, score, source))

            print("[GENERATION] Calling LLM...")
            from rag.generation import generate_answer
            answer = generate_answer(query, retrieved)
            print("[GENERATION] Done!")

        print("[WEBSOCKET] Sending final response...")
        try:
            await websocket.send_text(json.dumps({
                "type": "final",
                "query": query,
                "answer": answer,
                "passages": [
                    {"text": t, "score": round(s, 3), "source": src}
                    for t, s, src in retrieved
                ]
            }))
            print("[WEBSOCKET] Final response sent!\n")
        except:
            print("[WEBSOCKET] Client disconnected before final send\n")
