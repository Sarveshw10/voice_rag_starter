# main.py - Browser STT version: receives transcribed TEXT from index.html
import os
import uuid
import json
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from docx import Document
import io
import faiss
import numpy as np

# Import your modules
from rag.retrieval import retrieve_chunks
from rag.generation import generate_answer

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
print("Embedding model loaded!")

# Global vector store
index = faiss.IndexFlatL2(dimension)
chunks = []         # list of chunk texts
metadata = []       # list of dicts: {"doc": filename, "chunk_idx": int}

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
        return {"error": "Unsupported file type"}
    
    print(f"[INGEST] Extracted {len(text)} characters")
    if len(text.strip()) == 0:
        print("[INGEST] No text found — likely image-based PDF")
        return {"status": "warning", "chunks_added": 0, "note": "Image-based PDF — text not extracted"}
    
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
            metadata.append({
                "doc": filename,
                "chunk_idx": start_idx + j
            })
    
    print(f"[INGEST] Success! Total chunks in DB: {len(chunks)}\n")
    return {"status": "success", "chunks_added": len(new_chunks)}

@app.websocket("/ask")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    print(f"\n[WEBSOCKET] Session {session_id} started")

    final_query = ""

    try:
        while True:
            data = await websocket.receive_text()
            stripped = data.strip()
            
            if stripped == "__DONE__":  # Stop signal from client
                print("[WEBSOCKET] Received stop signal - processing query")
                break
            
            if stripped:
                final_query += stripped + " "
                print(f"[STT] Received partial/final text: '{stripped}'")
                await websocket.send_text(json.dumps({
                    "type": "partial",
                    "text": f"Received: {stripped}"
                }))

    except WebSocketDisconnect:
        print("[WEBSOCKET] Client disconnected unexpectedly")
        return  # Exit early if client disconnects

    # Process the query
    final_query = final_query.strip()
    if not final_query:
        final_query = "No query spoken. Please speak clearly."

    print(f"[STT] Final query to process: '{final_query}'")

    if index.ntotal == 0:
        answer = "No document uploaded yet. Please upload a file first."
        retrieved = []
    else:
        print(f"[RETRIEVAL] Searching for: '{final_query}'")
        retrieved = retrieve_chunks(
            final_query,
            vector_store={
                "index": index,
                "chunks": chunks,
                "metadatas": metadata
            },
            embedder=embedder,
            top_k=5
        )
        
        print("[RETRIEVAL] Top results:")
        for chunk, score, meta in retrieved:
            print(f"  - Score: {score:.3f} | Doc: {meta['doc']} | Chunk {meta['chunk_idx']}")

    print("[GENERATION] Calling local Ollama...")
    answer = generate_answer(final_query, retrieved)
    print("[GENERATION] Done!")

    print("[WEBSOCKET] Sending final response...")
    try:
        await websocket.send_text(json.dumps({
            "type": "final",
            "query": final_query,
            "answer": answer,
            "passages": [
                {
                    "text": t,
                    "score": float(round(s, 3)),
                    "source": f"{meta['doc']} (chunk {meta['chunk_idx']})"
                }
                for t, s, meta in retrieved
            ]
        }))
        print("[WEBSOCKET] Final response sent!\n")
    except Exception as e:
        print(f"[WEBSOCKET] Send failed: {e}")