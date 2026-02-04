# retrieval.py - Fixed: pass embedder from main.py, use real metadata
from typing import List, Tuple, Dict

def retrieve_chunks(
    query: str,
    vector_store: Dict,
    embedder,  # Required: the SentenceTransformer instance from main.py
    top_k: int = 5
) -> List[Tuple[str, float, Dict]]:
    """
    Retrieve top-k relevant chunks using the same embedding model.
    
    Args:
        query: The user's transcribed question
        vector_store: dict with 'index' (FAISS), 'chunks' (list), 'metadatas' (list of dicts)
        embedder: SentenceTransformer model (must match the one used for indexing)
        top_k: Number of results to return
    
    Returns:
        List of (chunk_text, similarity_score, metadata_dict)
    """
    if embedder is None:
        raise ValueError("embedder must be provided (SentenceTransformer object)")

    # Embed query using the EXACT same model as documents
    query_embedding = embedder.encode([query])[0]

    # Search in FAISS (L2 distance)
    distances, indices = vector_store["index"].search(
        query_embedding.reshape(1, -1),
        top_k
    )

    retrieved = []
    for j, idx in enumerate(indices[0]):
        if idx == -1:  # Invalid FAISS result
            continue

        chunk_text = vector_store["chunks"][idx]
        distance = distances[0][j]

        # Approximate cosine similarity (valid for normalized embeddings)
        similarity = 1 - (distance ** 2) / 2
        similarity = max(0.0, min(1.0, similarity))

        # Real metadata (doc name + chunk index)
        meta = vector_store.get("metadatas", [{}])[idx]
        doc_name = meta.get("doc", "unknown")
        chunk_idx = meta.get("chunk_idx", idx)

        retrieved.append((chunk_text, similarity, {"doc": doc_name, "chunk_idx": chunk_idx}))

    return retrieved