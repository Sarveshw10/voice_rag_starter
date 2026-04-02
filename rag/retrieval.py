# rag/retrieval.py
from typing import List, Tuple, Dict
import logging

log = logging.getLogger(__name__)


def retrieve_chunks(
    query: str,
    vector_store: Dict,
    embedder,
    top_k: int = 5,
) -> List[Tuple[str, float, Dict]]:
    """
    Retrieve top-k relevant chunks from the FAISS index.

    Args:
        query:        The user's question (already cleaned).
        vector_store: Dict with keys 'index' (faiss), 'chunks' (list[str]),
                      'metadatas' (list[dict]).
        embedder:     SentenceTransformer — must be the same model used at ingest time.
        top_k:        How many results to return.

    Returns:
        List of (chunk_text, similarity_score, metadata_dict), best first.
    """
    if embedder is None:
        raise ValueError("embedder must be provided.")

    if not query.strip():
        raise ValueError("Query is empty.")

    # Embed the query with the same model used at ingest time
    query_vec = embedder.encode([query])[0].reshape(1, -1)

    # FAISS L2 search
    distances, indices = vector_store["index"].search(query_vec, top_k)

    results: List[Tuple[str, float, Dict]] = []
    for j, idx in enumerate(indices[0]):
        if idx == -1:
            continue  # FAISS returns -1 when fewer results than top_k exist

        chunk_text = vector_store["chunks"][idx]
        distance   = float(distances[0][j])

        # Convert L2 distance → approximate cosine similarity
        # Valid when embeddings are normalised (SentenceTransformers normalises by default)
        similarity = max(0.0, min(1.0, 1.0 - (distance ** 2) / 2))

        meta       = vector_store["metadatas"][idx] if idx < len(vector_store["metadatas"]) else {}
        results.append((chunk_text, similarity, meta))

    log.debug("Retrieved %d passages for query '%s'.", len(results), query[:60])
    return results
