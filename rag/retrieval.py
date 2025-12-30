def retrieve_chunks(query: str, vector_store: dict, top_k=5):
    embedding = model.encode([query])[0]
    distances, indices = vector_store["index"].search(embedding.reshape(1, -1), top_k)
    retrieved = [(vector_store["chunks"][i], distances[0][j], "doc_id_here") for j, i in enumerate(indices[0])]
    return retrieved