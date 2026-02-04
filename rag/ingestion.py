import fitz  # PyMuPDF
import io
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import nltk
nltk.download('punkt')

model = SentenceTransformer('all-MiniLM-L6-v2')

def ingest_document(content: bytes, file_type: str):
    text = ""
    if file_type == 'pdf':
        doc = fitz.open(stream=content, filetype="pdf")
        for page in doc:
            text += page.get_text()
    elif file_type == 'docx':
        doc = Document(io.BytesIO(content))
        text = "\n".join([p.text for p in doc.paragraphs])
    elif file_type == 'txt':
        text = content.decode('utf-8')
    else:
        raise ValueError("Unsupported file type")
    
    # Chunking: ~1000 tokens with 20% overlap
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    overlap = ""
    for sent in sentences:
        if len(current_chunk.split()) > 800:
            chunks.append(overlap + current_chunk)
            overlap = " ".join(current_chunk.split()[-int(0.2 * len(current_chunk.split())):])
            current_chunk = sent
        else:
            current_chunk += " " + sent
    if current_chunk:
        chunks.append(overlap + current_chunk)
    
    # Embeddings
    embeddings = model.encode(chunks)
    return list(zip(chunks, embeddings))

def get_vector_store():
    # FAISS index
    dimension = 384  # For all-MiniLM-L6-v2
    index = faiss.IndexFlatL2(dimension)
    return {"index": index, "chunks": [], "metadatas": []}  # Add methods to add/search

# Extend with add/search methods in your impl