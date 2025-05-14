# embedder.py
from sentence_transformers import SentenceTransformer
from typing import List
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def create_faiss_index(chunks: List[str]):
    embeddings = model.encode(chunks)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings, chunks
