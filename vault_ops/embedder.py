from sentence_transformers import SentenceTransformer
import faiss, numpy as np, os

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return MODEL.encode(texts, normalize_embeddings=True)

def load_faiss(d: int | None = None):
    if os.path.exists("./vector_store/faiss_index.bin"):
        return faiss.read_index("./vector_store/faiss_index.bin")
    if d is None:
        raise ValueError("Need dimension to create new index.")
    return faiss.IndexFlatIP(d)

def save_faiss(index, path):
    faiss.write_index(index, str(path))
