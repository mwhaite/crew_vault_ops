from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss, os

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    """Embeds a list of texts using the SentenceTransformer model."""
    return MODEL.encode(texts, normalize_embeddings=True)

def load_faiss(path: str | os.PathLike, d: int | None = None):
    """Loads a FAISS index from disk, or creates a new one if it doesn't exist."""

    index_path = Path(path)
    if index_path.exists():
        index = faiss.read_index(str(index_path))
        if d is not None and index.d != d:
            raise ValueError(
                f"FAISS index dimension mismatch: expected {d}, found {index.d}"
            )
        return index

    if d is None:
        raise ValueError("Need dimension to create new index when no index file is present.")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    return faiss.IndexFlatIP(d)

def save_faiss(index, path):
    """Saves a FAISS index to disk."""
    faiss.write_index(index, str(path))
