import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_chunks(chunks: list) -> np.ndarray:
    """
    Generate sentence embeddings for all chunks.
    Returns numpy array of shape (n_chunks, embedding_dim).
    """
    model = get_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build a FAISS flat L2 index."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def retrieve_top_k(
    query_embedding: np.ndarray,
    index: faiss.IndexFlatL2,
    chunks: list,
    k: int = 5,
    exclude_id: int = None
) -> list:
    """
    Retrieve top-K most similar chunks, excluding self.
    Returns list of chunk dicts.
    """
    query = query_embedding.reshape(1, -1).astype(np.float32)
    distances, indices = index.search(query, k + 1)

    results = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(chunks):
            continue
        if chunks[idx]["id"] == exclude_id:
            continue
        results.append(chunks[idx])
        if len(results) == k:
            break

    return results