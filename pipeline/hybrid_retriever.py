"""
Hybrid Retriever — combines BM25 (keyword) + FAISS (dense semantic)
for better retrieval accuracy on long company reports.

Why hybrid?
- BM25 is great for exact keyword matches (e.g. "revenue", "Section 4.2")
- FAISS is great for semantic similarity (same meaning, different words)
- Together they catch what either alone would miss
"""

import numpy as np
from rank_bm25 import BM25Okapi
from pipeline.embedder import get_model


class HybridRetriever:
    def __init__(self, chunks: list, embeddings: np.ndarray, faiss_index):
        """
        Args:
            chunks: list of labeled chunk dicts {id, label, text, ...}
            embeddings: numpy array of dense embeddings (n_chunks, dim)
            faiss_index: built FAISS index
        """
        self.chunks = chunks
        self.embeddings = embeddings
        self.faiss_index = faiss_index

        # Build BM25 index with stopword-filtered tokens (see embedder.py)
        from pipeline.embedder import tokenise_for_bm25
        tokenised = [tokenise_for_bm25(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenised)

    def retrieve(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        k: int = 5,
        exclude_id: int = None,
        alpha: float = 0.5
    ) -> list:
        """
        Retrieve top-K chunks using hybrid scoring.

        Score = alpha * dense_score + (1 - alpha) * bm25_score
        alpha=0.5 balances both equally. Increase for more semantic bias.

        Args:
            query_embedding: dense vector for the query chunk
            query_text: raw text of the query (for BM25)
            k: number of results to return
            exclude_id: chunk id to exclude (self)
            alpha: weight for dense retrieval (0=pure BM25, 1=pure FAISS)

        Returns:
            list of top-K chunk dicts
        """
        n = len(self.chunks)

        # ── Dense retrieval scores (FAISS) ────────────────────────────────────
        query = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.faiss_index.search(query, n)

        # Convert L2 distances to scores (lower distance = higher score)
        dense_scores = np.zeros(n)
        for rank, idx in enumerate(indices[0]):
            if 0 <= idx < n:
                # Normalise: give rank-based score
                dense_scores[idx] = (n - rank) / n

        # ── BM25 scores ───────────────────────────────────────────────────────
        from pipeline.embedder import tokenise_for_bm25
        query_tokens = tokenise_for_bm25(query_text)
        bm25_raw = self.bm25.get_scores(query_tokens)

        # Normalise BM25 to [0, 1]
        bm25_max = bm25_raw.max()
        if bm25_max > 0:
            bm25_scores = bm25_raw / bm25_max
        else:
            bm25_scores = bm25_raw

        # ── Combined score ────────────────────────────────────────────────────
        combined = alpha * dense_scores + (1 - alpha) * bm25_scores

        # Rank by combined score
        ranked_indices = np.argsort(combined)[::-1]

        results = []
        for idx in ranked_indices:
            if idx < 0 or idx >= n:
                continue
            chunk = self.chunks[idx]
            if chunk["id"] == exclude_id:
                continue
            results.append(chunk)
            if len(results) == k:
                break

        return results

    def search(self, query_text: str, k: int = 5) -> list:
        """
        Search by free-text query (for Q&A chat).
        Encodes query text on the fly.

        Args:
            query_text: user's natural language question
            k: number of chunks to retrieve

        Returns:
            list of top-K most relevant chunk dicts
        """
        model = get_model()
        query_embedding = model.encode([query_text], convert_to_numpy=True)[0]
        from pipeline.embedder import tokenise_for_bm25
        return self.retrieve(
            query_embedding=query_embedding,
            query_text=query_text,
            k=k,
            exclude_id=None,
            alpha=0.5
        )

    def _bm25_tokens(self, text: str) -> list:
        from pipeline.embedder import tokenise_for_bm25
        return tokenise_for_bm25(text)