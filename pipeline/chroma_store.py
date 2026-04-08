"""
Persistent vector store using ChromaDB.

Why ChromaDB instead of Neo4j or Redis?
- Fully local and free (no server needed)
- Persistent on disk — survives restarts
- Built-in embedding + metadata storage
- Lightweight (no RAM overhead like Redis)
- Simple Python API

Data is stored in ./chroma_db/ folder automatically.
"""

import os
import hashlib
import chromadb
from chromadb.config import Settings


CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db")


def get_collection(doc_name: str):
    """
    Get or create a ChromaDB collection for a document.
    Collection name is based on a hash of the document name
    so different docs don't collide.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Sanitise collection name (ChromaDB requires alphanumeric + underscores)
    safe_name = "doc_" + hashlib.md5(doc_name.encode()).hexdigest()[:12]

    collection = client.get_or_create_collection(
        name=safe_name,
        metadata={"doc_name": doc_name, "hnsw:space": "cosine"}
    )
    return collection


def store_chunks(doc_name: str, chunks: list, embeddings) -> None:
    """
    Store chunks and their embeddings persistently in ChromaDB.

    Args:
        doc_name: name of the source document
        chunks: list of chunk dicts {id, label, text, ...}
        embeddings: numpy array of shape (n_chunks, dim)
    """
    collection = get_collection(doc_name)

    # Clear existing entries for this doc (in case of re-upload)
    existing = collection.count()
    if existing > 0:
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)

    ids = [str(c["id"]) for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "label": c["label"],
            "display": c.get("display", c["label"]),
            "heading": c.get("heading", ""),
            "doc_name": doc_name
        }
        for c in chunks
    ]
    embeddings_list = embeddings.tolist()

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings_list,
        metadatas=metadatas
    )


def load_chunks(doc_name: str) -> list | None:
    """
    Load previously stored chunks for a document from ChromaDB.
    Returns list of chunk dicts, or None if not found.
    """
    try:
        collection = get_collection(doc_name)
        if collection.count() == 0:
            return None

        result = collection.get(include=["documents", "metadatas", "embeddings"])
        chunks = []
        for i, doc_id in enumerate(result["ids"]):
            meta = result["metadatas"][i]
            chunks.append({
                "id": int(doc_id),
                "label": meta["label"],
                "display": meta.get("display", meta["label"]),
                "heading": meta.get("heading", ""),
                "text": result["documents"][i]
            })
        # Sort by id to restore original order
        chunks.sort(key=lambda x: x["id"])
        return chunks

    except Exception:
        return None


def list_stored_documents() -> list:
    """Return names of all documents stored in ChromaDB."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collections = client.list_collections()
        doc_names = []
        for col in collections:
            meta = col.metadata or {}
            if "doc_name" in meta:
                doc_names.append(meta["doc_name"])
        return doc_names
    except Exception:
        return []


def delete_document(doc_name: str) -> bool:
    """Delete a stored document from ChromaDB."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        safe_name = "doc_" + hashlib.md5(doc_name.encode()).hexdigest()[:12]
        client.delete_collection(safe_name)
        return True
    except Exception:
        return False