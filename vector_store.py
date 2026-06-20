"""
vector_store.py
----------------
Handles:
  1. Embedding chunks using a local, open-source model (sentence-transformers).
     No API calls, no cost, runs fully on your machine (CPU is fine).
  2. Storing embeddings in ChromaDB in "persistent" mode — this just saves
     everything to a folder on disk (./chroma_db). No server, no Docker,
     nothing extra to run.
  3. Querying: given a question, find the most relevant chunks.

Why this embedding model?
"all-MiniLM-L6-v2" is the most popular beginner-friendly open-source
embedding model: small (~80MB), fast even on CPU, and good general-purpose
quality. It's the standard "default choice" in the open-source RAG world.
"""

from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict
from pathlib import Path

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = str(Path(__file__).parent.parent / "chroma_db")
COLLECTION_NAME = "pdf_documents"


class VectorStore:
    def __init__(self):
        print(f"🔧 Loading local embedding model '{EMBEDDING_MODEL_NAME}' (first run downloads it, ~80MB)...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        print(f"🔧 Connecting to local ChromaDB at {CHROMA_DB_PATH}")
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity = standard for text embeddings
        )

    def add_chunks(self, chunks: List[Dict], batch_size: int = 64):
        """Embed and store a list of chunks. Skips chunks already in the DB (by chunk_id)."""
        if not chunks:
            print("⚠️  No chunks to add.")
            return

        existing_ids = set(self.collection.get()["ids"]) if self.collection.count() > 0 else set()
        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

        if not new_chunks:
            print("✅ All chunks already indexed. Nothing new to add.")
            return

        print(f"🧮 Embedding {len(new_chunks)} new chunks locally (no API calls)...")

        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i: i + batch_size]
            texts = [c["content"] for c in batch]
            embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()

            self.collection.add(
                ids=[c["chunk_id"] for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {
                        "source_file": c["source_file"],
                        "page_number": c["page_number"],
                        "chunk_type": c["chunk_type"],
                    }
                    for c in batch
                ],
            )
            print(f"   -> indexed {min(i + batch_size, len(new_chunks))}/{len(new_chunks)}")

        print(f"✅ Done. Collection now has {self.collection.count()} total chunks.")

    def search(self, query: str, top_k: int = 6) -> List[Dict]:
        """
        Find the top_k most relevant chunks for a query.
        Returns a list of dicts with content + metadata + similarity score.
        """
        query_embedding = self.embedder.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "content": results["documents"][0][i],
                "source_file": results["metadatas"][0][i]["source_file"],
                "page_number": results["metadatas"][0][i]["page_number"],
                "chunk_type": results["metadatas"][0][i]["chunk_type"],
                # ChromaDB returns "distance" (lower = more similar) for cosine space.
                # We convert to a 0-1 "similarity" score, which is more intuitive.
                "similarity": 1 - results["distances"][0][i],
            })
        return hits

    def count(self) -> int:
        return self.collection.count()
