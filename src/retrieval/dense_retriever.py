"""
AutoDrive RAG v2.0 — Dense Retriever
Uses sentence-transformers for embedding and FAISS for vector similarity search.

This module implements the dense retrieval component of the hybrid retrieval
pipeline. It embeds documents and queries into a shared vector space and
retrieves the most semantically similar documents using cosine similarity
via a FAISS index.

Supported models (configurable via DENSE_MODEL env var):
  - all-MiniLM-L6-v2 (default, fast, 384d)
  - BAAI/bge-small-en-v1.5 (better quality, 384d)
  - BAAI/bge-base-en-v1.5 (best quality, 768d)
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("chatbot.retrieval.dense")

# Default embedding model — small, fast, free
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class DenseRetriever:
    """
    Dense vector retriever using sentence-transformers + FAISS.

    Encodes documents and queries into dense embeddings, builds a FAISS
    index for efficient similarity search, and retrieves top-k results.

    Attributes:
        model_name: Name of the sentence-transformers model to use.
        index: The FAISS index storing document embeddings.
        documents: List of raw document strings (parallel to index vectors).
        metadata: List of metadata dicts (parallel to index vectors).
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        Initialize the dense retriever.

        Args:
            model_name: HuggingFace model name for sentence-transformers.
                        Defaults to DENSE_MODEL env var or all-MiniLM-L6-v2.
        """
        self.model_name = model_name or os.getenv("DENSE_MODEL", DEFAULT_MODEL)
        self._model = None  # Lazy-loaded
        self.index = None
        self.documents: list[str] = []
        self.metadata: list[dict] = []
        self._dimension: int = 0

    @property
    def model(self):
        """Lazy-load the sentence-transformers model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading dense embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(
                "Dense model loaded ✓ (dimension=%d)", self._dimension
            )
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of document texts into dense embeddings.

        Args:
            texts: List of document strings to embed.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=len(texts) > 100,
            batch_size=64,
            normalize_embeddings=True,  # For cosine similarity via inner product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encode a single query into a dense embedding.

        Args:
            query: The search query string.

        Returns:
            numpy array of shape (1, embedding_dim).
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32)

    def build_index(
        self,
        documents: list[str],
        metadata: Optional[list[dict]] = None,
        use_ivf: bool = False,
        nlist: int = 100,
    ) -> None:
        """
        Build a FAISS index from a list of documents.

        Args:
            documents: List of document text strings to index.
            metadata: Optional list of metadata dicts (one per document).
            use_ivf: If True, use IVF index for faster search on large datasets.
                     Recommended when len(documents) > 10,000.
            nlist: Number of IVF clusters (only used when use_ivf=True).
        """
        import faiss

        if not documents:
            logger.warning("No documents to index — skipping build_index")
            return

        self.documents = documents
        self.metadata = metadata or [{} for _ in documents]

        logger.info("Embedding %d documents for FAISS index...", len(documents))
        embeddings = self.embed_documents(documents)
        self._dimension = embeddings.shape[1]

        if use_ivf and len(documents) > nlist:
            # IVF index: faster search for large datasets (>10k docs)
            quantizer = faiss.IndexFlatIP(self._dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, self._dimension, nlist, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = min(10, nlist)  # Search 10 clusters
            logger.info(
                "Built IVF index: %d vectors, %d clusters", len(documents), nlist
            )
        else:
            # Flat index: exact search, best for < 10k docs
            self.index = faiss.IndexFlatIP(self._dimension)
            self.index.add(embeddings)
            logger.info("Built Flat index: %d vectors", len(documents))

    def search(
        self, query: str, top_k: int = 5
    ) -> list[dict]:
        """
        Search the FAISS index for documents most similar to the query.

        Args:
            query: The search query string.
            top_k: Number of top results to return.

        Returns:
            List of dicts, each containing:
              - 'text': The document text
              - 'score': Similarity score (higher is better)
              - 'rank': 1-indexed rank
              - 'metadata': Associated metadata dict
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Dense index is empty — returning no results")
            return []

        query_embedding = self.embed_query(query)
        k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_embedding, k)

        results = []
        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]), start=1
        ):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            results.append({
                "text": self.documents[idx],
                "score": float(score),
                "rank": rank,
                "metadata": self.metadata[idx],
                "source": "dense",
            })

        return results

    def save_index(self, path: str) -> None:
        """
        Persist the FAISS index and document store to disk.

        Args:
            path: Directory path to save the index files.
        """
        import faiss

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, str(save_dir / "faiss.index"))

        with open(save_dir / "documents.pkl", "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "metadata": self.metadata,
                    "model_name": self.model_name,
                    "dimension": self._dimension,
                },
                f,
            )
        logger.info("Dense index saved to %s", save_dir)

    def load_index(self, path: str) -> None:
        """
        Load a persisted FAISS index and document store from disk.

        Args:
            path: Directory path containing the saved index files.
        """
        import faiss

        save_dir = Path(path)
        index_path = save_dir / "faiss.index"
        docs_path = save_dir / "documents.pkl"

        if not index_path.exists() or not docs_path.exists():
            raise FileNotFoundError(
                f"Index files not found in {save_dir}. Run build_index first."
            )

        self.index = faiss.read_index(str(index_path))

        with open(docs_path, "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
            self.model_name = data.get("model_name", DEFAULT_MODEL)
            self._dimension = data.get("dimension", 0)

        logger.info(
            "Dense index loaded: %d vectors from %s",
            self.index.ntotal,
            save_dir,
        )

    def __len__(self) -> int:
        """Return the number of indexed documents."""
        return self.index.ntotal if self.index else 0

    def __repr__(self) -> str:
        return (
            f"DenseRetriever(model={self.model_name!r}, "
            f"docs={len(self)}, dim={self._dimension})"
        )
