"""
AutoDrive RAG v2.0 — Sparse Retriever (BM25)
Implements keyword-based retrieval using the BM25 (Okapi) algorithm.

BM25 excels at exact keyword matching and is complementary to dense
retrieval which captures semantic similarity. Together they form
the hybrid retrieval pipeline.

BM25 scoring: score(q, d) = Σ IDF(qi) · (tf(qi,d) · (k1+1)) / (tf(qi,d) + k1 · (1 - b + b · |d|/avgdl))
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("chatbot.retrieval.sparse")


def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer with lowercasing.
    Falls back to regex if NLTK is not available.

    Args:
        text: Input text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    try:
        from nltk.tokenize import word_tokenize

        return [t.lower() for t in word_tokenize(text) if t.isalnum()]
    except (ImportError, LookupError):
        # Fallback: regex-based tokenization
        return [t.lower() for t in re.findall(r"\b\w+\b", text)]


class SparseRetriever:
    """
    BM25-based sparse retriever for keyword-matching retrieval.

    Uses the rank_bm25 library for efficient BM25 scoring. Documents
    are tokenized and indexed, and queries are scored against all
    documents to find the best keyword matches.

    Attributes:
        bm25: The BM25Okapi index instance.
        documents: List of raw document strings.
        metadata: List of metadata dicts (parallel to documents).
        tokenized_corpus: List of tokenized document representations.
    """

    def __init__(self) -> None:
        self.bm25 = None
        self.documents: list[str] = []
        self.metadata: list[dict] = []
        self.tokenized_corpus: list[list[str]] = []

    def build_index(
        self,
        documents: list[str],
        metadata: Optional[list[dict]] = None,
    ) -> None:
        """
        Build the BM25 index from a list of documents.

        Args:
            documents: List of document text strings to index.
            metadata: Optional list of metadata dicts (one per document).
        """
        from rank_bm25 import BM25Okapi

        if not documents:
            logger.warning("No documents to index — skipping BM25 build")
            return

        self.documents = documents
        self.metadata = metadata or [{} for _ in documents]

        logger.info("Tokenizing %d documents for BM25 index...", len(documents))
        self.tokenized_corpus = [_tokenize(doc) for doc in documents]

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("BM25 index built ✓ (%d documents)", len(documents))

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search the BM25 index for documents matching the query keywords.

        Args:
            query: The search query string.
            top_k: Number of top results to return.

        Returns:
            List of dicts, each containing:
              - 'text': The document text
              - 'score': BM25 relevance score (higher is better)
              - 'rank': 1-indexed rank
              - 'metadata': Associated metadata dict
        """
        if self.bm25 is None:
            logger.warning("BM25 index not built — returning no results")
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score (descending)
        top_k = min(top_k, len(self.documents))
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            if scores[idx] <= 0:
                continue  # Skip zero-score documents
            results.append({
                "text": self.documents[idx],
                "score": float(scores[idx]),
                "rank": rank,
                "metadata": self.metadata[idx],
                "source": "sparse",
            })

        return results

    def save_index(self, path: str) -> None:
        """
        Persist the BM25 index and document store to disk.

        Args:
            path: Directory path to save the index files.
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "bm25_index.pkl", "wb") as f:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "documents": self.documents,
                    "metadata": self.metadata,
                    "tokenized_corpus": self.tokenized_corpus,
                },
                f,
            )
        logger.info("BM25 index saved to %s", save_dir)

    def load_index(self, path: str) -> None:
        """
        Load a persisted BM25 index from disk.

        Args:
            path: Directory path containing the saved index files.
        """
        save_dir = Path(path)
        index_path = save_dir / "bm25_index.pkl"

        if not index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {index_path}. Run build_index first."
            )

        with open(index_path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.documents = data["documents"]
            self.metadata = data["metadata"]
            self.tokenized_corpus = data["tokenized_corpus"]

        logger.info(
            "BM25 index loaded: %d documents from %s",
            len(self.documents),
            save_dir,
        )

    def __len__(self) -> int:
        """Return the number of indexed documents."""
        return len(self.documents)

    def __repr__(self) -> str:
        return f"SparseRetriever(docs={len(self)})"
