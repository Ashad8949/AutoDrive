"""
AutoDrive RAG v2.0 — Hybrid Retriever
Combines Dense (semantic) and Sparse (BM25) retrieval results using
configurable fusion strategies.

Supported fusion strategies:
  1. Reciprocal Rank Fusion (RRF) — rank-based, parameter-free
  2. Weighted Score Fusion — normalized score interpolation
  3. Distribution-Based Score Fusion (DBSF) — z-score normalization

RRF is the default and recommended strategy for most use cases.
See: Cormack, Clarke & Buettcher (2009) — "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods"
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Optional

import numpy as np

from .dense_retriever import DenseRetriever
from .sparse_retriever import SparseRetriever

logger = logging.getLogger("chatbot.retrieval.hybrid")


class FusionStrategy(str, Enum):
    """Available fusion strategies for combining retrieval results."""

    RRF = "rrf"                    # Reciprocal Rank Fusion
    WEIGHTED = "weighted"          # Weighted score interpolation
    DBSF = "dbsf"                  # Distribution-Based Score Fusion


class HybridRetriever:
    """
    Hybrid retriever combining dense and sparse retrieval with fusion.

    This implements the industry-standard approach of combining semantic
    (vector) search with keyword (BM25) search, fusing results using
    strategies like Reciprocal Rank Fusion (RRF).

    Attributes:
        dense: The dense (semantic) retriever instance.
        sparse: The sparse (BM25) retriever instance.
        fusion_strategy: Which fusion method to use.
        alpha: Weight for dense scores in weighted fusion (0=sparse only, 1=dense only).
        rrf_k: Smoothing constant for RRF (default: 60, from original paper).
    """

    def __init__(
        self,
        dense: Optional[DenseRetriever] = None,
        sparse: Optional[SparseRetriever] = None,
        fusion_strategy: Optional[str] = None,
        alpha: Optional[float] = None,
        rrf_k: int = 60,
    ) -> None:
        """
        Initialize the hybrid retriever.

        Args:
            dense: Pre-built DenseRetriever instance. Created if None.
            sparse: Pre-built SparseRetriever instance. Created if None.
            fusion_strategy: One of 'rrf', 'weighted', 'dbsf'. Default from env or 'rrf'.
            alpha: Dense weight for weighted fusion. Default from env or 0.7.
            rrf_k: RRF smoothing constant. Higher = more weight to lower-ranked docs.
        """
        self.dense = dense or DenseRetriever()
        self.sparse = sparse or SparseRetriever()
        self.fusion_strategy = FusionStrategy(
            fusion_strategy or os.getenv("FUSION_STRATEGY", "rrf")
        )
        self.alpha = alpha if alpha is not None else float(
            os.getenv("FUSION_ALPHA", "0.7")
        )
        self.rrf_k = rrf_k

    def build_index(
        self,
        documents: list[str],
        metadata: Optional[list[dict]] = None,
        use_ivf: bool = False,
    ) -> None:
        """
        Build both dense and sparse indices from the same documents.

        Args:
            documents: List of document text strings to index.
            metadata: Optional list of metadata dicts (one per document).
            use_ivf: If True, use IVF for the FAISS dense index.
        """
        logger.info(
            "Building hybrid index: %d documents, strategy=%s",
            len(documents),
            self.fusion_strategy.value,
        )
        self.dense.build_index(documents, metadata, use_ivf=use_ivf)
        self.sparse.build_index(documents, metadata)
        logger.info("Hybrid index built ✓")

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Perform hybrid search by combining dense and sparse results.

        The search pipeline:
          1. Retrieve top-N from dense retriever
          2. Retrieve top-N from sparse retriever
          3. Fuse results using the configured strategy
          4. Return top-K fused results

        Args:
            query: The search query string.
            top_k: Number of final fused results to return.
            dense_top_k: Override top-k for dense retrieval (default: top_k * 3).
            sparse_top_k: Override top-k for sparse retrieval (default: top_k * 3).

        Returns:
            List of dicts with 'text', 'score', 'rank', 'metadata', 'source'.
        """
        # Over-retrieve from both to ensure good fusion coverage
        retrieve_k = top_k * 3
        d_k = dense_top_k or retrieve_k
        s_k = sparse_top_k or retrieve_k

        dense_results = self.dense.search(query, top_k=d_k)
        sparse_results = self.sparse.search(query, top_k=s_k)

        # Fuse results
        if self.fusion_strategy == FusionStrategy.RRF:
            fused = self._reciprocal_rank_fusion(dense_results, sparse_results)
        elif self.fusion_strategy == FusionStrategy.WEIGHTED:
            fused = self._weighted_score_fusion(dense_results, sparse_results)
        elif self.fusion_strategy == FusionStrategy.DBSF:
            fused = self._distribution_based_fusion(dense_results, sparse_results)
        else:
            fused = self._reciprocal_rank_fusion(dense_results, sparse_results)

        # Re-rank and return top-k
        fused.sort(key=lambda x: x["score"], reverse=True)
        for i, result in enumerate(fused[:top_k], start=1):
            result["rank"] = i

        return fused[:top_k]

    def search_dense_only(self, query: str, top_k: int = 5) -> list[dict]:
        """Search using only the dense retriever (for A/B testing)."""
        return self.dense.search(query, top_k=top_k)

    def search_sparse_only(self, query: str, top_k: int = 5) -> list[dict]:
        """Search using only the sparse retriever (for A/B testing)."""
        return self.sparse.search(query, top_k=top_k)

    # ── Fusion Strategies ───────────────────────────────────────────

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion (RRF).

        Combines results by their rank positions, independent of score scales.
        Formula: RRF_score(d) = Σ 1 / (k + rank_i(d))
        where k is a smoothing constant (default 60).

        This is the most robust fusion strategy as it doesn't require
        score normalization and works well across different score distributions.
        """
        doc_scores: dict[str, dict] = {}

        for result in dense_results:
            doc_key = result["text"]
            if doc_key not in doc_scores:
                doc_scores[doc_key] = {
                    "text": result["text"],
                    "score": 0.0,
                    "metadata": result["metadata"],
                    "source": "hybrid",
                    "dense_rank": result["rank"],
                    "sparse_rank": None,
                }
            doc_scores[doc_key]["score"] += 1.0 / (self.rrf_k + result["rank"])

        for result in sparse_results:
            doc_key = result["text"]
            if doc_key not in doc_scores:
                doc_scores[doc_key] = {
                    "text": result["text"],
                    "score": 0.0,
                    "metadata": result["metadata"],
                    "source": "hybrid",
                    "dense_rank": None,
                    "sparse_rank": result["rank"],
                }
            else:
                doc_scores[doc_key]["sparse_rank"] = result["rank"]
            doc_scores[doc_key]["score"] += 1.0 / (self.rrf_k + result["rank"])

        return list(doc_scores.values())

    def _weighted_score_fusion(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
    ) -> list[dict]:
        """
        Weighted Score Fusion.

        Normalizes scores from both retrievers to [0, 1] using min-max
        normalization, then combines with a weighted average:
          fused_score = α * dense_score_norm + (1 - α) * sparse_score_norm

        Args use self.alpha as the dense weight.
        """
        dense_norm = self._min_max_normalize(dense_results)
        sparse_norm = self._min_max_normalize(sparse_results)

        doc_scores: dict[str, dict] = {}

        for result in dense_norm:
            doc_key = result["text"]
            doc_scores[doc_key] = {
                "text": result["text"],
                "score": self.alpha * result["score"],
                "metadata": result["metadata"],
                "source": "hybrid",
            }

        for result in sparse_norm:
            doc_key = result["text"]
            if doc_key in doc_scores:
                doc_scores[doc_key]["score"] += (1 - self.alpha) * result["score"]
            else:
                doc_scores[doc_key] = {
                    "text": result["text"],
                    "score": (1 - self.alpha) * result["score"],
                    "metadata": result["metadata"],
                    "source": "hybrid",
                }

        return list(doc_scores.values())

    def _distribution_based_fusion(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
    ) -> list[dict]:
        """
        Distribution-Based Score Fusion (DBSF).

        Normalizes scores using z-score normalization (mean=0, std=1),
        then shifts to positive range and combines. More statistically
        robust than min-max normalization for skewed distributions.
        """
        dense_norm = self._z_score_normalize(dense_results)
        sparse_norm = self._z_score_normalize(sparse_results)

        doc_scores: dict[str, dict] = {}

        for result in dense_norm:
            doc_key = result["text"]
            doc_scores[doc_key] = {
                "text": result["text"],
                "score": self.alpha * result["score"],
                "metadata": result["metadata"],
                "source": "hybrid",
            }

        for result in sparse_norm:
            doc_key = result["text"]
            if doc_key in doc_scores:
                doc_scores[doc_key]["score"] += (1 - self.alpha) * result["score"]
            else:
                doc_scores[doc_key] = {
                    "text": result["text"],
                    "score": (1 - self.alpha) * result["score"],
                    "metadata": result["metadata"],
                    "source": "hybrid",
                }

        return list(doc_scores.values())

    # ── Normalization Helpers ───────────────────────────────────────

    @staticmethod
    def _min_max_normalize(results: list[dict]) -> list[dict]:
        """Normalize scores to [0, 1] using min-max scaling."""
        if not results:
            return results

        scores = [r["score"] for r in results]
        min_s, max_s = min(scores), max(scores)
        range_s = max_s - min_s

        normalized = []
        for r in results:
            new_score = (r["score"] - min_s) / range_s if range_s > 0 else 0.5
            normalized.append({**r, "score": new_score})

        return normalized

    @staticmethod
    def _z_score_normalize(results: list[dict]) -> list[dict]:
        """Normalize scores using z-score (mean=0, std=1), shifted to positive."""
        if not results:
            return results

        scores = np.array([r["score"] for r in results])
        mean = scores.mean()
        std = scores.std()

        normalized = []
        for r in results:
            z = (r["score"] - mean) / std if std > 0 else 0.0
            # Shift to positive range using sigmoid-like transform
            new_score = 1 / (1 + np.exp(-z))
            normalized.append({**r, "score": float(new_score)})

        return normalized

    # ── Persistence ─────────────────────────────────────────────────

    def save_index(self, path: str) -> None:
        """Save both dense and sparse indices to disk."""
        self.dense.save_index(f"{path}/dense")
        self.sparse.save_index(f"{path}/sparse")
        logger.info("Hybrid index saved to %s", path)

    def load_index(self, path: str) -> None:
        """Load both dense and sparse indices from disk."""
        self.dense.load_index(f"{path}/dense")
        self.sparse.load_index(f"{path}/sparse")
        logger.info("Hybrid index loaded from %s", path)

    def __len__(self) -> int:
        return len(self.dense)

    def __repr__(self) -> str:
        return (
            f"HybridRetriever(strategy={self.fusion_strategy.value}, "
            f"alpha={self.alpha}, dense={len(self.dense)}, "
            f"sparse={len(self.sparse)})"
        )
