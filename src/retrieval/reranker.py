"""
AutoDrive RAG v2.0 — Cross-Encoder Re-Ranker
Second-stage re-ranking of retrieved documents using cross-encoder models.

Cross-encoders jointly encode (query, document) pairs, producing much
more accurate relevance scores than bi-encoder similarity. However,
they are O(n) per query (vs O(1) for bi-encoder), so they are used
only on the top-K results from first-stage retrieval.

Pipeline: Retrieve top-50 → Re-rank → Return top-5

Supported models:
  - cross-encoder/ms-marco-MiniLM-L-6-v2 (default, fast, good quality)
  - BAAI/bge-reranker-base (better quality, slower)
  - cross-encoder/ms-marco-TinyBERT-L-2-v2 (fastest, lower quality)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("chatbot.retrieval.reranker")

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Re-ranks retrieval results using a cross-encoder model.

    Takes (query, document) pairs from first-stage retrieval and
    produces fine-grained relevance scores for re-ordering.

    Attributes:
        model_name: Name of the cross-encoder model.
        top_n: Number of documents to return after re-ranking.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        top_n: int = 5,
    ) -> None:
        """
        Initialize the cross-encoder re-ranker.

        Args:
            model_name: HuggingFace cross-encoder model name.
            top_n: Number of top results to return after re-ranking.
        """
        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL", DEFAULT_RERANKER_MODEL
        )
        self.top_n = top_n
        self._model = None  # Lazy-loaded

    @property
    def model(self):
        """Lazy-load the cross-encoder model on first use."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder model: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder loaded ✓")
        return self._model

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_n: Optional[int] = None,
    ) -> list[dict]:
        """
        Re-rank retrieval results using the cross-encoder.

        Args:
            query: The original search query.
            results: List of retrieval result dicts (must have 'text' key).
            top_n: Override default top_n for this call.

        Returns:
            Re-ranked list of result dicts, sorted by cross-encoder score.
            Each result gets an additional 'rerank_score' field.
        """
        if not results:
            return results

        n = top_n or self.top_n

        # Create (query, document) pairs for scoring
        pairs = [(query, result["text"]) for result in results]

        # Score all pairs
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Attach scores and sort
        scored_results = []
        for result, score in zip(results, scores):
            scored_results.append({
                **result,
                "rerank_score": float(score),
                "original_score": result.get("score", 0.0),
            })

        # Sort by rerank score (descending)
        scored_results.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Re-assign ranks
        for i, result in enumerate(scored_results[:n], start=1):
            result["rank"] = i

        logger.debug(
            "Re-ranked %d → %d results for query: %s",
            len(results),
            min(n, len(scored_results)),
            query[:50],
        )

        return scored_results[:n]

    def score_pair(self, query: str, document: str) -> float:
        """
        Score a single (query, document) pair.

        Args:
            query: The search query.
            document: The document text.

        Returns:
            Relevance score (higher is more relevant).
        """
        score = self.model.predict([(query, document)], show_progress_bar=False)
        return float(score[0])

    def batch_score(
        self, query: str, documents: list[str]
    ) -> list[float]:
        """
        Score multiple documents against a single query.

        Args:
            query: The search query.
            documents: List of document texts.

        Returns:
            List of relevance scores (parallel to documents).
        """
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]

    def __repr__(self) -> str:
        return (
            f"CrossEncoderReranker(model={self.model_name!r}, top_n={self.top_n})"
        )
