"""
AutoDrive RAG v2.0 — Semantic Cache
Caches query-response pairs using embedding similarity so that
similar queries return cached responses instantly.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger("chatbot.cache")


class SemanticCache:
    """
    Embedding-based semantic cache for RAG responses.

    If an incoming query is semantically similar (above threshold)
    to a previously seen query, returns the cached response instantly
    without running the full RAG pipeline.

    TTL-based invalidation ensures stale responses are discarded
    when inventory changes.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 3600,
        max_entries: int = 1000,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries

        self._model = None
        self._entries: list[dict] = []  # {embedding, query, response, timestamp}
        self._embeddings: Optional[np.ndarray] = None
        self._hits = 0
        self._misses = 0

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def get(self, query: str) -> Optional[str]:
        """
        Check if a similar query exists in cache.

        Args:
            query: The incoming user query.

        Returns:
            Cached response string if found, None if cache miss.
        """
        if not self._entries:
            self._misses += 1
            return None

        self._evict_expired()

        query_emb = self.model.encode([query], normalize_embeddings=True)

        if self._embeddings is None or len(self._embeddings) == 0:
            self._misses += 1
            return None

        similarities = np.dot(self._embeddings, query_emb.T).flatten()
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= self.similarity_threshold:
            self._hits += 1
            entry = self._entries[best_idx]
            logger.info(
                "Cache HIT (score=%.3f): '%s' ≈ '%s'",
                best_score, query[:40], entry["query"][:40],
            )
            return entry["response"]

        self._misses += 1
        return None

    def put(self, query: str, response: str) -> None:
        """Store a query-response pair in cache."""
        if len(self._entries) >= self.max_entries:
            self._evict_oldest()

        embedding = self.model.encode([query], normalize_embeddings=True)[0]

        self._entries.append({
            "query": query,
            "response": response,
            "timestamp": time.time(),
            "embedding": embedding,
        })

        # Rebuild embedding matrix
        self._embeddings = np.array([e["embedding"] for e in self._entries])
        logger.debug("Cache PUT: '%s' (total=%d)", query[:40], len(self._entries))

    def invalidate_all(self) -> None:
        """Clear the entire cache (e.g., after inventory refresh)."""
        self._entries.clear()
        self._embeddings = None
        logger.info("Cache invalidated (all entries cleared)")

    def _evict_expired(self) -> None:
        now = time.time()
        self._entries = [
            e for e in self._entries
            if (now - e["timestamp"]) < self.ttl_seconds
        ]
        if self._entries:
            self._embeddings = np.array([e["embedding"] for e in self._entries])
        else:
            self._embeddings = None

    def _evict_oldest(self) -> None:
        if self._entries:
            self._entries.pop(0)

    def get_stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "entries": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }
