"""
AutoDrive RAG v2.0 — Context Compressor
Reduces retrieved documents to only the most relevant sentences
before sending to the LLM, saving tokens and improving quality.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("chatbot.memory.compressor")


class ContextCompressor:
    """
    Compresses retrieved documents by extracting only relevant sentences.

    Two modes:
      1. Embedding-based: keep sentences whose embeddings are close to the query
      2. LLM-based: ask the LLM to extract relevant sentences
    """

    EXTRACT_PROMPT = (
        "Given the question and document below, extract ONLY the sentences "
        "that are directly relevant to answering the question. Return them "
        "as a bulleted list. If nothing is relevant, say 'No relevant info'.\n\n"
        "Question: {query}\n\nDocument:\n{document}\n\nRelevant sentences:"
    )

    def __init__(self, llm=None, similarity_threshold: float = 0.3) -> None:
        self.llm = llm
        self.similarity_threshold = similarity_threshold

    def compress(
        self, query: str, documents: list[dict], method: str = "embedding"
    ) -> list[dict]:
        """
        Compress documents to only relevant content.

        Args:
            query: User's question.
            documents: Retrieved documents.
            method: 'embedding' or 'llm'.

        Returns:
            Compressed documents with only relevant sentences.
        """
        if method == "llm" and self.llm:
            return self._compress_llm(query, documents)
        return self._compress_embedding(query, documents)

    def _compress_embedding(self, query: str, documents: list[dict]) -> list[dict]:
        """Keep sentences whose embeddings are similar to the query."""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            model = SentenceTransformer("all-MiniLM-L6-v2")
            query_emb = model.encode([query], normalize_embeddings=True)[0]

            compressed = []
            for doc in documents:
                sentences = re.split(r"(?<=[.!?])\s+", doc["text"])
                if not sentences:
                    continue

                sent_embs = model.encode(sentences, normalize_embeddings=True)
                similarities = np.dot(sent_embs, query_emb)

                relevant = [
                    s for s, sim in zip(sentences, similarities)
                    if sim >= self.similarity_threshold
                ]

                if relevant:
                    compressed.append({
                        **doc,
                        "text": " ".join(relevant),
                        "original_text": doc["text"],
                        "compression_ratio": len(" ".join(relevant)) / max(len(doc["text"]), 1),
                    })

            logger.info("Compressed %d → %d documents", len(documents), len(compressed))
            return compressed if compressed else documents

        except ImportError:
            logger.warning("sentence-transformers not available — skipping compression")
            return documents

    def _compress_llm(self, query: str, documents: list[dict]) -> list[dict]:
        """Use LLM to extract relevant sentences."""
        compressed = []
        for doc in documents:
            try:
                prompt = self.EXTRACT_PROMPT.format(
                    query=query, document=doc["text"][:2000]
                )
                resp = self.llm.invoke(prompt)
                text = resp.content if hasattr(resp, "content") else str(resp)

                if "no relevant" not in text.lower():
                    compressed.append({
                        **doc,
                        "text": text.strip(),
                        "original_text": doc["text"],
                    })
            except Exception as e:
                logger.warning("LLM compression failed: %s", e)
                compressed.append(doc)

        return compressed if compressed else documents
