"""
AutoDrive RAG v2.0 — Late Chunker
Implements the "Late Chunking" technique where the full document is
embedded first (preserving global context), then token embeddings
are split into chunks and mean-pooled.

Traditional chunking → embed each chunk independently (loses context).
Late chunking → embed full doc → split embeddings → pool per chunk.

This ensures every chunk's embedding retains awareness of the full
document's semantics, dramatically improving retrieval for context-
dependent passages.

Reference: Günther et al. (2024) — "Late Chunking: Contextual Chunk
Embeddings Using Long-Context Embedding Models"
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("chatbot.chunking.late")


class LateChunker:
    """
    Late chunking: embed the full document first, then split and pool.

    This approach leverages the transformer's attention mechanism to
    create context-aware chunk embeddings. Each chunk's embedding
    reflects not just its own content but its role within the document.

    Attributes:
        model_name: Name of the sentence-transformers model to use.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Overlap between chunks in tokens.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 128,
        chunk_overlap: int = 32,
    ) -> None:
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        """Lazy-load the sentence-transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading model for late chunking: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            self._tokenizer = self._model.tokenizer
            logger.info("Late chunking model loaded ✓")
        return self._model

    @property
    def tokenizer(self):
        """Get the tokenizer from the loaded model."""
        _ = self.model  # Ensure model is loaded
        return self._tokenizer

    def chunk_and_embed(
        self,
        text: str,
        doc_id: str = "",
        metadata: Optional[dict] = None,
    ) -> list[dict]:
        """
        Perform late chunking: embed full document, then split embeddings.

        Steps:
          1. Tokenize the full document
          2. Pass through the transformer to get token-level embeddings
          3. Split token embeddings into chunks
          4. Mean-pool each chunk's token embeddings to get chunk embeddings

        Args:
            text: Full document text.
            doc_id: Document identifier.
            metadata: Additional metadata for each chunk.

        Returns:
            List of dicts with 'text', 'embedding', 'metadata' for each chunk.
        """
        import torch

        base_metadata = metadata or {}

        # Check if document fits within model's max length
        max_length = self.tokenizer.model_max_length
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            padding=False,
        )
        input_ids = tokens["input_ids"][0]
        total_tokens = len(input_ids)

        if total_tokens > max_length:
            logger.warning(
                "Document '%s' has %d tokens (max=%d) — "
                "falling back to standard chunking + embedding",
                doc_id[:30],
                total_tokens,
                max_length,
            )
            return self._fallback_chunking(text, doc_id, base_metadata)

        # Step 1: Get token-level embeddings from transformer
        with torch.no_grad():
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            model_output = self.model[0].auto_model(**encoded)
            token_embeddings = model_output.last_hidden_state[0]  # (seq_len, dim)

        # Remove special tokens ([CLS], [SEP])
        # Typically first and last tokens
        token_embeddings = token_embeddings[1:-1]
        token_ids = input_ids[1:-1]
        actual_tokens = len(token_embeddings)

        # Step 2: Split into chunks
        chunks = []
        start = 0
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = self.chunk_size

        chunk_idx = 0
        while start < actual_tokens:
            end = min(start + self.chunk_size, actual_tokens)

            # Mean-pool the token embeddings for this chunk
            chunk_emb = token_embeddings[start:end].mean(dim=0)
            chunk_emb = chunk_emb / chunk_emb.norm()  # L2 normalize

            # Decode chunk text from token IDs
            chunk_text = self.tokenizer.decode(
                token_ids[start:end],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            chunks.append({
                "text": chunk_text,
                "embedding": chunk_emb.cpu().numpy().astype(np.float32),
                "metadata": {
                    **base_metadata,
                    "parent_doc_id": doc_id,
                    "chunk_index": chunk_idx,
                    "strategy": "late_chunking",
                    "token_start": start,
                    "token_end": end,
                },
            })

            start += step
            chunk_idx += 1

        logger.debug(
            "Late-chunked document '%s': %d chunks from %d tokens",
            doc_id[:30],
            len(chunks),
            actual_tokens,
        )
        return chunks

    def _fallback_chunking(
        self,
        text: str,
        doc_id: str,
        metadata: dict,
    ) -> list[dict]:
        """
        Fallback for documents exceeding model's max length.
        Uses standard sentence-level chunking + independent embedding.
        """
        from .chunker import DocumentChunker

        chunker = DocumentChunker(
            strategy="sentence",
            chunk_size=500,
            chunk_overlap=100,
        )
        chunks = chunker.chunk_document(text, doc_id=doc_id, metadata=metadata)

        results = []
        for chunk in chunks:
            embedding = self.model.encode(
                chunk.text,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)

            results.append({
                "text": chunk.text,
                "embedding": embedding,
                "metadata": {
                    **chunk.metadata,
                    "strategy": "late_chunking_fallback",
                },
            })

        return results

    def chunk_and_embed_batch(
        self,
        documents: list[dict],
    ) -> list[dict]:
        """
        Late-chunk and embed multiple documents.

        Args:
            documents: List of dicts with 'text', 'doc_id', optional 'metadata'.

        Returns:
            Flat list of all chunk dicts from all documents.
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_and_embed(
                text=doc["text"],
                doc_id=doc.get("doc_id", ""),
                metadata=doc.get("metadata", {}),
            )
            all_chunks.extend(chunks)

        logger.info(
            "Late-chunked %d documents → %d total chunks",
            len(documents),
            len(all_chunks),
        )
        return all_chunks

    def __repr__(self) -> str:
        return (
            f"LateChunker(model={self.model_name!r}, "
            f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
