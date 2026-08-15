"""
AutoDrive RAG v2.0 — Document Chunker
Implements 5 document chunking strategies for breaking large documents
into smaller, semantically coherent chunks for effective retrieval.

Strategies:
  1. FIXED       — Fixed-size with configurable overlap
  2. SENTENCE    — Sentence-aware splitting (preserves sentence boundaries)
  3. PARAGRAPH   — Paragraph-based splitting (preserves natural breaks)
  4. SEMANTIC    — Groups sentences by embedding similarity
  5. RECURSIVE   — Hierarchical: paragraph → sentence → character fallback

Each chunk includes metadata: source_doc_id, chunk_index, strategy, char_range.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("chatbot.chunking")


class ChunkingStrategy(str, Enum):
    """Available document chunking strategies."""

    FIXED = "fixed"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"


@dataclass
class Chunk:
    """Represents a single chunk of a document."""

    text: str
    metadata: dict = field(default_factory=dict)
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    parent_doc_id: str = ""
    strategy: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "parent_doc_id": self.parent_doc_id,
            "strategy": self.strategy,
        }


class DocumentChunker:
    """
    Multi-strategy document chunker.

    Splits documents into smaller chunks while preserving semantic
    coherence. Supports 5 strategies with configurable parameters.

    Attributes:
        strategy: Which chunking strategy to use.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks in characters.
        min_chunk_size: Minimum chunk size (discard smaller chunks).
    """

    def __init__(
        self,
        strategy: str | ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 50,
    ) -> None:
        self.strategy = ChunkingStrategy(strategy)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(
        self,
        text: str,
        doc_id: str = "",
        metadata: Optional[dict] = None,
    ) -> list[Chunk]:
        """
        Split a document into chunks using the configured strategy.

        Args:
            text: The full document text.
            doc_id: Identifier for the source document.
            metadata: Additional metadata to attach to each chunk.

        Returns:
            List of Chunk objects.
        """
        if not text or not text.strip():
            return []

        base_metadata = metadata or {}

        if self.strategy == ChunkingStrategy.FIXED:
            chunks = self._fixed_chunking(text)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            chunks = self._sentence_chunking(text)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            chunks = self._paragraph_chunking(text)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            chunks = self._semantic_chunking(text)
        elif self.strategy == ChunkingStrategy.RECURSIVE:
            chunks = self._recursive_chunking(text)
        else:
            chunks = self._recursive_chunking(text)

        # Enrich with metadata
        result = []
        for i, (chunk_text, start, end) in enumerate(chunks):
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue
            result.append(
                Chunk(
                    text=chunk_text.strip(),
                    metadata={**base_metadata},
                    chunk_index=i,
                    start_char=start,
                    end_char=end,
                    parent_doc_id=doc_id,
                    strategy=self.strategy.value,
                )
            )

        logger.debug(
            "Chunked document '%s': %d chunks (strategy=%s)",
            doc_id[:30],
            len(result),
            self.strategy.value,
        )
        return result

    def chunk_documents(
        self,
        documents: list[dict],
    ) -> list[Chunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of dicts with 'text', 'doc_id', and optional 'metadata'.

        Returns:
            Flat list of all chunks from all documents.
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(
                text=doc["text"],
                doc_id=doc.get("doc_id", ""),
                metadata=doc.get("metadata", {}),
            )
            all_chunks.extend(chunks)

        logger.info(
            "Chunked %d documents → %d total chunks", len(documents), len(all_chunks)
        )
        return all_chunks

    # ── Strategy 1: Fixed-size Chunking ─────────────────────────────

    def _fixed_chunking(self, text: str) -> list[tuple[str, int, int]]:
        """
        Split text into fixed-size chunks with overlap.

        Simple and predictable. Works well when document structure
        is not important (e.g., plain text files).
        """
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunks.append((text[start:end], start, end))

            # Move forward by (chunk_size - overlap)
            step = self.chunk_size - self.chunk_overlap
            if step <= 0:
                step = self.chunk_size  # Safety: prevent infinite loop
            start += step

        return chunks

    # ── Strategy 2: Sentence-aware Chunking ─────────────────────────

    def _sentence_chunking(self, text: str) -> list[tuple[str, int, int]]:
        """
        Split text on sentence boundaries, grouping sentences into
        chunks that don't exceed chunk_size.

        Preserves sentence integrity — no sentence is split mid-way.
        Uses NLTK sent_tokenize with regex fallback.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return [(text, 0, len(text))]

        chunks = []
        current_sentences: list[str] = []
        current_len = 0
        chunk_start = 0
        pos = 0

        for sentence in sentences:
            sent_len = len(sentence)

            if current_len + sent_len > self.chunk_size and current_sentences:
                # Emit current chunk
                chunk_text = " ".join(current_sentences)
                chunks.append((chunk_text, chunk_start, pos))

                # Overlap: keep last few sentences
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    if overlap_len + len(s) > self.chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s) + 1

                current_sentences = overlap_sentences
                current_len = overlap_len
                chunk_start = pos - overlap_len if overlap_len else pos

            current_sentences.append(sentence)
            current_len += sent_len + 1
            pos += sent_len + 1

        # Emit final chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append((chunk_text, chunk_start, len(text)))

        return chunks

    # ── Strategy 3: Paragraph-based Chunking ────────────────────────

    def _paragraph_chunking(self, text: str) -> list[tuple[str, int, int]]:
        """
        Split text on paragraph boundaries (double newlines).

        Preserves natural document structure. Paragraphs exceeding
        chunk_size are further split using sentence chunking.
        """
        paragraphs = re.split(r"\n\s*\n", text)
        if not paragraphs:
            return [(text, 0, len(text))]

        chunks = []
        current_paras: list[str] = []
        current_len = 0
        pos = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)

            # If single paragraph exceeds chunk_size, split it by sentences
            if para_len > self.chunk_size:
                # First emit accumulated paragraphs
                if current_paras:
                    chunk_text = "\n\n".join(current_paras)
                    chunks.append((chunk_text, pos - current_len, pos))
                    current_paras = []
                    current_len = 0

                # Split the large paragraph by sentences
                sub_chunks = self._sentence_chunking(para)
                for sub_text, sub_start, sub_end in sub_chunks:
                    chunks.append((sub_text, pos + sub_start, pos + sub_end))
                pos += para_len + 2
                continue

            if current_len + para_len > self.chunk_size and current_paras:
                chunk_text = "\n\n".join(current_paras)
                chunks.append((chunk_text, pos - current_len, pos))
                current_paras = []
                current_len = 0

            current_paras.append(para)
            current_len += para_len + 2
            pos += para_len + 2

        if current_paras:
            chunk_text = "\n\n".join(current_paras)
            chunks.append((chunk_text, pos - current_len, pos))

        return chunks

    # ── Strategy 4: Semantic Chunking ───────────────────────────────

    def _semantic_chunking(self, text: str) -> list[tuple[str, int, int]]:
        """
        Group sentences by semantic similarity using embeddings.

        Sentences are embedded, and consecutive sentences with high
        cosine similarity are grouped together. A new chunk starts
        when similarity drops below a threshold.

        Requires sentence-transformers (lazy import).
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [(text, 0, len(text))]

        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(sentences, normalize_embeddings=True)

            # Compute cosine similarity between consecutive sentences
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = np.dot(embeddings[i], embeddings[i + 1])
                similarities.append(sim)

            # Find split points where similarity drops below threshold
            # Use percentile-based threshold for adaptivity
            threshold = np.percentile(similarities, 30)

            chunks = []
            current_group: list[str] = [sentences[0]]
            current_len = len(sentences[0])
            pos = 0

            for i, sentence in enumerate(sentences[1:], start=1):
                # Split if similarity drops OR chunk exceeds max size
                should_split = (
                    similarities[i - 1] < threshold
                    or current_len + len(sentence) > self.chunk_size
                )

                if should_split and current_group:
                    chunk_text = " ".join(current_group)
                    chunks.append((chunk_text, pos, pos + len(chunk_text)))
                    pos += len(chunk_text) + 1
                    current_group = []
                    current_len = 0

                current_group.append(sentence)
                current_len += len(sentence) + 1

            if current_group:
                chunk_text = " ".join(current_group)
                chunks.append((chunk_text, pos, pos + len(chunk_text)))

            return chunks

        except ImportError:
            logger.warning(
                "sentence-transformers not available — "
                "falling back to sentence chunking for semantic strategy"
            )
            return self._sentence_chunking(text)

    # ── Strategy 5: Recursive Chunking ──────────────────────────────

    def _recursive_chunking(self, text: str) -> list[tuple[str, int, int]]:
        """
        Recursively split text using a hierarchy of separators:
        paragraph → sentence → character.

        Tries the largest separator first. If the resulting chunk is
        still too large, recursively splits with the next separator.
        This is the most robust strategy for mixed-format documents.
        """
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
        return self._recursive_split(text, separators, 0)

    def _recursive_split(
        self,
        text: str,
        separators: list[str],
        offset: int,
    ) -> list[tuple[str, int, int]]:
        """Recursively split text using the separator hierarchy."""
        if len(text) <= self.chunk_size:
            return [(text, offset, offset + len(text))]

        if not separators:
            # Base case: force-split by chunk_size
            return self._fixed_chunking(text)

        sep = separators[0]
        parts = text.split(sep)

        if len(parts) == 1:
            # Separator not found — try next one
            return self._recursive_split(text, separators[1:], offset)

        chunks = []
        current_parts: list[str] = []
        current_len = 0
        pos = offset

        for part in parts:
            part_len = len(part) + len(sep)

            if current_len + part_len > self.chunk_size and current_parts:
                chunk_text = sep.join(current_parts)
                if len(chunk_text) > self.chunk_size:
                    # Still too large — recurse with next separator
                    sub_chunks = self._recursive_split(
                        chunk_text, separators[1:], pos
                    )
                    chunks.extend(sub_chunks)
                else:
                    chunks.append((chunk_text, pos, pos + len(chunk_text)))
                pos += len(chunk_text) + len(sep)
                current_parts = []
                current_len = 0

            current_parts.append(part)
            current_len += part_len

        if current_parts:
            chunk_text = sep.join(current_parts)
            if len(chunk_text) > self.chunk_size:
                sub_chunks = self._recursive_split(
                    chunk_text, separators[1:], pos
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append((chunk_text, pos, pos + len(chunk_text)))

        return chunks

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using NLTK or regex fallback."""
        try:
            from nltk.tokenize import sent_tokenize

            return sent_tokenize(text)
        except (ImportError, LookupError):
            # Regex fallback: split on .!? followed by space/newline
            sentences = re.split(r"(?<=[.!?])\s+", text)
            return [s.strip() for s in sentences if s.strip()]

    def __repr__(self) -> str:
        return (
            f"DocumentChunker(strategy={self.strategy.value}, "
            f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
