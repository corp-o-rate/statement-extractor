"""
DocumentChunker - Token-aware text chunking for document processing.

Splits documents into chunks suitable for the extraction pipeline while
maintaining page and sentence boundary awareness.
"""

import logging
import re
from typing import Optional

from ..models.document import ChunkingConfig, Document, TextChunk

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Chunks documents into processable text segments.

    Uses the T5-Gemma tokenizer for accurate token counting and supports:
    - Page boundary awareness
    - Sentence boundary splitting
    - Configurable overlap between chunks
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize the chunker.

        Args:
            config: Chunking configuration (uses defaults if not provided)
        """
        self._config = config or ChunkingConfig()
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy-load the tokenizer from the T5-Gemma model."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            logger.debug("Loading T5-Gemma tokenizer for chunking")
            self._tokenizer = AutoTokenizer.from_pretrained(
                "Corp-o-Rate-Community/statement-extractor",
                trust_remote_code=True,
            )
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def chunk_document(self, document: Document) -> list[TextChunk]:
        """
        Chunk a document into text segments.

        Args:
            document: Document to chunk

        Returns:
            List of TextChunk objects
        """
        if not document.full_text:
            return []

        logger.info(f"Chunking document {document.document_id}: {document.char_count} chars")

        # If document has pages and we respect page boundaries, use page-aware chunking
        if document.pages and self._config.respect_page_boundaries:
            chunks = self._chunk_with_pages(document)
        else:
            chunks = self._chunk_text(
                text=document.full_text,
                document_id=document.document_id,
                page_getter=document.get_pages_in_range if document.pages else None,
            )

        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def chunk_text(
        self,
        text: str,
        document_id: str,
    ) -> list[TextChunk]:
        """
        Chunk plain text (without page structure).

        Args:
            text: Text to chunk
            document_id: Document ID to assign to chunks

        Returns:
            List of TextChunk objects
        """
        return self._chunk_text(text, document_id, page_getter=None)

    def _chunk_with_pages(self, document: Document) -> list[TextChunk]:
        """Chunk document respecting page boundaries."""
        chunks = []
        chunk_index = 0
        current_text = ""
        current_start = 0
        current_pages = []

        for page in document.pages:
            page_tokens = self.count_tokens(page.text)

            # Check if adding this page would exceed max_tokens
            current_tokens = self.count_tokens(current_text)

            if current_text and current_tokens + page_tokens > self._config.max_tokens:
                # Flush current chunk
                chunk = self._create_chunk(
                    chunk_index=chunk_index,
                    text=current_text,
                    start_char=current_start,
                    pages=current_pages,
                    document_id=document.document_id,
                    overlap_chars=0,
                )
                chunks.append(chunk)
                chunk_index += 1

                # Start new chunk with overlap from previous
                overlap_text = self._get_overlap_text(current_text)
                current_text = overlap_text + page.text
                current_start = page.char_offset - len(overlap_text)
                current_pages = [page.page_number]
            else:
                # Add page to current chunk
                if current_text:
                    current_text += "\n" + page.text
                else:
                    current_text = page.text
                    current_start = page.char_offset
                current_pages.append(page.page_number)

            # If current chunk exceeds target, try to split at sentence boundary
            current_tokens = self.count_tokens(current_text)
            if current_tokens > self._config.target_tokens:
                # Split within the page if it's too large
                sub_chunks = self._split_large_text(
                    text=current_text,
                    start_char=current_start,
                    pages=current_pages,
                    chunk_index=chunk_index,
                    document_id=document.document_id,
                )
                if len(sub_chunks) > 1:
                    chunks.extend(sub_chunks[:-1])
                    chunk_index += len(sub_chunks) - 1
                    last_chunk = sub_chunks[-1]
                    current_text = last_chunk.text
                    current_start = last_chunk.start_char
                    current_pages = last_chunk.page_numbers

        # Flush remaining text
        if current_text.strip():
            chunk = self._create_chunk(
                chunk_index=chunk_index,
                text=current_text,
                start_char=current_start,
                pages=current_pages,
                document_id=document.document_id,
                overlap_chars=0,
            )
            chunks.append(chunk)

        return chunks

    def _chunk_text(
        self,
        text: str,
        document_id: str,
        page_getter: Optional[callable] = None,
    ) -> list[TextChunk]:
        """Chunk text without page structure."""
        if not text.strip():
            return []

        chunks = []
        chunk_index = 0
        remaining_text = text
        current_start = 0

        while remaining_text:
            # Find a good split point
            chunk_text, chars_consumed = self._find_chunk_boundary(remaining_text)

            if not chunk_text.strip():
                break

            # Get pages for this chunk if page_getter is available
            end_char = current_start + len(chunk_text)
            pages = page_getter(current_start, end_char) if page_getter else []

            # Calculate overlap from previous chunk
            overlap_chars = 0
            if chunks:
                prev_chunk = chunks[-1]
                if current_start < prev_chunk.end_char:
                    overlap_chars = prev_chunk.end_char - current_start

            chunk = self._create_chunk(
                chunk_index=chunk_index,
                text=chunk_text,
                start_char=current_start,
                pages=pages,
                document_id=document_id,
                overlap_chars=overlap_chars,
            )
            chunks.append(chunk)
            chunk_index += 1

            # Move to next chunk with overlap
            remaining_text = remaining_text[chars_consumed:]
            current_start += chars_consumed

            # Add overlap from the end of current chunk to start of next
            if remaining_text:
                overlap = self._get_overlap_text(chunk_text)
                if overlap:
                    remaining_text = overlap + remaining_text
                    current_start -= len(overlap)

        return chunks

    def _find_chunk_boundary(self, text: str) -> tuple[str, int]:
        """
        Find a good boundary to split text at.

        Returns:
            Tuple of (chunk_text, chars_consumed)
        """
        total_tokens = self.count_tokens(text)

        # If text fits in target, return it all
        if total_tokens <= self._config.target_tokens:
            return text, len(text)

        # Binary search for the right split point
        target_chars = self._estimate_chars_for_tokens(text, self._config.target_tokens)

        if self._config.respect_sentence_boundaries:
            # Find sentence boundary near target
            split_pos = self._find_sentence_boundary(text, target_chars)
        else:
            split_pos = target_chars

        # Ensure we don't exceed max tokens
        chunk_text = text[:split_pos]
        while self.count_tokens(chunk_text) > self._config.max_tokens and split_pos > 100:
            split_pos = int(split_pos * 0.9)
            if self._config.respect_sentence_boundaries:
                split_pos = self._find_sentence_boundary(text, split_pos)
            chunk_text = text[:split_pos]

        return chunk_text, split_pos

    def _estimate_chars_for_tokens(self, text: str, target_tokens: int) -> int:
        """Estimate character count for a target token count."""
        total_tokens = self.count_tokens(text)
        if total_tokens == 0:
            return len(text)

        # Estimate chars per token ratio
        chars_per_token = len(text) / total_tokens
        return min(len(text), int(target_tokens * chars_per_token))

    def _find_sentence_boundary(self, text: str, near_pos: int) -> int:
        """Find a sentence boundary near the given position."""
        # Look for sentence endings near the position
        search_start = max(0, near_pos - 200)
        search_end = min(len(text), near_pos + 200)
        search_region = text[search_start:search_end]

        # Find all sentence boundaries in the region
        sentence_pattern = r'[.!?]+[\s"\')]*'
        matches = list(re.finditer(sentence_pattern, search_region))

        if not matches:
            # No sentence boundary found, fall back to word boundary
            return self._find_word_boundary(text, near_pos)

        # Find the boundary closest to our target position
        target_in_region = near_pos - search_start
        best_match = min(matches, key=lambda m: abs(m.end() - target_in_region))
        return search_start + best_match.end()

    def _find_word_boundary(self, text: str, near_pos: int) -> int:
        """Find a word boundary near the given position."""
        # Look for whitespace near the position
        search_start = max(0, near_pos - 50)
        search_end = min(len(text), near_pos + 50)

        # Prefer splitting at whitespace after the position
        for i in range(near_pos, search_end):
            if text[i].isspace():
                return i + 1

        # Fall back to whitespace before
        for i in range(near_pos, search_start, -1):
            if text[i].isspace():
                return i + 1

        # No good boundary found
        return near_pos

    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of a chunk."""
        if self._config.overlap_tokens <= 0:
            return ""

        # Estimate characters for overlap tokens
        target_chars = self._estimate_chars_for_tokens(
            text[-1000:] if len(text) > 1000 else text,
            self._config.overlap_tokens
        )

        # Get text from the end
        overlap_text = text[-target_chars:] if target_chars < len(text) else text

        # Try to start at a sentence or word boundary
        sentence_match = re.search(r'[.!?]+[\s"\')]*', overlap_text)
        if sentence_match:
            overlap_text = overlap_text[sentence_match.end():]
        else:
            # Start at word boundary
            word_match = re.search(r'\s+', overlap_text)
            if word_match:
                overlap_text = overlap_text[word_match.end():]

        return overlap_text

    def _split_large_text(
        self,
        text: str,
        start_char: int,
        pages: list[int],
        chunk_index: int,
        document_id: str,
    ) -> list[TextChunk]:
        """Split text that's too large into multiple chunks."""
        chunks = []
        remaining = text
        current_start = start_char
        current_index = chunk_index

        while remaining:
            chunk_text, chars_consumed = self._find_chunk_boundary(remaining)
            if not chunk_text.strip():
                break

            chunk = self._create_chunk(
                chunk_index=current_index,
                text=chunk_text,
                start_char=current_start,
                pages=pages,  # All sub-chunks share the same pages
                document_id=document_id,
                overlap_chars=(
                    0 if current_index == chunk_index
                    else len(self._get_overlap_text(chunks[-1].text))
                ),
            )
            chunks.append(chunk)
            current_index += 1

            remaining = remaining[chars_consumed:]
            current_start += chars_consumed

            # Add overlap
            if remaining:
                overlap = self._get_overlap_text(chunk_text)
                if overlap:
                    remaining = overlap + remaining
                    current_start -= len(overlap)

        return chunks

    def _create_chunk(
        self,
        chunk_index: int,
        text: str,
        start_char: int,
        pages: list[int],
        document_id: str,
        overlap_chars: int,
    ) -> TextChunk:
        """Create a TextChunk object."""
        return TextChunk(
            chunk_index=chunk_index,
            text=text,
            start_char=start_char,
            end_char=start_char + len(text),
            page_numbers=pages,
            token_count=self.count_tokens(text),
            overlap_chars=overlap_chars,
            document_id=document_id,
        )
