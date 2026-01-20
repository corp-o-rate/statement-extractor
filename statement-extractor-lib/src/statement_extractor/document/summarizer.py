"""
DocumentSummarizer - Generate document summaries using Gemma3.

Creates concise summaries focused on entities, events, and relationships
that are useful for providing context during extraction.
"""

import logging
from typing import Optional

from ..models.document import Document

logger = logging.getLogger(__name__)


class DocumentSummarizer:
    """
    Generates document summaries using the Gemma3 LLM.

    Summaries focus on:
    - Key entities mentioned
    - Important events and actions
    - Relationships between entities
    """

    MAX_INPUT_TOKENS = 10_000
    DEFAULT_MAX_OUTPUT_TOKENS = 300

    def __init__(
        self,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ):
        """
        Initialize the summarizer.

        Args:
            max_input_tokens: Maximum tokens of input to send to the LLM
            max_output_tokens: Maximum tokens for the summary output
        """
        self._max_input_tokens = max_input_tokens
        self._max_output_tokens = max_output_tokens
        self._llm = None
        self._tokenizer = None

    @property
    def llm(self):
        """Lazy-load the LLM."""
        if self._llm is None:
            from ..llm import get_llm
            logger.debug("Loading LLM for summarization")
            self._llm = get_llm()
        return self._llm

    @property
    def tokenizer(self):
        """Lazy-load tokenizer for token counting."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                "Corp-o-Rate-Community/statement-extractor",
                trust_remote_code=True,
            )
        return self._tokenizer

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to a maximum number of tokens.

        Tries to truncate at sentence boundaries when possible.
        """
        token_count = self._count_tokens(text)

        if token_count <= max_tokens:
            return text

        # Estimate chars per token
        chars_per_token = len(text) / token_count
        target_chars = int(max_tokens * chars_per_token * 0.95)  # 5% buffer

        # Truncate
        truncated = text[:target_chars]

        # Try to end at a sentence boundary
        last_period = truncated.rfind(". ")
        last_newline = truncated.rfind("\n")
        split_pos = max(last_period, last_newline)

        if split_pos > target_chars * 0.7:  # Don't lose too much text
            truncated = truncated[:split_pos + 1]

        logger.debug(f"Truncated text from {len(text)} to {len(truncated)} chars")
        return truncated

    def summarize(
        self,
        document: Document,
        custom_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a summary of the document.

        Args:
            document: Document to summarize
            custom_prompt: Optional custom prompt (uses default if not provided)

        Returns:
            Summary string
        """
        if not document.full_text.strip():
            logger.warning("Cannot summarize empty document")
            return ""

        logger.info(f"Generating summary for document {document.document_id}")

        # Truncate text to max input tokens
        text = self._truncate_to_tokens(document.full_text, self._max_input_tokens)

        # Build prompt
        if custom_prompt:
            prompt = f"{custom_prompt}\n\n{text}"
        else:
            prompt = self._build_prompt(text, document)

        # Generate summary
        try:
            summary = self.llm.generate(
                prompt=prompt,
                max_tokens=self._max_output_tokens,
                stop=["\n\n\n", "---"],
            )
            summary = summary.strip()
            logger.info(f"Generated summary ({len(summary)} chars):")
            # Log summary with indentation for readability
            for line in summary.split("\n"):
                logger.info(f"  {line}")
            return summary

        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            raise

    def _build_prompt(self, text: str, document: Document) -> str:
        """Build the summarization prompt."""
        # Include document metadata context if available
        context_parts = []
        if document.metadata.title:
            context_parts.append(f"Title: {document.metadata.title}")
        if document.metadata.authors:
            context_parts.append(f"Authors: {', '.join(document.metadata.authors)}")
        if document.metadata.source_type:
            context_parts.append(f"Source type: {document.metadata.source_type}")

        context = "\n".join(context_parts) if context_parts else ""

        prompt = f"""Summarize the following document, focusing on:
1. Key entities (companies, people, locations) mentioned
2. Important events, actions, and decisions
3. Relationships between entities
4. Main topics and themes

Keep the summary concise (2-3 paragraphs) and factual.

{context}

Document text:
{text}

Summary:"""

        return prompt

    def summarize_text(
        self,
        text: str,
        title: Optional[str] = None,
    ) -> str:
        """
        Generate a summary from plain text.

        Convenience method that creates a temporary Document.

        Args:
            text: Text to summarize
            title: Optional document title for context

        Returns:
            Summary string
        """
        document = Document.from_text(text, title=title)
        return self.summarize(document)
