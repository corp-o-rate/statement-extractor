"""Tests for document processing pipeline."""

import pytest


# =============================================================================
# Document Model Tests
# =============================================================================

class TestDocumentModels:
    """Tests for document-related models."""

    def test_document_metadata_format_citation_full(self):
        """Test citation formatting with all fields."""
        from statement_extractor.models.document import DocumentMetadata

        metadata = DocumentMetadata(
            title="Annual Report",
            authors=["John Smith"],
            year=2024,
        )

        citation = metadata.format_citation(page_number=5)
        assert "Annual Report" in citation
        assert "John Smith" in citation
        assert "2024" in citation
        assert "p. 5" in citation

    def test_document_metadata_format_citation_multiple_authors(self):
        """Test citation with multiple authors."""
        from statement_extractor.models.document import DocumentMetadata

        metadata = DocumentMetadata(
            title="Research Paper",
            authors=["Alice", "Bob", "Charlie"],
            year=2023,
        )

        citation = metadata.format_citation()
        assert "Alice et al." in citation

    def test_document_metadata_format_citation_two_authors(self):
        """Test citation with two authors."""
        from statement_extractor.models.document import DocumentMetadata

        metadata = DocumentMetadata(
            title="Joint Paper",
            authors=["Alice", "Bob"],
        )

        citation = metadata.format_citation()
        assert "Alice & Bob" in citation

    def test_document_metadata_format_citation_empty(self):
        """Test citation with no metadata."""
        from statement_extractor.models.document import DocumentMetadata

        metadata = DocumentMetadata()
        citation = metadata.format_citation()
        assert citation == ""

    def test_document_from_text(self):
        """Test creating document from plain text."""
        from statement_extractor.models.document import Document

        text = "This is a test document with some text."
        doc = Document.from_text(text, title="Test Doc", url="https://example.com")

        assert doc.full_text == text
        assert doc.metadata.title == "Test Doc"
        assert doc.metadata.url == "https://example.com"
        assert doc.metadata.source_type == "text"
        assert doc.document_id  # Should have auto-generated ID

    def test_document_from_pages(self):
        """Test creating document from pages."""
        from statement_extractor.models.document import Document

        pages = [
            "Page 1 content here.",
            "Page 2 content here.",
            "Page 3 content here.",
        ]

        doc = Document.from_pages(pages, title="PDF Document")

        assert len(doc.pages) == 3
        assert doc.pages[0].page_number == 1
        assert doc.pages[1].page_number == 2
        assert doc.pages[2].page_number == 3

        # Check character offsets
        assert doc.pages[0].char_offset == 0
        assert doc.pages[1].char_offset > 0

    def test_document_get_page_at_char(self):
        """Test finding page by character offset."""
        from statement_extractor.models.document import Document

        pages = ["Short.", "Medium text here.", "Longer text content."]
        doc = Document.from_pages(pages)

        # Character in first page
        assert doc.get_page_at_char(2) == 1

        # Character in second page
        page2_start = doc.pages[1].char_offset
        assert doc.get_page_at_char(page2_start + 2) == 2

    def test_document_get_pages_in_range(self):
        """Test finding pages in character range."""
        from statement_extractor.models.document import Document

        pages = ["First page.", "Second page.", "Third page."]
        doc = Document.from_pages(pages)

        # Range spanning multiple pages
        pages_found = doc.get_pages_in_range(5, doc.char_count - 5)
        assert len(pages_found) >= 2

    def test_text_chunk_model(self):
        """Test TextChunk model."""
        from statement_extractor.models.document import TextChunk

        chunk = TextChunk(
            chunk_index=0,
            text="Sample chunk text",
            start_char=0,
            end_char=17,
            page_numbers=[1, 2],
            token_count=4,
            document_id="test-doc-id",
        )

        assert chunk.primary_page == 1
        assert chunk.chunk_index == 0

    def test_chunking_config_defaults(self):
        """Test ChunkingConfig default values."""
        from statement_extractor.models.document import ChunkingConfig

        config = ChunkingConfig()

        assert config.max_tokens == 2000
        assert config.target_tokens == 1000
        assert config.overlap_tokens == 100
        assert config.respect_page_boundaries is True
        assert config.respect_sentence_boundaries is True


# =============================================================================
# Chunker Tests
# =============================================================================

class TestDocumentChunker:
    """Tests for DocumentChunker."""

    def test_chunker_init(self):
        """Test chunker initialization."""
        from statement_extractor.document import DocumentChunker
        from statement_extractor.models.document import ChunkingConfig

        config = ChunkingConfig(target_tokens=500)
        chunker = DocumentChunker(config)

        assert chunker._config.target_tokens == 500

    def test_chunker_empty_document(self):
        """Test chunking empty document."""
        from statement_extractor.document import DocumentChunker, Document

        doc = Document.from_text("")
        chunker = DocumentChunker()
        chunks = chunker.chunk_document(doc)

        assert chunks == []

    @pytest.mark.slow
    def test_chunker_small_document(self):
        """Test chunking document smaller than target."""
        from statement_extractor.document import DocumentChunker, Document
        from statement_extractor.models.document import ChunkingConfig

        text = "This is a small document that should fit in one chunk."
        doc = Document.from_text(text)

        config = ChunkingConfig(target_tokens=1000)
        chunker = DocumentChunker(config)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].chunk_index == 0

    @pytest.mark.slow
    def test_chunker_large_document(self):
        """Test chunking document that needs multiple chunks."""
        from statement_extractor.document import DocumentChunker, Document
        from statement_extractor.models.document import ChunkingConfig

        # Create text that should need multiple chunks
        text = "This is a sentence. " * 500
        doc = Document.from_text(text)

        config = ChunkingConfig(target_tokens=100, max_tokens=200, overlap_tokens=20)
        chunker = DocumentChunker(config)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 1

        # Check chunk indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    @pytest.mark.slow
    def test_chunker_count_tokens(self):
        """Test token counting."""
        from statement_extractor.document import DocumentChunker

        chunker = DocumentChunker()
        count = chunker.count_tokens("Hello world")

        assert count > 0
        assert isinstance(count, int)

    @pytest.mark.slow
    def test_chunker_with_pages(self):
        """Test chunking document with page structure."""
        from statement_extractor.document import DocumentChunker, Document
        from statement_extractor.models.document import ChunkingConfig

        pages = [
            "This is page one with some content.",
            "This is page two with more content.",
            "This is page three with final content.",
        ]
        doc = Document.from_pages(pages)

        config = ChunkingConfig(
            target_tokens=50,
            max_tokens=100,
            respect_page_boundaries=True,
        )
        chunker = DocumentChunker(config)
        chunks = chunker.chunk_document(doc)

        # All chunks should have page numbers
        for chunk in chunks:
            assert len(chunk.page_numbers) > 0


# =============================================================================
# Deduplicator Tests
# =============================================================================

class TestStatementDeduplicator:
    """Tests for StatementDeduplicator."""

    def test_deduplicator_init(self):
        """Test deduplicator initialization."""
        from statement_extractor.document import StatementDeduplicator

        dedup = StatementDeduplicator()
        assert dedup.seen_count == 0

    def test_deduplicator_reset(self):
        """Test resetting deduplicator state."""
        from statement_extractor.document import StatementDeduplicator
        from statement_extractor.models.statement import PipelineStatement
        from statement_extractor.models.entity import ExtractedEntity, EntityType

        dedup = StatementDeduplicator()

        stmt = PipelineStatement(
            subject=ExtractedEntity(text="Apple", type=EntityType.ORG),
            predicate="announced",
            object=ExtractedEntity(text="iPhone", type=EntityType.PRODUCT),
            source_text="Apple announced iPhone.",
        )

        dedup.is_duplicate(stmt)
        assert dedup.seen_count == 1

        dedup.reset()
        assert dedup.seen_count == 0

    def test_deduplicator_exact_duplicate(self):
        """Test detecting exact duplicates."""
        from statement_extractor.document import StatementDeduplicator
        from statement_extractor.models.statement import PipelineStatement
        from statement_extractor.models.entity import ExtractedEntity, EntityType

        dedup = StatementDeduplicator()

        stmt1 = PipelineStatement(
            subject=ExtractedEntity(text="Apple", type=EntityType.ORG),
            predicate="announced",
            object=ExtractedEntity(text="iPhone", type=EntityType.PRODUCT),
            source_text="Apple announced iPhone.",
        )

        stmt2 = PipelineStatement(
            subject=ExtractedEntity(text="Apple", type=EntityType.ORG),
            predicate="announced",
            object=ExtractedEntity(text="iPhone", type=EntityType.PRODUCT),
            source_text="Different source.",
        )

        assert not dedup.is_duplicate(stmt1)
        assert dedup.is_duplicate(stmt2)

    def test_deduplicator_case_insensitive(self):
        """Test case-insensitive duplicate detection."""
        from statement_extractor.document import StatementDeduplicator
        from statement_extractor.models.statement import PipelineStatement
        from statement_extractor.models.entity import ExtractedEntity, EntityType

        dedup = StatementDeduplicator()

        stmt1 = PipelineStatement(
            subject=ExtractedEntity(text="Apple", type=EntityType.ORG),
            predicate="announced",
            object=ExtractedEntity(text="iPhone", type=EntityType.PRODUCT),
            source_text="Apple announced iPhone.",
        )

        stmt2 = PipelineStatement(
            subject=ExtractedEntity(text="APPLE", type=EntityType.ORG),
            predicate="ANNOUNCED",
            object=ExtractedEntity(text="IPHONE", type=EntityType.PRODUCT),
            source_text="Different source.",
        )

        assert not dedup.is_duplicate(stmt1)
        assert dedup.is_duplicate(stmt2)

    def test_deduplicator_filter_list(self):
        """Test filtering duplicates from a list."""
        from statement_extractor.document import StatementDeduplicator
        from statement_extractor.models.statement import PipelineStatement
        from statement_extractor.models.entity import ExtractedEntity, EntityType

        dedup = StatementDeduplicator()

        statements = [
            PipelineStatement(
                subject=ExtractedEntity(text="Apple", type=EntityType.ORG),
                predicate="announced",
                object=ExtractedEntity(text="iPhone", type=EntityType.PRODUCT),
                source_text="Source 1.",
            ),
            PipelineStatement(
                subject=ExtractedEntity(text="Apple", type=EntityType.ORG),
                predicate="announced",
                object=ExtractedEntity(text="iPhone", type=EntityType.PRODUCT),
                source_text="Source 2.",
            ),
            PipelineStatement(
                subject=ExtractedEntity(text="Google", type=EntityType.ORG),
                predicate="released",
                object=ExtractedEntity(text="Pixel", type=EntityType.PRODUCT),
                source_text="Source 3.",
            ),
        ]

        filtered = dedup.filter_duplicates(statements)

        assert len(filtered) == 2
        assert filtered[0].subject.text == "Apple"
        assert filtered[1].subject.text == "Google"

    def test_deduplicator_with_pipeline_statements(self):
        """Test deduplication with PipelineStatement."""
        from statement_extractor.document import StatementDeduplicator
        from statement_extractor.models.statement import PipelineStatement
        from statement_extractor.models.entity import ExtractedEntity, EntityType

        dedup = StatementDeduplicator()

        stmt1 = PipelineStatement(
            subject=ExtractedEntity(text="Apple", type=EntityType.ORG),
            predicate="announced",
            object=ExtractedEntity(text="iPhone", type=EntityType.PRODUCT),
            source_text="Apple announced iPhone.",
        )

        stmt2 = PipelineStatement(
            subject=ExtractedEntity(text="Apple", type=EntityType.ORG),
            predicate="announced",
            object=ExtractedEntity(text="iPhone", type=EntityType.PRODUCT),
            source_text="Different source.",
        )

        assert not dedup.is_duplicate(stmt1)
        assert dedup.is_duplicate(stmt2)


# =============================================================================
# Document Context Tests
# =============================================================================

class TestDocumentContext:
    """Tests for DocumentContext."""

    def test_document_context_init(self):
        """Test DocumentContext initialization."""
        from statement_extractor.document import DocumentContext, Document

        doc = Document.from_text("Test document")
        ctx = DocumentContext(document=doc)

        assert ctx.statement_count == 0
        assert ctx.chunk_count == 0
        assert ctx.duplicates_removed == 0

    def test_document_context_add_error(self):
        """Test adding errors to context."""
        from statement_extractor.document import DocumentContext, Document

        doc = Document.from_text("Test")
        ctx = DocumentContext(document=doc)

        ctx.add_error("Test error")
        assert len(ctx.processing_errors) == 1
        assert "Test error" in ctx.processing_errors[0]

    def test_document_context_record_timing(self):
        """Test recording stage timings."""
        from statement_extractor.document import DocumentContext, Document

        doc = Document.from_text("Test")
        ctx = DocumentContext(document=doc)

        ctx.record_timing("stage1", 1.5)
        ctx.record_timing("stage1", 0.5)  # Should accumulate

        assert ctx.stage_timings["stage1"] == 2.0

    def test_document_context_as_dict(self):
        """Test converting context to dict."""
        from statement_extractor.document import DocumentContext, Document

        doc = Document.from_text("Test", title="Test Doc")
        ctx = DocumentContext(document=doc)

        result = ctx.as_dict()

        assert result["document_title"] == "Test Doc"
        assert "statement_count" in result
        assert "chunk_count" in result


# =============================================================================
# Document Pipeline Config Tests
# =============================================================================

class TestDocumentPipelineConfig:
    """Tests for DocumentPipelineConfig."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from statement_extractor.document import DocumentPipelineConfig

        config = DocumentPipelineConfig()

        assert config.generate_summary is True
        assert config.deduplicate_across_chunks is True
        assert config.batch_size == 10
        assert config.chunking is not None

    def test_config_custom_values(self):
        """Test custom configuration."""
        from statement_extractor.document import DocumentPipelineConfig
        from statement_extractor.models.document import ChunkingConfig

        chunking = ChunkingConfig(target_tokens=500)
        config = DocumentPipelineConfig(
            chunking=chunking,
            generate_summary=False,
            batch_size=20,
        )

        assert config.chunking.target_tokens == 500
        assert config.generate_summary is False
        assert config.batch_size == 20


# =============================================================================
# Statement Model Document Field Tests
# =============================================================================

class TestStatementDocumentFields:
    """Tests for document fields on statement models."""

    def test_raw_triple_document_fields(self):
        """Test document fields on RawTriple (now SplitSentence)."""
        from statement_extractor.models.statement import RawTriple

        # RawTriple is now an alias for SplitSentence which has a single text field
        triple = RawTriple(
            text="Apple announced iPhone.",
            document_id="doc-123",
            page_number=5,
            chunk_index=2,
        )

        assert triple.document_id == "doc-123"
        assert triple.page_number == 5
        assert triple.chunk_index == 2

    def test_pipeline_statement_document_fields(self):
        """Test document fields on PipelineStatement."""
        from statement_extractor.models.statement import PipelineStatement
        from statement_extractor.models.entity import ExtractedEntity, EntityType

        stmt = PipelineStatement(
            subject=ExtractedEntity(text="Apple", type=EntityType.ORG),
            predicate="announced",
            object=ExtractedEntity(text="iPhone", type=EntityType.PRODUCT),
            source_text="Apple announced iPhone.",
            document_id="doc-456",
            page_number=10,
            chunk_index=3,
        )

        assert stmt.document_id == "doc-456"
        assert stmt.page_number == 10
        assert stmt.chunk_index == 3

    def test_labeled_statement_document_fields(self):
        """Test document fields on LabeledStatement."""
        from statement_extractor.models.statement import PipelineStatement
        from statement_extractor.models.labels import LabeledStatement
        from statement_extractor.models.canonical import CanonicalEntity
        from statement_extractor.models.qualifiers import QualifiedEntity
        from statement_extractor.models.entity import ExtractedEntity, EntityType

        stmt = PipelineStatement(
            subject=ExtractedEntity(text="Apple", type=EntityType.ORG),
            predicate="announced",
            object=ExtractedEntity(text="iPhone", type=EntityType.PRODUCT),
            source_text="Apple announced iPhone.",
        )

        subj_qualified = QualifiedEntity(
            entity_ref="apple_ref",
            original_text="Apple",
            entity_type=EntityType.ORG,
        )
        obj_qualified = QualifiedEntity(
            entity_ref="iphone_ref",
            original_text="iPhone",
            entity_type=EntityType.PRODUCT,
        )

        labeled = LabeledStatement(
            statement=stmt,
            subject_canonical=CanonicalEntity.from_qualified(subj_qualified),
            object_canonical=CanonicalEntity.from_qualified(obj_qualified),
            document_id="doc-789",
            page_number=15,
            citation="Annual Report - Apple, 2024, p. 15",
        )

        assert labeled.document_id == "doc-789"
        assert labeled.page_number == 15
        assert labeled.citation == "Annual Report - Apple, 2024, p. 15"

        # Test as_dict includes document fields
        result = labeled.as_dict()
        assert result["document_id"] == "doc-789"
        assert result["page_number"] == 15
        assert result["citation"] == "Annual Report - Apple, 2024, p. 15"
