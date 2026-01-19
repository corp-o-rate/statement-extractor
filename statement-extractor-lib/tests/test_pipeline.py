"""Tests for the extraction pipeline."""

import pytest


class TestPipelineModels:
    """Tests for pipeline data models."""

    def test_raw_triple_creation(self):
        """Test creating a RawTriple."""
        from statement_extractor.models import RawTriple

        raw = RawTriple(
            subject_text="Apple Inc.",
            predicate_text="announced",
            object_text="new iPhone",
            source_sentence="Apple Inc. announced a new iPhone today.",
        )

        assert raw.subject_text == "Apple Inc."
        assert raw.predicate_text == "announced"
        assert raw.object_text == "new iPhone"
        assert raw.confidence == 1.0

    def test_raw_triple_as_tuple(self):
        """Test RawTriple as_tuple method."""
        from statement_extractor.models import RawTriple

        raw = RawTriple(
            subject_text="Apple",
            predicate_text="makes",
            object_text="iPhones",
            source_sentence="Apple makes iPhones.",
        )

        assert raw.as_tuple() == ("Apple", "makes", "iPhones")

    def test_extracted_entity_creation(self):
        """Test creating an ExtractedEntity."""
        from statement_extractor.models import ExtractedEntity, EntityType

        entity = ExtractedEntity(
            text="Tim Cook",
            type=EntityType.PERSON,
            confidence=0.95,
        )

        assert entity.text == "Tim Cook"
        assert entity.type == EntityType.PERSON
        assert entity.confidence == 0.95
        assert entity.entity_ref is not None  # UUID auto-generated

    def test_pipeline_statement_creation(self):
        """Test creating a PipelineStatement."""
        from statement_extractor.models import (
            PipelineStatement,
            ExtractedEntity,
            EntityType,
        )

        stmt = PipelineStatement(
            subject=ExtractedEntity(text="Apple", type=EntityType.ORG),
            predicate="announced",
            object=ExtractedEntity(text="iPhone", type=EntityType.PRODUCT),
            source_text="Apple announced iPhone.",
            confidence_score=0.9,
        )

        assert stmt.subject.text == "Apple"
        assert stmt.predicate == "announced"
        assert stmt.object.text == "iPhone"
        assert stmt.as_triple() == ("Apple", "announced", "iPhone")

    def test_entity_qualifiers_merge(self):
        """Test merging EntityQualifiers."""
        from statement_extractor.models import EntityQualifiers

        q1 = EntityQualifiers(role="CEO")
        q2 = EntityQualifiers(org="Apple Inc", identifiers={"lei": "123456"})

        merged = q1.merge_with(q2)

        assert merged.role == "CEO"
        assert merged.org == "Apple Inc"
        assert merged.identifiers["lei"] == "123456"

    def test_qualified_entity_creation(self):
        """Test creating a QualifiedEntity."""
        from statement_extractor.models import (
            QualifiedEntity,
            EntityQualifiers,
            EntityType,
        )

        qualified = QualifiedEntity(
            entity_ref="test-123",
            original_text="Tim Cook",
            entity_type=EntityType.PERSON,
            qualifiers=EntityQualifiers(role="CEO", org="Apple Inc"),
        )

        assert qualified.entity_ref == "test-123"
        assert qualified.original_text == "Tim Cook"
        assert qualified.qualifiers.role == "CEO"
        assert qualified.qualifiers.org == "Apple Inc"

    def test_canonical_match_creation(self):
        """Test creating a CanonicalMatch."""
        from statement_extractor.models import CanonicalMatch

        match = CanonicalMatch(
            canonical_id="LEI:12345",
            canonical_name="Apple Inc.",
            match_method="identifier",
            match_confidence=1.0,
        )

        assert match.canonical_id == "LEI:12345"
        assert match.match_method == "identifier"
        assert match.is_high_confidence()

    def test_canonical_entity_from_qualified(self):
        """Test creating CanonicalEntity from QualifiedEntity."""
        from statement_extractor.models import (
            CanonicalEntity,
            QualifiedEntity,
            CanonicalMatch,
            EntityQualifiers,
            EntityType,
        )

        qualified = QualifiedEntity(
            entity_ref="test-123",
            original_text="Tim Cook",
            entity_type=EntityType.PERSON,
            qualifiers=EntityQualifiers(role="CEO", org="Apple Inc"),
        )

        match = CanonicalMatch(
            canonical_name="Timothy D. Cook",
            match_method="name_exact",
            match_confidence=0.95,
        )

        canonical = CanonicalEntity.from_qualified(qualified, match)

        assert canonical.entity_ref == "test-123"
        assert canonical.canonical_match.canonical_name == "Timothy D. Cook"
        # FQN should include role and org
        assert "CEO" in canonical.fqn
        assert "Apple Inc" in canonical.fqn

    def test_statement_label_creation(self):
        """Test creating a StatementLabel."""
        from statement_extractor.models import StatementLabel

        label = StatementLabel(
            label_type="sentiment",
            label_value="positive",
            confidence=0.85,
            labeler="sentiment_labeler",
        )

        assert label.label_type == "sentiment"
        assert label.label_value == "positive"
        assert label.is_high_confidence(threshold=0.8)

    def test_labeled_statement_creation(self):
        """Test creating a LabeledStatement."""
        from statement_extractor.models import (
            LabeledStatement,
            PipelineStatement,
            ExtractedEntity,
            EntityType,
            CanonicalEntity,
            QualifiedEntity,
            StatementLabel,
        )

        stmt = PipelineStatement(
            subject=ExtractedEntity(
                text="Apple",
                type=EntityType.ORG,
                entity_ref="subj-1",
            ),
            predicate="announced",
            object=ExtractedEntity(
                text="iPhone",
                type=EntityType.PRODUCT,
                entity_ref="obj-1",
            ),
            source_text="Apple announced iPhone.",
        )

        subj_qualified = QualifiedEntity(
            entity_ref="subj-1",
            original_text="Apple",
            entity_type=EntityType.ORG,
        )
        obj_qualified = QualifiedEntity(
            entity_ref="obj-1",
            original_text="iPhone",
            entity_type=EntityType.PRODUCT,
        )

        subj_canonical = CanonicalEntity.from_qualified(subj_qualified)
        obj_canonical = CanonicalEntity.from_qualified(obj_qualified)

        labeled = LabeledStatement(
            statement=stmt,
            subject_canonical=subj_canonical,
            object_canonical=obj_canonical,
        )

        # Add a label
        labeled.add_label(StatementLabel(
            label_type="sentiment",
            label_value="neutral",
            confidence=0.7,
        ))

        assert labeled.statement.predicate == "announced"
        assert len(labeled.labels) == 1
        assert labeled.get_label("sentiment").label_value == "neutral"


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from statement_extractor.pipeline import PipelineConfig

        config = PipelineConfig.default()

        assert config.enabled_stages == {1, 2, 3, 4, 5}
        assert config.enabled_plugins is None
        assert len(config.disabled_plugins) == 0

    def test_minimal_config(self):
        """Test minimal configuration."""
        from statement_extractor.pipeline import PipelineConfig

        config = PipelineConfig.minimal()

        assert config.enabled_stages == {1, 2}
        assert 3 not in config.enabled_stages

    def test_from_stage_string(self):
        """Test creating config from stage string."""
        from statement_extractor.pipeline import PipelineConfig

        config = PipelineConfig.from_stage_string("1,2,3")
        assert config.enabled_stages == {1, 2, 3}

        config2 = PipelineConfig.from_stage_string("1-3")
        assert config2.enabled_stages == {1, 2, 3}

        config3 = PipelineConfig.from_stage_string("1-5")
        assert config3.enabled_stages == {1, 2, 3, 4, 5}

    def test_is_stage_enabled(self):
        """Test stage enabled check."""
        from statement_extractor.pipeline import PipelineConfig

        config = PipelineConfig(enabled_stages={1, 2, 3})

        assert config.is_stage_enabled(1)
        assert config.is_stage_enabled(2)
        assert config.is_stage_enabled(3)
        assert not config.is_stage_enabled(4)
        assert not config.is_stage_enabled(5)

    def test_is_plugin_enabled(self):
        """Test plugin enabled check."""
        from statement_extractor.pipeline import PipelineConfig

        config = PipelineConfig(
            disabled_plugins={"plugin_a"},
        )

        assert config.is_plugin_enabled("plugin_b")
        assert not config.is_plugin_enabled("plugin_a")


class TestPipelineContext:
    """Tests for PipelineContext."""

    def test_context_creation(self):
        """Test creating a PipelineContext."""
        from statement_extractor.pipeline import PipelineContext

        ctx = PipelineContext(
            source_text="Apple announced a new iPhone.",
            source_metadata={"doc_id": "123"},
        )

        assert ctx.source_text == "Apple announced a new iPhone."
        assert ctx.source_metadata["doc_id"] == "123"
        assert len(ctx.raw_triples) == 0
        assert len(ctx.statements) == 0

    def test_context_error_handling(self):
        """Test context error handling."""
        from statement_extractor.pipeline import PipelineContext

        ctx = PipelineContext(source_text="Test")

        ctx.add_error("Error 1")
        ctx.add_warning("Warning 1")

        assert len(ctx.processing_errors) == 1
        assert len(ctx.processing_warnings) == 1
        assert ctx.has_errors

    def test_context_timing(self):
        """Test context timing recording."""
        from statement_extractor.pipeline import PipelineContext

        ctx = PipelineContext(source_text="Test")

        ctx.record_timing("splitting", 0.5)
        ctx.record_timing("extraction", 1.2)

        assert ctx.stage_timings["splitting"] == 0.5
        assert ctx.stage_timings["extraction"] == 1.2


class TestPluginRegistry:
    """Tests for PluginRegistry."""

    def test_registry_clear(self):
        """Test clearing the registry."""
        from statement_extractor.pipeline.registry import PluginRegistry

        # Clear to start fresh
        PluginRegistry.clear()

        assert len(PluginRegistry.get_splitters()) == 0
        assert len(PluginRegistry.get_extractors()) == 0

    def test_list_plugins(self):
        """Test listing plugins."""
        from statement_extractor.pipeline.registry import PluginRegistry

        PluginRegistry.clear()

        plugins = PluginRegistry.list_plugins()
        # Should be empty after clear
        assert len(plugins) == 0


class TestPluginBases:
    """Tests for plugin base classes."""

    def test_plugin_capability_flags(self):
        """Test plugin capability flags."""
        from statement_extractor.plugins.base import PluginCapability

        caps = PluginCapability.BATCH_PROCESSING | PluginCapability.EXTERNAL_API

        assert PluginCapability.BATCH_PROCESSING in caps
        assert PluginCapability.EXTERNAL_API in caps
        assert PluginCapability.LLM_REQUIRED not in caps


class TestCanonicalizerHelpers:
    """Tests for canonicalizer helper functions."""

    def test_normalize_org_name(self):
        """Test organization name normalization."""
        from statement_extractor.plugins.canonicalizers.organization import normalize_org_name

        assert normalize_org_name("Apple Inc.") == "apple"
        assert normalize_org_name("Microsoft Corporation") == "microsoft"
        assert normalize_org_name("Google LLC") == "google"
        assert normalize_org_name("Meta Platforms, Inc.") == "meta platforms"

    def test_trigram_similarity(self):
        """Test trigram similarity calculation."""
        from statement_extractor.plugins.canonicalizers.organization import trigram_similarity

        # Same string
        assert trigram_similarity("apple", "apple") == 1.0

        # Similar strings
        sim = trigram_similarity("apple", "appel")
        assert 0.5 < sim < 1.0

        # Different strings
        sim2 = trigram_similarity("apple", "microsoft")
        assert sim2 < 0.3

    def test_person_name_matching(self):
        """Test person name matching."""
        from statement_extractor.plugins.canonicalizers.person import names_match

        # Exact match
        match, conf = names_match("Tim Cook", "Tim Cook")
        assert match
        assert conf == 1.0

        # Variant match
        match, conf = names_match("Tim Cook", "Timothy Cook")
        assert match
        assert conf == 0.9

        # Different people
        match, conf = names_match("Tim Cook", "John Smith")
        assert not match

    def test_location_normalization(self):
        """Test location normalization."""
        from statement_extractor.plugins.canonicalizers.location import normalize_location

        assert normalize_location("U.S.A.") == "usa"
        assert normalize_location(" United States ") == "united states"
        assert normalize_location("U.K.") == "uk"


class TestLabelerHelpers:
    """Tests for labeler helper functions."""

    def test_sentiment_classification(self):
        """Test sentiment classification."""
        from statement_extractor.plugins.labelers.sentiment import classify_sentiment

        # Positive
        sentiment, conf = classify_sentiment("Company announced great quarterly earnings")
        assert sentiment == "positive"

        # Negative
        sentiment, conf = classify_sentiment("Company lost significant market share")
        assert sentiment == "negative"

        # Neutral
        sentiment, conf = classify_sentiment("Company is located in California")
        assert sentiment == "neutral"

    def test_relation_type_classification(self):
        """Test relation type classification."""
        from statement_extractor.plugins.labelers.relation_type import classify_relation

        # Employment
        rel_type, conf = classify_relation("works for", "John works for Apple Inc.")
        assert rel_type == "employment"

        # Acquisition
        rel_type, conf = classify_relation("acquired", "Microsoft acquired LinkedIn.")
        assert rel_type == "acquisition"

        # Location
        rel_type, conf = classify_relation("headquartered", "Company is headquartered in SF.")
        assert rel_type == "location"


class TestQualifierPlugins:
    """Tests for qualifier plugins."""

    def test_person_qualifier_patterns(self):
        """Test person qualifier pattern matching."""
        from statement_extractor.plugins.qualifiers.person import PersonQualifierPlugin
        from statement_extractor.models import ExtractedEntity, EntityType
        from statement_extractor.pipeline import PipelineContext, PipelineStatement

        plugin = PersonQualifierPlugin(use_llm=False)  # Disable LLM for testing

        # Create a context with a statement
        ctx = PipelineContext(source_text="Tim Cook, CEO of Apple, announced...")
        ctx.statements = [
            PipelineStatement(
                subject=ExtractedEntity(text="Tim Cook", type=EntityType.PERSON, entity_ref="tim-1"),
                predicate="announced",
                object=ExtractedEntity(text="iPhone", type=EntityType.PRODUCT, entity_ref="iphone-1"),
                source_text="Tim Cook, CEO of Apple, announced the new iPhone.",
            )
        ]

        entity = ctx.statements[0].subject
        qualifiers = plugin.qualify(entity, ctx)

        # Should extract role from pattern matching
        assert qualifiers is not None
        assert qualifiers.role == "CEO"


@pytest.mark.slow
class TestPipelineIntegration:
    """Integration tests for the full pipeline (requires models)."""

    def test_pipeline_context_flow(self):
        """Test that context flows through pipeline stages correctly."""
        from statement_extractor.pipeline import PipelineConfig, PipelineContext

        # Create a context
        ctx = PipelineContext(
            source_text="Apple Inc. announced the new iPhone 15.",
        )

        # Verify initial state
        assert len(ctx.raw_triples) == 0
        assert len(ctx.statements) == 0
        assert len(ctx.qualified_entities) == 0
        assert len(ctx.canonical_entities) == 0
        assert len(ctx.labeled_statements) == 0
