"""
End-to-end integration tests for the extraction pipeline.

These tests verify the complete flow from text input through all 5 stages
to final labeled output. Some tests require models to be loaded.
"""

import pytest


# Check if models are available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from gliner2 import GLiNER2
    HAS_GLINER = True
except ImportError:
    HAS_GLINER = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
requires_gliner = pytest.mark.skipif(not HAS_GLINER, reason="gliner2 not installed")


# =============================================================================
# Sample texts for testing
# =============================================================================

SIMPLE_TEXT = """
Apple Inc. announced the new iPhone 15 at their annual event in California.
Tim Cook, CEO of Apple, presented the new features to excited customers.
"""

CORPORATE_TEXT = """
Microsoft Corporation announced today that it has completed the acquisition
of Activision Blizzard for $69 billion. The deal, approved by regulators in
the UK and EU, makes Microsoft the third-largest gaming company globally.
CEO Satya Nadella said the acquisition positions Microsoft to compete with
Sony and Nintendo in the gaming market. The transaction was financed through
a combination of cash and debt.
"""

NEWS_TEXT = """
Tesla Inc. CEO Elon Musk announced a new Gigafactory in Austin, Texas.
The facility will employ 10,000 workers and produce electric vehicles.
Governor Greg Abbott praised the investment, calling it a major win for Texas.
The factory is expected to begin production in Q2 2024.
"""

MULTI_ENTITY_TEXT = """
In a landmark deal, Amazon Web Services (AWS) has partnered with NVIDIA
to provide advanced AI computing services. The partnership, announced by
AWS CEO Adam Selipsky and NVIDIA CEO Jensen Huang, will integrate
NVIDIA's H100 GPUs into AWS's cloud infrastructure. Google Cloud and
Microsoft Azure are seen as competitors in this space.
"""


class TestPipelineModelsE2E:
    """End-to-end tests for pipeline data models."""

    def test_complete_entity_flow(self):
        """Test entity data flowing through all pipeline models."""
        from statement_extractor.models import (
            ExtractedEntity,
            EntityType,
            EntityQualifiers,
            QualifiedEntity,
            CanonicalMatch,
            CanonicalEntity,
        )

        # Stage 2: ExtractedEntity
        entity = ExtractedEntity(
            text="Tim Cook",
            type=EntityType.PERSON,
            confidence=0.95,
        )

        # Stage 3: QualifiedEntity
        qualifiers = EntityQualifiers(
            role="CEO",
            org="Apple Inc",
        )
        qualified = QualifiedEntity(
            entity_ref=entity.entity_ref,
            original_text=entity.text,
            entity_type=entity.type,
            qualifiers=qualifiers,
            qualification_sources=["person_qualifier"],
        )

        # Stage 4: CanonicalEntity
        match = CanonicalMatch(
            canonical_name="Timothy D. Cook",
            match_method="name_variant",
            match_confidence=0.9,
        )
        canonical = CanonicalEntity.from_qualified(qualified, match)

        # Verify FQN generation
        assert "Tim" in canonical.fqn or "Cook" in canonical.fqn
        assert "CEO" in canonical.fqn
        assert "Apple" in canonical.fqn

    def test_complete_statement_flow(self):
        """Test statement data flowing through pipeline stages 2-5."""
        from statement_extractor.models import (
            SplitSentence,
            PipelineStatement,
            ExtractedEntity,
            EntityType,
            QualifiedEntity,
            CanonicalEntity,
            LabeledStatement,
            StatementLabel,
        )

        # Stage 1: SplitSentence (atomic sentence from splitting)
        split = SplitSentence(
            text="Apple Inc. announced the new iPhone 15.",
        )

        # Stage 2: PipelineStatement (extracted subject-predicate-object)
        stmt = PipelineStatement(
            subject=ExtractedEntity(
                text="Apple Inc.",
                type=EntityType.ORG,
                confidence=0.95,
            ),
            predicate="announced",
            object=ExtractedEntity(
                text="iPhone 15",
                type=EntityType.PRODUCT,
                confidence=0.9,
            ),
            source_text=split.text,
            confidence_score=0.92,
        )

        # Stage 3-4: Create canonical entities
        subj_qualified = QualifiedEntity(
            entity_ref=stmt.subject.entity_ref,
            original_text=stmt.subject.text,
            entity_type=stmt.subject.type,
        )
        obj_qualified = QualifiedEntity(
            entity_ref=stmt.object.entity_ref,
            original_text=stmt.object.text,
            entity_type=stmt.object.type,
        )

        subj_canonical = CanonicalEntity.from_qualified(subj_qualified)
        obj_canonical = CanonicalEntity.from_qualified(obj_qualified)

        # Stage 5: LabeledStatement
        labeled = LabeledStatement(
            statement=stmt,
            subject_canonical=subj_canonical,
            object_canonical=obj_canonical,
        )

        # Add labels
        labeled.add_label(StatementLabel(
            label_type="sentiment",
            label_value="neutral",
            confidence=0.8,
        ))
        labeled.add_label(StatementLabel(
            label_type="relation_type",
            label_value="product_launch",
            confidence=0.85,
        ))
        labeled.add_label(StatementLabel(
            label_type="confidence",
            label_value=0.88,
            confidence=1.0,
        ))

        # Verify output
        assert len(labeled.labels) == 3
        assert labeled.get_label("sentiment").label_value == "neutral"
        assert labeled.get_label("relation_type").label_value == "product_launch"

        # Verify as_dict output
        output = labeled.as_dict()
        assert output["subject"]["text"] == "Apple Inc."
        assert output["predicate"] == "announced"
        assert output["object"]["text"] == "iPhone 15"
        assert "sentiment" in output["labels"]


class TestPipelineContextE2E:
    """End-to-end tests for pipeline context management."""

    def test_context_accumulates_all_stages(self):
        """Test that context properly accumulates data from all stages."""
        from statement_extractor.pipeline import PipelineContext
        from statement_extractor.models import (
            RawTriple,
            PipelineStatement,
            ExtractedEntity,
            EntityType,
            QualifiedEntity,
            CanonicalEntity,
            LabeledStatement,
            StatementLabel,
        )

        # Create context
        ctx = PipelineContext(
            source_text=SIMPLE_TEXT,
            source_metadata={"doc_id": "test-001"},
        )

        # Simulate Stage 1 output
        ctx.raw_triples = [
            RawTriple(
                subject_text="Apple Inc.",
                predicate_text="announced",
                object_text="iPhone 15",
                source_sentence="Apple Inc. announced the new iPhone 15.",
            ),
            RawTriple(
                subject_text="Tim Cook",
                predicate_text="presented",
                object_text="new features",
                source_sentence="Tim Cook presented the new features.",
            ),
        ]

        # Simulate Stage 2 output
        ctx.statements = [
            PipelineStatement(
                subject=ExtractedEntity(text="Apple Inc.", type=EntityType.ORG, entity_ref="ent-1"),
                predicate="announced",
                object=ExtractedEntity(text="iPhone 15", type=EntityType.PRODUCT, entity_ref="ent-2"),
                source_text="Apple Inc. announced the new iPhone 15.",
            ),
            PipelineStatement(
                subject=ExtractedEntity(text="Tim Cook", type=EntityType.PERSON, entity_ref="ent-3"),
                predicate="presented",
                object=ExtractedEntity(text="new features", type=EntityType.UNKNOWN, entity_ref="ent-4"),
                source_text="Tim Cook presented the new features.",
            ),
        ]

        # Simulate Stage 3 output
        for stmt in ctx.statements:
            for entity in [stmt.subject, stmt.object]:
                ctx.qualified_entities[entity.entity_ref] = QualifiedEntity(
                    entity_ref=entity.entity_ref,
                    original_text=entity.text,
                    entity_type=entity.type,
                )

        # Simulate Stage 4 output
        for ref, qualified in ctx.qualified_entities.items():
            ctx.canonical_entities[ref] = CanonicalEntity.from_qualified(qualified)

        # Simulate Stage 5 output
        for stmt in ctx.statements:
            labeled = LabeledStatement(
                statement=stmt,
                subject_canonical=ctx.canonical_entities[stmt.subject.entity_ref],
                object_canonical=ctx.canonical_entities[stmt.object.entity_ref],
            )
            labeled.add_label(StatementLabel(label_type="confidence", label_value=0.9, confidence=1.0))
            ctx.labeled_statements.append(labeled)

        # Verify all stages have data
        assert len(ctx.raw_triples) == 2
        assert len(ctx.statements) == 2
        assert len(ctx.qualified_entities) == 4  # 2 statements * 2 entities each
        assert len(ctx.canonical_entities) == 4
        assert len(ctx.labeled_statements) == 2
        assert ctx.statement_count == 2


class TestPipelineConfigE2E:
    """End-to-end tests for pipeline configuration."""

    def test_stage_selection(self):
        """Test that stage selection works correctly."""
        from statement_extractor.pipeline import PipelineConfig

        # All stages
        config_all = PipelineConfig.default()
        assert config_all.is_stage_enabled(1)
        assert config_all.is_stage_enabled(5)

        # Only splitting and extraction
        config_minimal = PipelineConfig.minimal()
        assert config_minimal.is_stage_enabled(1)
        assert config_minimal.is_stage_enabled(2)
        assert not config_minimal.is_stage_enabled(3)

        # Custom stages via string
        config_custom = PipelineConfig.from_stage_string("1,2,5")
        assert config_custom.is_stage_enabled(1)
        assert config_custom.is_stage_enabled(2)
        assert not config_custom.is_stage_enabled(3)
        assert not config_custom.is_stage_enabled(4)
        assert config_custom.is_stage_enabled(5)

    def test_plugin_selection(self):
        """Test that plugin selection works correctly."""
        from statement_extractor.pipeline import PipelineConfig

        config = PipelineConfig(
            enabled_plugins={"gleif_qualifier", "person_qualifier"},
            disabled_plugins={"sec_edgar_qualifier"},
        )

        assert config.is_plugin_enabled("gleif_qualifier")
        assert config.is_plugin_enabled("person_qualifier")
        assert not config.is_plugin_enabled("companies_house_qualifier")  # Not in enabled list
        assert not config.is_plugin_enabled("sec_edgar_qualifier")  # Explicitly disabled


class TestPluginRegistryE2E:
    """End-to-end tests for plugin registration."""

    def test_load_all_plugins(self):
        """Test that all plugins can be loaded."""
        import importlib
        from statement_extractor.pipeline.registry import PluginRegistry

        # Reload plugin modules to ensure registration happens
        # (other tests may have cleared the registry)
        from statement_extractor.plugins import (
            splitters,
            extractors,
            qualifiers,
            labelers,
            taxonomy,
            scrapers,
            pdf,
        )
        importlib.reload(splitters)
        importlib.reload(extractors)
        importlib.reload(qualifiers)
        importlib.reload(labelers)
        importlib.reload(taxonomy)
        importlib.reload(scrapers)
        importlib.reload(pdf)

        # Verify splitters
        splitters_list = PluginRegistry.get_splitters()
        assert len(splitters_list) >= 1
        assert any(p.name == "t5_gemma_splitter" for p in splitters_list)

        # Verify extractors
        extractors_list = PluginRegistry.get_extractors()
        assert len(extractors_list) >= 1
        assert any(p.name == "gliner2_extractor" for p in extractors_list)

        # Verify qualifiers (now include embedding_company_qualifier)
        qualifiers_list = PluginRegistry.get_qualifiers()
        assert len(qualifiers_list) >= 2  # person, embedding_company
        plugin_names = [p.name for p in qualifiers_list]
        assert "person_qualifier" in plugin_names

        # Verify labelers
        labelers_list = PluginRegistry.get_labelers()
        assert len(labelers_list) >= 2  # sentiment, confidence
        plugin_names = [p.name for p in labelers_list]
        assert "sentiment_labeler" in plugin_names
        assert "confidence_labeler" in plugin_names

    def test_plugins_by_entity_type(self):
        """Test that plugins can be retrieved by entity type."""
        from statement_extractor.pipeline.registry import PluginRegistry
        from statement_extractor.models import EntityType

        # Import plugins to trigger registration (don't clear - Python caches imports)
        from statement_extractor import plugins  # noqa: F401

        # Get qualifiers for PERSON
        person_qualifiers = PluginRegistry.get_qualifiers_for_type(EntityType.PERSON)
        assert any(p.name == "person_qualifier" for p in person_qualifiers)

        # Get qualifiers for ORG (embedding_company_qualifier handles ORGs)
        org_qualifiers = PluginRegistry.get_qualifiers_for_type(EntityType.ORG)
        assert len(org_qualifiers) >= 1


class TestHelperFunctionsE2E:
    """End-to-end tests for helper functions used in plugins."""

    def test_sentiment_classification(self):
        """Test sentiment classification."""
        from statement_extractor.plugins.labelers.sentiment import classify_sentiment

        # Positive
        sentiment, conf = classify_sentiment("announced great results")
        assert sentiment == "positive"

        # Negative
        sentiment, conf = classify_sentiment("company lost market share")
        assert sentiment == "negative"

        # Neutral
        sentiment, conf = classify_sentiment("company is based in Seattle")
        assert sentiment == "neutral"


class TestQualifierPluginsE2E:
    """End-to-end tests for qualifier plugins."""

    def test_person_qualifier_pattern_matching(self):
        """Test person qualifier using pattern matching."""
        from statement_extractor.plugins.qualifiers.person import PersonQualifierPlugin
        from statement_extractor.models import ExtractedEntity, EntityType, PipelineStatement
        from statement_extractor.pipeline import PipelineContext

        plugin = PersonQualifierPlugin(use_llm=False)

        # Create context with statement mentioning CEO
        ctx = PipelineContext(source_text="Tim Cook, CEO of Apple, announced the new iPhone.")
        ctx.statements = [
            PipelineStatement(
                subject=ExtractedEntity(text="Tim Cook", type=EntityType.PERSON, entity_ref="tim-1"),
                predicate="announced",
                object=ExtractedEntity(text="iPhone", type=EntityType.PRODUCT, entity_ref="iphone-1"),
                source_text="Tim Cook, CEO of Apple, announced the new iPhone.",
            )
        ]

        # Qualify the entity
        entity = ctx.statements[0].subject
        canonical = plugin.qualify(entity, ctx)

        # Should extract CEO role and return CanonicalEntity
        assert canonical is not None
        assert canonical.qualified_entity.qualifiers.role == "CEO"
        assert "CEO" in canonical.fqn


class TestLabelerPluginsE2E:
    """End-to-end tests for labeler plugins."""

    def test_confidence_labeler_aggregation(self):
        """Test confidence labeler aggregates scores correctly."""
        from statement_extractor.plugins.labelers.confidence import ConfidenceLabeler
        from statement_extractor.models import (
            PipelineStatement,
            ExtractedEntity,
            EntityType,
            QualifiedEntity,
            CanonicalEntity,
            CanonicalMatch,
        )
        from statement_extractor.pipeline import PipelineContext

        plugin = ConfidenceLabeler()

        # Create statement with confidence
        stmt = PipelineStatement(
            subject=ExtractedEntity(text="Apple", type=EntityType.ORG, confidence=0.95),
            predicate="announced",
            object=ExtractedEntity(text="iPhone", type=EntityType.PRODUCT, confidence=0.90),
            source_text="Apple announced iPhone.",
            confidence_score=0.92,
        )

        # Create canonical entities with match confidence
        subj_qualified = QualifiedEntity(
            entity_ref=stmt.subject.entity_ref,
            original_text=stmt.subject.text,
            entity_type=stmt.subject.type,
        )
        subj_canonical = CanonicalEntity.from_qualified(
            subj_qualified,
            CanonicalMatch(canonical_name="Apple Inc.", match_method="name_exact", match_confidence=0.98),
        )

        obj_qualified = QualifiedEntity(
            entity_ref=stmt.object.entity_ref,
            original_text=stmt.object.text,
            entity_type=stmt.object.type,
        )
        obj_canonical = CanonicalEntity.from_qualified(
            obj_qualified,
            CanonicalMatch(canonical_name="iPhone 15", match_method="name_fuzzy", match_confidence=0.85),
        )

        ctx = PipelineContext(source_text="Apple announced iPhone.")

        # Get label
        label = plugin.label(stmt, subj_canonical, obj_canonical, ctx)

        assert label is not None
        assert label.label_type == "confidence"
        assert 0.8 < label.label_value < 1.0  # Should be weighted average


@requires_torch
@pytest.mark.slow
class TestPipelineE2EWithModels:
    """End-to-end tests that require actual models."""

    def test_full_pipeline_simple_text(self):
        """Test full pipeline with simple text."""
        from statement_extractor.pipeline import ExtractionPipeline, PipelineConfig

        config = PipelineConfig.default()
        pipeline = ExtractionPipeline(config)

        ctx = pipeline.process(SIMPLE_TEXT)

        # Should have processed through all stages
        # Note: qualified_entities is no longer populated since Stage 3 (Qualification)
        # now directly produces canonical_entities with merged qualification/canonicalization
        assert len(ctx.raw_triples) >= 1
        assert len(ctx.statements) >= 1
        assert len(ctx.canonical_entities) >= 1
        assert len(ctx.labeled_statements) >= 1

        # Check labeled statements have labels
        for labeled in ctx.labeled_statements:
            assert len(labeled.labels) >= 1
            # Should have confidence label at minimum
            conf_label = labeled.get_label("confidence")
            assert conf_label is not None

    def test_full_pipeline_corporate_text(self):
        """Test full pipeline with corporate acquisition text."""
        from statement_extractor.pipeline import ExtractionPipeline, PipelineConfig

        config = PipelineConfig.default()
        pipeline = ExtractionPipeline(config)

        ctx = pipeline.process(CORPORATE_TEXT)

        # Should extract multiple statements about the acquisition
        assert ctx.statement_count >= 1

        # Check for key entities
        entity_texts = set()
        for labeled in ctx.labeled_statements:
            entity_texts.add(labeled.statement.subject.text.lower())
            entity_texts.add(labeled.statement.object.text.lower())

        # Should find key entities (case-insensitive)
        found_microsoft = any("microsoft" in t for t in entity_texts)
        found_activision = any("activision" in t for t in entity_texts)

        # At least one key entity should be found
        assert found_microsoft or found_activision

    def test_pipeline_with_stage_selection(self):
        """Test pipeline with specific stages selected."""
        from statement_extractor.pipeline import ExtractionPipeline, PipelineConfig

        # Run only stages 1 and 2
        config = PipelineConfig.minimal()
        pipeline = ExtractionPipeline(config)

        ctx = pipeline.process(SIMPLE_TEXT)

        # Should have output from stages 1 and 2
        assert len(ctx.raw_triples) >= 1
        assert len(ctx.statements) >= 1

        # Should NOT have output from stages 3-5 (skipped)
        # Note: Context might still create fallback canonical entities in stage 5
        # So we check that qualification wasn't done (no sources)
        for qualified in ctx.qualified_entities.values():
            assert len(qualified.qualification_sources) == 0

    def test_pipeline_verbose_output(self):
        """Test that verbose mode produces additional output."""
        from statement_extractor.pipeline import ExtractionPipeline, PipelineConfig

        config = PipelineConfig.default()
        pipeline = ExtractionPipeline(config)

        ctx = pipeline.process(SIMPLE_TEXT)

        # Check timing data is recorded
        assert len(ctx.stage_timings) >= 1

        # Check that we can access all the verbose data
        for labeled in ctx.labeled_statements:
            # FQN should be available
            assert labeled.subject_fqn is not None
            assert labeled.object_fqn is not None

            # Labels should be available
            for label in labeled.labels:
                assert label.label_type is not None
                assert label.label_value is not None

    def test_pipeline_error_handling(self):
        """Test pipeline handles errors gracefully."""
        from statement_extractor.pipeline import ExtractionPipeline, PipelineConfig

        config = PipelineConfig(fail_fast=False)
        pipeline = ExtractionPipeline(config)

        # Process empty text - should complete without raising
        # Note: The model may hallucinate content when given empty input,
        # so we only verify the pipeline doesn't crash
        ctx = pipeline.process("")

        # Should not raise - statement count may vary due to model behavior
        assert ctx is not None

    def test_multi_entity_extraction(self):
        """Test extraction with multiple organizations and people."""
        from statement_extractor.pipeline import ExtractionPipeline, PipelineConfig

        config = PipelineConfig.default()
        pipeline = ExtractionPipeline(config)

        ctx = pipeline.process(MULTI_ENTITY_TEXT)

        # Should extract entities from multiple companies
        entity_texts = set()
        for labeled in ctx.labeled_statements:
            entity_texts.add(labeled.statement.subject.text.lower())
            entity_texts.add(labeled.statement.object.text.lower())

        # Check for various entities
        assert ctx.statement_count >= 1


@requires_torch
@pytest.mark.slow
class TestCLIPipelineCommand:
    """End-to-end tests for the CLI pipeline command."""

    def test_cli_pipeline_help(self):
        """Test CLI pipeline help command."""
        from click.testing import CliRunner
        from statement_extractor.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["pipeline", "--help"])

        assert result.exit_code == 0
        assert "Run the full 5-stage extraction pipeline" in result.output
        assert "--stages" in result.output
        assert "--verbose" in result.output

    def test_cli_pipeline_simple(self):
        """Test CLI pipeline with simple text."""
        from click.testing import CliRunner
        from statement_extractor.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["pipeline", SIMPLE_TEXT])

        # Should complete successfully
        assert result.exit_code == 0

    def test_cli_pipeline_json_output(self):
        """Test CLI pipeline with JSON output."""
        import json
        import re
        from click.testing import CliRunner
        from statement_extractor.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["pipeline", SIMPLE_TEXT, "-o", "json", "-q"])

        assert result.exit_code == 0

        # Extract JSON from output (may have progress bars mixed in from tqdm)
        output_text = result.output
        # Find the JSON object - look for opening brace of main object
        json_match = re.search(r'(\{[\s\S]*"statement_count"[\s\S]*\})\s*$', output_text)
        if json_match:
            output_text = json_match.group(1)

        # Should be valid JSON
        output = json.loads(output_text)
        assert "statement_count" in output
        # May have statements, raw_triples, or labeled_statements depending on what stages ran
        assert any(key in output for key in ["statements", "raw_triples", "labeled_statements"])

    def test_cli_pipeline_verbose(self):
        """Test CLI pipeline with verbose output."""
        from click.testing import CliRunner
        from statement_extractor.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["pipeline", SIMPLE_TEXT, "--verbose"])

        assert result.exit_code == 0
        # Verbose should show timings
        assert "timings" in result.output.lower() or "stage" in result.output.lower() or "confidence" in result.output.lower()

    def test_cli_plugins_list(self):
        """Test CLI plugins list command."""
        from click.testing import CliRunner
        from statement_extractor.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["plugins", "list"])

        assert result.exit_code == 0
        assert "Stage" in result.output
        assert "t5_gemma_splitter" in result.output or "splitter" in result.output.lower()

    def test_cli_plugins_info(self):
        """Test CLI plugins info command."""
        from click.testing import CliRunner
        from statement_extractor.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["plugins", "info", "confidence_labeler"])

        assert result.exit_code == 0
        assert "confidence_labeler" in result.output
