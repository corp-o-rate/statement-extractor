"""
ExtractionPipeline - Main orchestrator for the 5-stage extraction pipeline.

Coordinates the flow of data through all pipeline stages:
1. Splitting: Text → RawTriple
2. Extraction: RawTriple → PipelineStatement
3. Qualification: Entity → QualifiedEntity
4. Canonicalization: QualifiedEntity → CanonicalEntity
5. Labeling: Statement → LabeledStatement
"""

import logging
import time
from typing import Any, Optional

from .context import PipelineContext
from .config import PipelineConfig, get_stage_name
from .registry import PluginRegistry
from ..models import (
    QualifiedEntity,
    EntityQualifiers,
    CanonicalEntity,
    LabeledStatement,
    TaxonomyResult,
)

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """
    Main pipeline orchestrator.

    Coordinates the flow of data through all 5 stages, invoking registered
    plugins in priority order and accumulating results in PipelineContext.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration (uses defaults if not provided)
        """
        self.config = config or PipelineConfig.default()

    def process(
        self,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> PipelineContext:
        """
        Process text through the extraction pipeline.

        Args:
            text: Input text to process
            metadata: Optional metadata about the source

        Returns:
            PipelineContext with accumulated results from all stages
        """
        # Merge config options into metadata for plugins
        combined_metadata = metadata.copy() if metadata else {}

        # Pass extractor options from config to context
        if self.config.extractor_options:
            existing_extractor_opts = combined_metadata.get("extractor_options", {})
            combined_metadata["extractor_options"] = {
                **self.config.extractor_options,
                **existing_extractor_opts,  # Allow explicit metadata to override config
            }

        ctx = PipelineContext(
            source_text=text,
            source_metadata=combined_metadata,
        )

        logger.info(f"Starting pipeline processing: {len(text)} chars")

        try:
            # Stage 1: Splitting
            if self.config.is_stage_enabled(1):
                ctx = self._run_splitting(ctx)

            # Stage 2: Extraction
            if self.config.is_stage_enabled(2):
                ctx = self._run_extraction(ctx)

            # Stage 3: Qualification
            if self.config.is_stage_enabled(3):
                ctx = self._run_qualification(ctx)

            # Stage 4: Canonicalization
            if self.config.is_stage_enabled(4):
                ctx = self._run_canonicalization(ctx)

            # Stage 5: Labeling
            if self.config.is_stage_enabled(5):
                ctx = self._run_labeling(ctx)

            # Stage 6: Taxonomy classification
            if self.config.is_stage_enabled(6):
                ctx = self._run_taxonomy(ctx)

        except Exception as e:
            logger.exception("Pipeline processing failed")
            ctx.add_error(f"Pipeline error: {str(e)}")
            if self.config.fail_fast:
                raise

        logger.info(
            f"Pipeline complete: {ctx.statement_count} statements, "
            f"{len(ctx.processing_errors)} errors"
        )

        return ctx

    def _run_splitting(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 1: Split text into raw triples."""
        stage_name = get_stage_name(1)
        logger.debug(f"Running {stage_name} stage")
        start_time = time.time()

        splitters = PluginRegistry.get_splitters()
        if not splitters:
            ctx.add_warning("No splitter plugins registered")
            return ctx

        # Use first enabled splitter (highest priority)
        for splitter in splitters:
            if not self.config.is_plugin_enabled(splitter.name):
                continue

            logger.debug(f"Using splitter: {splitter.name}")
            try:
                raw_triples = splitter.split(ctx.source_text, ctx)
                ctx.raw_triples = raw_triples
                logger.info(f"Splitting produced {len(raw_triples)} raw triples")
                break
            except Exception as e:
                logger.exception(f"Splitter {splitter.name} failed")
                ctx.add_error(f"Splitter {splitter.name} failed: {str(e)}")
                if self.config.fail_fast:
                    raise

        ctx.record_timing(stage_name, time.time() - start_time)
        return ctx

    def _run_extraction(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 2: Extract statements with typed entities from raw triples."""
        stage_name = get_stage_name(2)
        logger.debug(f"Running {stage_name} stage")
        start_time = time.time()

        if not ctx.raw_triples:
            logger.debug("No raw triples to extract from")
            return ctx

        extractors = PluginRegistry.get_extractors()
        if not extractors:
            ctx.add_warning("No extractor plugins registered")
            return ctx

        # Collect classification schemas from labelers for the extractor
        classification_schemas = self._collect_classification_schemas()
        if classification_schemas:
            logger.debug(f"Collected {len(classification_schemas)} classification schemas from labelers")

        # Use first enabled extractor (highest priority)
        for extractor in extractors:
            if not self.config.is_plugin_enabled(extractor.name):
                continue

            # Pass classification schemas to extractor if it supports them
            if classification_schemas and hasattr(extractor, 'add_classification_schema'):
                for schema in classification_schemas:
                    extractor.add_classification_schema(schema)

            logger.debug(f"Using extractor: {extractor.name}")
            try:
                statements = extractor.extract(ctx.raw_triples, ctx)
                ctx.statements = statements
                logger.info(f"Extraction produced {len(statements)} statements")
                break
            except Exception as e:
                logger.exception(f"Extractor {extractor.name} failed")
                ctx.add_error(f"Extractor {extractor.name} failed: {str(e)}")
                if self.config.fail_fast:
                    raise

        ctx.record_timing(stage_name, time.time() - start_time)
        return ctx

    def _collect_classification_schemas(self) -> list:
        """Collect classification schemas from enabled labelers."""
        schemas = []
        labelers = PluginRegistry.get_labelers()

        for labeler in labelers:
            if not self.config.is_plugin_enabled(labeler.name):
                continue

            # Check for classification schema (simple multi-choice)
            if hasattr(labeler, 'classification_schema') and labeler.classification_schema:
                schemas.append(labeler.classification_schema)
                logger.debug(
                    f"Labeler {labeler.name} provides classification schema: "
                    f"{labeler.classification_schema}"
                )

        return schemas

    def _run_qualification(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 3: Add qualifiers to entities."""
        stage_name = get_stage_name(3)
        logger.debug(f"Running {stage_name} stage")
        start_time = time.time()

        if not ctx.statements:
            logger.debug("No statements to qualify")
            return ctx

        # Collect all unique entities from statements
        entities_to_qualify = {}
        for stmt in ctx.statements:
            for entity in [stmt.subject, stmt.object]:
                if entity.entity_ref not in entities_to_qualify:
                    entities_to_qualify[entity.entity_ref] = entity

        logger.debug(f"Qualifying {len(entities_to_qualify)} unique entities")

        # Qualify each entity using applicable plugins
        for entity_ref, entity in entities_to_qualify.items():
            qualifiers = EntityQualifiers()
            sources = []

            # Get qualifiers for this entity type
            type_qualifiers = PluginRegistry.get_qualifiers_for_type(entity.type)

            for qualifier_plugin in type_qualifiers:
                if not self.config.is_plugin_enabled(qualifier_plugin.name):
                    continue

                try:
                    plugin_qualifiers = qualifier_plugin.qualify(entity, ctx)
                    if plugin_qualifiers and plugin_qualifiers.has_any_qualifier():
                        qualifiers = qualifiers.merge_with(plugin_qualifiers)
                        sources.append(qualifier_plugin.name)
                except Exception as e:
                    logger.error(f"Qualifier {qualifier_plugin.name} failed for {entity.text}: {e}")
                    ctx.add_error(f"Qualifier {qualifier_plugin.name} failed: {str(e)}")
                    if self.config.fail_fast:
                        raise

            # Create QualifiedEntity
            qualified = QualifiedEntity(
                entity_ref=entity_ref,
                original_text=entity.text,
                entity_type=entity.type,
                qualifiers=qualifiers,
                qualification_sources=sources,
            )
            ctx.qualified_entities[entity_ref] = qualified

        logger.info(f"Qualified {len(ctx.qualified_entities)} entities")
        ctx.record_timing(stage_name, time.time() - start_time)
        return ctx

    def _run_canonicalization(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 4: Resolve entities to canonical forms."""
        stage_name = get_stage_name(4)
        logger.debug(f"Running {stage_name} stage")
        start_time = time.time()

        if not ctx.qualified_entities:
            # Create basic qualified entities if stage 3 was skipped
            for stmt in ctx.statements:
                for entity in [stmt.subject, stmt.object]:
                    if entity.entity_ref not in ctx.qualified_entities:
                        ctx.qualified_entities[entity.entity_ref] = QualifiedEntity(
                            entity_ref=entity.entity_ref,
                            original_text=entity.text,
                            entity_type=entity.type,
                        )

        # Canonicalize each qualified entity
        for entity_ref, qualified in ctx.qualified_entities.items():
            canonical_match = None
            fqn = None

            # Get canonicalizers for this entity type
            type_canonicalizers = PluginRegistry.get_canonicalizers_for_type(qualified.entity_type)

            for canon_plugin in type_canonicalizers:
                if not self.config.is_plugin_enabled(canon_plugin.name):
                    continue

                try:
                    match = canon_plugin.find_canonical(qualified, ctx)
                    if match:
                        canonical_match = match
                        fqn = canon_plugin.format_fqn(qualified, match)
                        break  # Use first successful match
                except Exception as e:
                    logger.error(f"Canonicalizer {canon_plugin.name} failed for {qualified.original_text}: {e}")
                    ctx.add_error(f"Canonicalizer {canon_plugin.name} failed: {str(e)}")
                    if self.config.fail_fast:
                        raise

            # Create CanonicalEntity
            canonical = CanonicalEntity.from_qualified(
                qualified=qualified,
                canonical_match=canonical_match,
                fqn=fqn,
            )
            ctx.canonical_entities[entity_ref] = canonical

        logger.info(f"Canonicalized {len(ctx.canonical_entities)} entities")
        ctx.record_timing(stage_name, time.time() - start_time)
        return ctx

    def _run_labeling(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 5: Apply labels to statements."""
        stage_name = get_stage_name(5)
        logger.debug(f"Running {stage_name} stage")
        start_time = time.time()

        if not ctx.statements:
            logger.debug("No statements to label")
            return ctx

        # Ensure canonical entities exist
        if not ctx.canonical_entities:
            self._run_canonicalization(ctx)

        labelers = PluginRegistry.get_labelers()

        for stmt in ctx.statements:
            # Get canonical entities
            subj_canonical = ctx.canonical_entities.get(stmt.subject.entity_ref)
            obj_canonical = ctx.canonical_entities.get(stmt.object.entity_ref)

            if not subj_canonical or not obj_canonical:
                # Create fallback canonical entities
                if not subj_canonical:
                    subj_qualified = ctx.qualified_entities.get(
                        stmt.subject.entity_ref,
                        QualifiedEntity(
                            entity_ref=stmt.subject.entity_ref,
                            original_text=stmt.subject.text,
                            entity_type=stmt.subject.type,
                        )
                    )
                    subj_canonical = CanonicalEntity.from_qualified(subj_qualified)

                if not obj_canonical:
                    obj_qualified = ctx.qualified_entities.get(
                        stmt.object.entity_ref,
                        QualifiedEntity(
                            entity_ref=stmt.object.entity_ref,
                            original_text=stmt.object.text,
                            entity_type=stmt.object.type,
                        )
                    )
                    obj_canonical = CanonicalEntity.from_qualified(obj_qualified)

            # Create labeled statement
            labeled = LabeledStatement(
                statement=stmt,
                subject_canonical=subj_canonical,
                object_canonical=obj_canonical,
            )

            # Apply all labelers
            for labeler in labelers:
                if not self.config.is_plugin_enabled(labeler.name):
                    continue

                try:
                    label = labeler.label(stmt, subj_canonical, obj_canonical, ctx)
                    if label:
                        labeled.add_label(label)
                except Exception as e:
                    logger.error(f"Labeler {labeler.name} failed: {e}")
                    ctx.add_error(f"Labeler {labeler.name} failed: {str(e)}")
                    if self.config.fail_fast:
                        raise

            ctx.labeled_statements.append(labeled)

        logger.info(f"Labeled {len(ctx.labeled_statements)} statements")
        ctx.record_timing(stage_name, time.time() - start_time)
        return ctx

    def _run_taxonomy(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 6: Classify statements against taxonomies."""
        stage_name = get_stage_name(6)
        logger.debug(f"Running {stage_name} stage")
        start_time = time.time()

        if not ctx.labeled_statements:
            logger.debug("No labeled statements to classify")
            return ctx

        taxonomy_classifiers = PluginRegistry.get_taxonomy_classifiers()
        if not taxonomy_classifiers:
            logger.debug("No taxonomy classifiers registered")
            return ctx

        total_results = 0
        for labeled_stmt in ctx.labeled_statements:
            stmt = labeled_stmt.statement
            subj_canonical = labeled_stmt.subject_canonical
            obj_canonical = labeled_stmt.object_canonical

            # Apply all taxonomy classifiers
            for classifier in taxonomy_classifiers:
                if not self.config.is_plugin_enabled(classifier.name):
                    continue

                try:
                    results = classifier.classify(stmt, subj_canonical, obj_canonical, ctx)
                    if results:
                        # Store taxonomy results in context (list of results per key)
                        key = (stmt.source_text, classifier.taxonomy_name)
                        if key not in ctx.taxonomy_results:
                            ctx.taxonomy_results[key] = []
                        ctx.taxonomy_results[key].extend(results)
                        total_results += len(results)

                        # Also add to the labeled statement for easy access
                        labeled_stmt.taxonomy_results.extend(results)

                        for result in results:
                            logger.debug(
                                f"Taxonomy {classifier.name}: {result.full_label} "
                                f"(confidence={result.confidence:.2f})"
                            )
                except Exception as e:
                    logger.error(f"Taxonomy classifier {classifier.name} failed: {e}")
                    ctx.add_error(f"Taxonomy classifier {classifier.name} failed: {str(e)}")
                    if self.config.fail_fast:
                        raise

        logger.info(f"Taxonomy produced {total_results} labels across {len(ctx.taxonomy_results)} statement-taxonomy pairs")
        ctx.record_timing(stage_name, time.time() - start_time)
        return ctx
