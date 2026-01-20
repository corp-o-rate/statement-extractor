"""
EmbeddingCompanyQualifier - Qualifies ORG entities using embedding similarity.

Uses a local embedding database to:
1. Find similar company names by embedding
2. Use LLM to confirm the best match
3. Return qualified entity with canonical ID
"""

import logging
from typing import Optional

from ..base import BaseQualifierPlugin, PluginCapability
from ...pipeline.context import PipelineContext
from ...pipeline.registry import PluginRegistry
from ...models import ExtractedEntity, EntityQualifiers, EntityType

logger = logging.getLogger(__name__)


# LLM prompt template for company matching confirmation
COMPANY_MATCH_PROMPT = """You are matching a company name extracted from text to a database of known companies.

Extracted name: "{query_name}"
{context_line}
Candidate matches (sorted by similarity):
{candidates}

Task: Select the BEST match, or respond "NONE" if no candidate is a good match.

Rules:
- The match should refer to the same legal entity
- Minor spelling differences or abbreviations are OK (e.g., "Apple" matches "Apple Inc.")
- Different companies with similar names should NOT match
- Consider the REGION when matching - prefer companies from regions mentioned in or relevant to the context
- If the extracted name is too generic or ambiguous, respond "NONE"

Respond with ONLY the number of the best match (1, 2, 3, etc.) or "NONE".
"""


@PluginRegistry.qualifier
class EmbeddingCompanyQualifier(BaseQualifierPlugin):
    """
    Qualifier plugin for ORG entities using embedding similarity.

    Uses a pre-built embedding database to find and confirm company matches.
    This runs before API-based qualifiers (GLEIF, Companies House, SEC Edgar)
    and provides faster, offline matching when the database is available.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        top_k: int = 20,
        min_similarity: float = 0.5,
        use_llm_confirmation: bool = True,
        auto_download_db: bool = True,
    ):
        """
        Initialize the embedding company qualifier.

        Args:
            db_path: Path to company database (auto-detects if None)
            top_k: Number of candidates to retrieve
            min_similarity: Minimum similarity threshold
            use_llm_confirmation: Whether to use LLM for match confirmation
            auto_download_db: Whether to auto-download database from HuggingFace
        """
        self._db_path = db_path
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._use_llm_confirmation = use_llm_confirmation
        self._auto_download_db = auto_download_db

        # Lazy-loaded components
        self._database = None
        self._embedder = None
        self._llm = None
        self._cache: dict[str, Optional[EntityQualifiers]] = {}

    @property
    def name(self) -> str:
        return "embedding_company_qualifier"

    @property
    def priority(self) -> int:
        return 5  # Runs before API-based qualifiers (GLEIF=10, CH=20, SEC=30)

    @property
    def capabilities(self) -> PluginCapability:
        caps = PluginCapability.CACHING | PluginCapability.BATCH_PROCESSING
        if self._use_llm_confirmation:
            caps |= PluginCapability.LLM_REQUIRED
        return caps

    @property
    def description(self) -> str:
        return "Qualifies ORG entities using embedding similarity search with optional LLM confirmation"

    @property
    def supported_entity_types(self) -> set[EntityType]:
        return {EntityType.ORG}

    @property
    def supported_identifier_types(self) -> list[str]:
        return ["lei", "sec_cik", "ch_number"]

    @property
    def provided_identifier_types(self) -> list[str]:
        return ["lei", "sec_cik", "ch_number", "canonical_id"]

    def _get_database(self):
        """Get or initialize the company database."""
        if self._database is not None:
            return self._database

        from ...database import CompanyDatabase
        from ...database.hub import get_database_path

        # Find database path
        db_path = self._db_path
        if db_path is None:
            db_path = get_database_path(auto_download=self._auto_download_db)

        if db_path is None:
            logger.warning("Company database not available. Skipping embedding qualification.")
            return None

        self._database = CompanyDatabase(db_path=db_path)
        logger.info(f"Loaded company database from {db_path}")
        return self._database

    def _get_embedder(self):
        """Get or initialize the embedder."""
        if self._embedder is not None:
            return self._embedder

        from ...database import CompanyEmbedder
        self._embedder = CompanyEmbedder()
        return self._embedder

    def _get_llm(self):
        """Get or initialize the LLM for confirmation."""
        if self._llm is not None:
            return self._llm

        if not self._use_llm_confirmation:
            return None

        try:
            from ...llm import get_llm
            self._llm = get_llm()
            return self._llm
        except Exception as e:
            logger.warning(f"LLM not available for confirmation: {e}")
            return None

    def qualify(
        self,
        entity: ExtractedEntity,
        context: PipelineContext,
    ) -> Optional[EntityQualifiers]:
        """
        Qualify an ORG entity using embedding similarity.

        Args:
            entity: The ORG entity to qualify
            context: Pipeline context

        Returns:
            EntityQualifiers with identifiers, or None if no match
        """
        if entity.type != EntityType.ORG:
            return None

        # Check cache
        cache_key = entity.text.lower().strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Get database
        database = self._get_database()
        if database is None:
            return None

        # Get embedder
        embedder = self._get_embedder()

        # Embed query name
        logger.debug(f"    Embedding query: '{entity.text}'")
        query_embedding = embedder.embed(entity.text)

        # Search for similar companies
        logger.debug(f"    Searching database for similar companies...")
        results = database.search(query_embedding, top_k=self._top_k)

        # Filter by minimum similarity
        results = [(r, s) for r, s in results if s >= self._min_similarity]

        if not results:
            logger.debug(f"    No matches found above threshold {self._min_similarity}")
            self._cache[cache_key] = None
            return None

        logger.info(f"    Found {len(results)} candidates, top: '{results[0][0].name}' ({results[0][1]:.3f})")

        # Get best match (optionally with LLM confirmation)
        logger.debug(f"    Selecting best match (LLM={self._use_llm_confirmation})...")
        best_match = self._select_best_match(entity.text, results, context)

        if best_match is None:
            logger.info(f"    No confident match for '{entity.text}'")
            self._cache[cache_key] = None
            return None

        record, similarity = best_match
        logger.info(f"    Matched: '{record.legal_name}' (source={record.source}, similarity={similarity:.3f})")

        # Build qualifiers from matched record
        qualifiers = self._build_qualifiers(record, similarity)

        self._cache[cache_key] = qualifiers
        return qualifiers

    def _select_best_match(
        self,
        query_name: str,
        candidates: list[tuple],
        context: "PipelineContext",
    ) -> Optional[tuple]:
        """
        Select the best match from candidates.

        Uses LLM if available and configured, otherwise returns top match.
        """
        if not candidates:
            return None

        # If only one strong match, use it directly
        if len(candidates) == 1 and candidates[0][1] >= 0.9:
            logger.debug(f"Single strong match: {candidates[0][0].name} ({candidates[0][1]:.3f})")
            return candidates[0]

        # Try LLM confirmation
        llm = self._get_llm()
        if llm is not None:
            try:
                return self._llm_select_match(query_name, candidates, context)
            except Exception as e:
                logger.debug(f"LLM confirmation failed: {e}")

        # Fallback: use top match if similarity is high enough
        top_record, top_similarity = candidates[0]
        if top_similarity >= 0.85:
            logger.debug(f"Using top match: {top_record.name} ({top_similarity:.3f})")
            return candidates[0]

        logger.debug(f"No confident match for '{query_name}' (top: {top_similarity:.3f})")
        return None

    def _llm_select_match(
        self,
        query_name: str,
        candidates: list[tuple],
        context: "PipelineContext",
    ) -> Optional[tuple]:
        """Use LLM to select the best match."""
        # Format candidates for prompt with region info
        candidate_lines = []
        for i, (record, similarity) in enumerate(candidates[:10], 1):  # Limit to top 10
            region_str = f", region: {record.region}" if record.region else ""
            candidate_lines.append(
                f"{i}. {record.legal_name} (source: {record.source}{region_str}, similarity: {similarity:.3f})"
            )

        # Build context line from source text if available
        context_line = ""
        if context.source_text:
            # Truncate source text for prompt
            source_preview = context.source_text[:500] + "..." if len(context.source_text) > 500 else context.source_text
            context_line = f"Source text context: \"{source_preview}\"\n"

        prompt = COMPANY_MATCH_PROMPT.format(
            query_name=query_name,
            context_line=context_line,
            candidates="\n".join(candidate_lines),
        )

        # Get LLM response
        response = self._llm.generate(prompt, max_tokens=10, stop=["\n"])
        response = response.strip()

        logger.debug(f"LLM match response for '{query_name}': {response}")

        # Parse response
        if response.upper() == "NONE":
            return None

        try:
            idx = int(response) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]
        except ValueError:
            pass

        # Fallback to top match if LLM response is unclear
        return candidates[0] if candidates[0][1] >= 0.8 else None

    def _build_qualifiers(self, record, similarity: float) -> EntityQualifiers:
        """Build EntityQualifiers from a matched record."""
        identifiers = {}

        # Add source-specific identifiers
        source = record.source
        source_id = record.source_id

        if source == "gleif":
            identifiers["lei"] = source_id
        elif source == "sec_edgar":
            identifiers["sec_cik"] = source_id
            if record.record.get("ticker"):
                identifiers["ticker"] = record.record["ticker"]
        elif source == "companies_house":
            identifiers["ch_number"] = source_id

        # Add canonical ID
        identifiers["canonical_id"] = record.canonical_id

        # Extract location info from record
        record_data = record.record
        jurisdiction = record_data.get("jurisdiction")
        country = record_data.get("country")
        city = record_data.get("city")

        return EntityQualifiers(
            jurisdiction=jurisdiction,
            country=country,
            city=city,
            identifiers=identifiers,
        )


# For testing without decorator
EmbeddingCompanyQualifierClass = EmbeddingCompanyQualifier
