"""
HuggingFace Hub integration for entity/organization database distribution.

Provides functionality to:
- Download pre-built entity databases from HuggingFace Hub
- Upload/publish database updates
- Version management for database files
- Create "lite" versions without full records for smaller downloads
"""

import logging
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default HuggingFace repo for entity database
DEFAULT_REPO_ID = "Corp-o-Rate-Community/entity-references"
DEFAULT_DB_FILENAME = "entities-lite.db"  # Lite is the default (smaller download)
DEFAULT_DB_FULL_FILENAME = "entities.db"
DEFAULT_DB_LITE_FILENAME = "entities-lite.db"

# Local cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "corp-extractor"


def get_database_path(
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_DB_FILENAME,
    auto_download: bool = True,
    full: bool = False,
) -> Optional[Path]:
    """
    Get path to entity database, downloading if necessary.

    Args:
        repo_id: HuggingFace repo ID
        filename: Database filename (overrides full flag if specified)
        auto_download: Whether to download if not cached
        full: If True, get the full database instead of lite

    Returns:
        Path to database file, or None if not available
    """
    # Override filename if full is requested and using default
    if full and filename == DEFAULT_DB_FILENAME:
        filename = DEFAULT_DB_FULL_FILENAME
    # Check if database exists in cache
    cache_dir = DEFAULT_CACHE_DIR

    # Check common locations
    possible_paths = [
        cache_dir / filename,
        cache_dir / "entities.db",
        Path.home() / ".cache" / "huggingface" / "hub" / f"datasets--{repo_id.replace('/', '--')}" / filename,
    ]

    for path in possible_paths:
        if path.exists():
            logger.debug(f"Found cached database at {path}")
            return path

    # Try to download
    if auto_download:
        try:
            return download_database(repo_id=repo_id, filename=filename)
        except Exception as e:
            logger.warning(f"Failed to download database: {e}")
            return None

    return None


def upload_database(
    db_path: str | Path,
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_DB_FILENAME,
    commit_message: str = "Update entity database",
    token: Optional[str] = None,
) -> str:
    """
    Upload entity database to HuggingFace Hub.

    Args:
        db_path: Local path to database file
        repo_id: HuggingFace repo ID
        filename: Target filename in repo
        commit_message: Git commit message
        token: HuggingFace API token (uses HF_TOKEN env var if not provided)

    Returns:
        URL of the uploaded file
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for database upload. "
            "Install with: pip install huggingface_hub"
        )

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HuggingFace token required. Set HF_TOKEN env var or pass token argument.")

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
            token=token,
        )
    except Exception as e:
        logger.debug(f"Repo creation note: {e}")

    # Upload file
    logger.info(f"Uploading database to {repo_id}...")

    result = api.upload_file(
        path_or_fileobj=str(db_path),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )

    logger.info("Database uploaded successfully")
    return result


def get_latest_version(repo_id: str = DEFAULT_REPO_ID) -> Optional[str]:
    """
    Get the latest version/commit of the database repo.

    Args:
        repo_id: HuggingFace repo ID

    Returns:
        Latest commit SHA or None if unavailable
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.repo_info(repo_id=repo_id, repo_type="dataset")
        return info.sha
    except Exception as e:
        logger.debug(f"Failed to get repo info: {e}")
        return None


def check_for_updates(
    repo_id: str = DEFAULT_REPO_ID,
    current_version: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """
    Check if a newer version of the database is available.

    Args:
        repo_id: HuggingFace repo ID
        current_version: Current cached version (commit SHA)

    Returns:
        Tuple of (update_available: bool, latest_version: str or None)
    """
    latest = get_latest_version(repo_id)

    if latest is None:
        return False, None

    if current_version is None:
        return True, latest

    return latest != current_version, latest


def vacuum_database(db_path: str | Path) -> None:
    """
    VACUUM the database to reclaim space and optimize it.

    Args:
        db_path: Path to the database file
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    original_size = db_path.stat().st_size
    logger.info(f"Running VACUUM on {db_path} ({original_size / (1024*1024):.1f}MB)")

    # Use isolation_level=None for autocommit (required for VACUUM)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute("VACUUM")
    finally:
        conn.close()

    new_size = db_path.stat().st_size
    reduction = (1 - new_size / original_size) * 100

    logger.info(f"After VACUUM: {new_size / (1024*1024):.1f}MB (reduced {reduction:.1f}%)")


def create_lite_database(
    source_db_path: str | Path,
    output_path: Optional[str | Path] = None,
) -> Path:
    """
    Create a lite version of the database without full records.

    The lite version strips the `record` column content (sets to empty {}),
    significantly reducing file size while keeping embeddings and core fields.

    Args:
        source_db_path: Path to the full database
        output_path: Output path for lite database (default: adds -lite suffix)

    Returns:
        Path to the lite database
    """
    source_db_path = Path(source_db_path)
    if not source_db_path.exists():
        raise FileNotFoundError(f"Source database not found: {source_db_path}")

    if output_path is None:
        output_path = source_db_path.with_stem(source_db_path.stem + "-lite")
    output_path = Path(output_path)

    logger.info(f"Creating lite database from {source_db_path}")
    logger.info(f"Output: {output_path}")

    # Copy the database first
    shutil.copy2(source_db_path, output_path)

    # Connect and strip record contents
    # Use isolation_level=None for autocommit (required for VACUUM)
    conn = sqlite3.connect(str(output_path), isolation_level=None)
    try:
        # Update all records to have empty record JSON
        conn.execute("BEGIN")
        cursor = conn.execute("UPDATE organizations SET record = '{}'")
        updated = cursor.rowcount
        logger.info(f"Stripped {updated} record fields")
        conn.execute("COMMIT")

        # Vacuum to reclaim space (must be outside transaction)
        conn.execute("VACUUM")
    finally:
        conn.close()

    # Log size reduction
    original_size = source_db_path.stat().st_size
    lite_size = output_path.stat().st_size
    reduction = (1 - lite_size / original_size) * 100

    logger.info(f"Original size: {original_size / (1024*1024):.1f}MB")
    logger.info(f"Lite size: {lite_size / (1024*1024):.1f}MB")
    logger.info(f"Size reduction: {reduction:.1f}%")

    return output_path


def upload_database_with_variants(
    db_path: str | Path,
    repo_id: str = DEFAULT_REPO_ID,
    commit_message: str = "Update entity database",
    token: Optional[str] = None,
    include_lite: bool = True,
    include_readme: bool = True,
) -> dict[str, str]:
    """
    Upload entity database with optional lite variant.

    First VACUUMs the database, then creates and uploads:
    - entities.db (full database)
    - entities-lite.db (without record data, smaller)
    - README.md (dataset card from HUGGINGFACE_README.md)

    Args:
        db_path: Local path to full database file
        repo_id: HuggingFace repo ID
        commit_message: Git commit message
        token: HuggingFace API token
        include_lite: Whether to create and upload lite version
        include_readme: Whether to upload the README.md dataset card

    Returns:
        Dict mapping filename to upload URL
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for database upload. "
            "Install with: pip install huggingface_hub"
        )

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HuggingFace token required. Set HF_TOKEN env var or pass token argument.")

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
            token=token,
        )
    except Exception as e:
        logger.debug(f"Repo creation note: {e}")

    # VACUUM the database first to optimize it
    vacuum_database(db_path)

    results = {}

    # Create temp directory for variants
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        files_to_upload = []

        # Full database
        files_to_upload.append((db_path, DEFAULT_DB_FULL_FILENAME))

        # Lite version
        if include_lite:
            lite_path = temp_path / DEFAULT_DB_LITE_FILENAME
            create_lite_database(db_path, lite_path)
            files_to_upload.append((lite_path, DEFAULT_DB_LITE_FILENAME))

        # Copy all files to a staging directory for upload_folder
        staging_dir = temp_path / "staging"
        staging_dir.mkdir()

        for local_path, remote_filename in files_to_upload:
            shutil.copy2(local_path, staging_dir / remote_filename)
            logger.info(f"Staged {remote_filename}")

        # Add README.md from HUGGINGFACE_README.md
        if include_readme:
            # Look for HUGGINGFACE_README.md in the package directory
            package_dir = Path(__file__).parent.parent.parent.parent  # Go up to statement-extractor-lib
            readme_source = package_dir / "HUGGINGFACE_README.md"
            if readme_source.exists():
                shutil.copy2(readme_source, staging_dir / "README.md")
                files_to_upload.append((readme_source, "README.md"))
                logger.info("Staged README.md from HUGGINGFACE_README.md")
            else:
                logger.warning(f"HUGGINGFACE_README.md not found at {readme_source}")

        # Upload all files in a single commit to avoid LFS pointer issues
        logger.info(f"Uploading {len(files_to_upload)} files to {repo_id}...")
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )

        for _, remote_filename in files_to_upload:
            results[remote_filename] = f"https://huggingface.co/datasets/{repo_id}/blob/main/{remote_filename}"
            logger.info(f"Uploaded {remote_filename}")

    return results


def download_database(
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_DB_FILENAME,
    revision: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Path:
    """
    Download entity database from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repo ID (e.g., "Corp-o-Rate-Community/entity-references")
        filename: Database filename in the repo
        revision: Git revision (branch, tag, commit) or None for latest
        cache_dir: Local cache directory
        force_download: Force re-download even if cached

    Returns:
        Path to the downloaded database file
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for database download. "
            "Install with: pip install huggingface_hub"
        )

    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading entity database from {repo_id}...")

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=str(cache_dir),
        force_download=force_download,
        repo_type="dataset",
    )

    logger.info(f"Database downloaded to {local_path}")
    return Path(local_path)
