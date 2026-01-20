"""
HuggingFace Hub integration for company database distribution.

Provides functionality to:
- Download pre-built company databases from HuggingFace Hub
- Upload/publish database updates
- Version management for database files
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default HuggingFace repo for company database
DEFAULT_REPO_ID = "Corp-o-Rate-Community/company-embeddings"
DEFAULT_DB_FILENAME = "companies.db"

# Local cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "corp-extractor"


def download_database(
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_DB_FILENAME,
    revision: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Path:
    """
    Download company database from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repo ID (e.g., "corp-o-rate/company-embeddings")
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

    logger.info(f"Downloading company database from {repo_id}...")

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


def get_database_path(
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_DB_FILENAME,
    auto_download: bool = True,
) -> Optional[Path]:
    """
    Get path to company database, downloading if necessary.

    Args:
        repo_id: HuggingFace repo ID
        filename: Database filename
        auto_download: Whether to download if not cached

    Returns:
        Path to database file, or None if not available
    """
    # Check if database exists in cache
    cache_dir = DEFAULT_CACHE_DIR

    # Check common locations
    possible_paths = [
        cache_dir / filename,
        cache_dir / "companies.db",
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
    commit_message: str = "Update company database",
    token: Optional[str] = None,
) -> str:
    """
    Upload company database to HuggingFace Hub.

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

    logger.info(f"Database uploaded successfully")
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
