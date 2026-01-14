"""Pytest configuration and fixtures."""

import pytest


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture
def sample_source_text():
    """Sample source text for testing."""
    return """
    Apple Inc. announced the new iPhone 15 at their annual September event.
    Tim Cook, CEO of Apple, presented the new features to customers worldwide.
    The company also released the Apple Watch Series 9 and AirPods Pro 2.
    """


@pytest.fixture
def sample_statements():
    """Sample statements for testing."""
    from statement_extractor.models import Entity, EntityType, Statement

    return [
        Statement(
            subject=Entity(text="Apple Inc.", type=EntityType.ORG),
            predicate="announced",
            object=Entity(text="iPhone 15", type=EntityType.PRODUCT),
            source_text="Apple Inc. announced the new iPhone 15",
        ),
        Statement(
            subject=Entity(text="Tim Cook", type=EntityType.PERSON),
            predicate="presented",
            object=Entity(text="new features", type=EntityType.UNKNOWN),
            source_text="Tim Cook presented the new features",
        ),
        Statement(
            subject=Entity(text="company", type=EntityType.ORG),
            predicate="released",
            object=Entity(text="Apple Watch Series 9", type=EntityType.PRODUCT),
            source_text="The company also released the Apple Watch Series 9",
        ),
    ]
