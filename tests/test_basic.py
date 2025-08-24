"""Basic tests for agentsystems_toolkit that won't hang CI."""

import pytest


def test_import():
    """Test basic package import."""
    from agentsystems_toolkit import get_model

    assert callable(get_model)


def test_framework_validation():
    """Test framework validation without file I/O."""
    from agentsystems_toolkit import get_model

    with pytest.raises(ValueError, match="Framework 'invalid' not supported"):
        get_model("any-model", "invalid")


def test_supported_frameworks():
    """Test that langchain is in supported frameworks."""
    from agentsystems_toolkit.models.router import SUPPORTED_FRAMEWORKS

    assert "langchain" in SUPPORTED_FRAMEWORKS
    assert len(SUPPORTED_FRAMEWORKS) >= 1
