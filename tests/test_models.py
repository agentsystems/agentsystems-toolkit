"""Tests for model routing functionality."""

from unittest.mock import mock_open, patch

import pytest
import yaml

from agentsystems_toolkit.models.router import (
    _load_model_connection,
    get_model,
    validate_model_dependencies,
)


@pytest.fixture
def mock_config():
    """Mock agentsystems-config.yml content."""
    return {
        "config_version": 1,
        "model_connections": {
            "claude-sonnet-4": {
                "hosting_provider": "anthropic",
                "hosting_provider_model_id": "claude-sonnet-4-20250514",
                "enabled": True,
                "auth": {"method": "api_key", "api_key_env": "ANTHROPIC_API_KEY"},
            },
            "gpt-4o": {
                "hosting_provider": "openai",
                "hosting_provider_model_id": "gpt-4o",
                "enabled": True,
                "auth": {"method": "api_key", "api_key_env": "OPENAI_API_KEY"},
            },
            "disabled-model": {
                "hosting_provider": "anthropic",
                "enabled": False,
                "auth": {"method": "api_key", "api_key_env": "TEST_KEY"},
            },
        },
    }


class TestModelRouter:
    """Test the main model routing functionality."""

    def test_unsupported_framework(self):
        """Test error handling for unsupported frameworks."""
        with pytest.raises(ValueError, match="Framework 'unsupported' not supported"):
            get_model("claude-sonnet-4", "unsupported")

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    def test_config_file_not_found(self, mock_exists, mock_file):
        """Test error when config file doesn't exist."""
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError, match="AgentSystems config not found"):
            _load_model_connection("claude-sonnet-4")

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    def test_model_not_configured(self, mock_exists, mock_file, mock_config):
        """Test error when model not in config."""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = yaml.dump(mock_config)

        with pytest.raises(
            ValueError, match="No connection configured for model 'missing-model'"
        ):
            _load_model_connection("missing-model")

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    def test_disabled_model(self, mock_exists, mock_file, mock_config):
        """Test error when model connection is disabled."""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = yaml.dump(mock_config)

        with pytest.raises(
            ValueError, match="Model connection 'disabled-model' is disabled"
        ):
            _load_model_connection("disabled-model")

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    def test_validate_model_dependencies(self, mock_exists, mock_file, mock_config):
        """Test model dependency validation."""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = yaml.dump(mock_config)

        result = validate_model_dependencies(
            ["claude-sonnet-4", "gpt-4o", "missing-model"]
        )

        assert result == {
            "claude-sonnet-4": True,
            "gpt-4o": True,
            "missing-model": False,
        }


class TestLangChainModels:
    """Test LangChain model creation."""

    @patch("agentsystems_toolkit.models.langchain._create_anthropic_model")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    def test_anthropic_model_creation(
        self, mock_exists, mock_file, mock_anthropic, mock_config
    ):
        """Test creating Anthropic model via LangChain."""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = yaml.dump(mock_config)
        mock_anthropic.return_value = "mocked_anthropic_model"

        result = get_model("claude-sonnet-4", "langchain")

        mock_anthropic.assert_called_once()
        assert result == "mocked_anthropic_model"
