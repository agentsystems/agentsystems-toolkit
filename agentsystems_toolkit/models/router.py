"""Model routing implementation for AgentSystems agents.

This module provides the main getModel() function that routes abstract model IDs
(like 'claude-sonnet-4') to user-configured hosting providers while returning
framework-native objects (LangChain, OpenAI client, etc.).
"""

import os
from pathlib import Path
from typing import Any

import yaml

# Framework-specific implementations
from .langchain import get_langchain_model

# Supported frameworks
SUPPORTED_FRAMEWORKS = ["langchain"]


def get_model(model_id: str, framework: str = "langchain", **kwargs: Any) -> Any:
    """Get a configured model instance for the specified framework.

    Args:
        model_id: Abstract model identifier (e.g., 'claude-sonnet-4', 'gpt-4o')
        framework: Target framework ('langchain', 'openai', etc.)
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        Framework-native model instance (e.g., ChatAnthropic for LangChain)

    Raises:
        ValueError: If framework not supported or model not configured
        FileNotFoundError: If agentsystems-config.yml not found

    Example:
        >>> llm = getModel("claude-sonnet-4", "langchain")
        >>> chain = prompt | llm | parser
    """
    if framework not in SUPPORTED_FRAMEWORKS:
        raise ValueError(
            f"Framework '{framework}' not supported. "
            f"Supported frameworks: {', '.join(SUPPORTED_FRAMEWORKS)}"
        )

    # Load model connection configuration
    connection = _load_model_connection(model_id)

    # Route to framework-specific implementation
    if framework == "langchain":
        return get_langchain_model(model_id, connection, **kwargs)

    # Future frameworks would be added here
    raise ValueError(f"Framework '{framework}' not yet implemented")


def _load_model_connection(model_id: str) -> dict[str, Any]:
    """Load model connection configuration from agentsystems-config.yml.

    Args:
        model_id: Model identifier to look up

    Returns:
        Model connection configuration dictionary

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If model connection not configured
    """
    # Try to find config file in standard locations
    config_path = os.getenv(
        "AGENTSYSTEMS_CONFIG_PATH", "/etc/agentsystems/agentsystems-config.yml"
    )

    if not Path(config_path).exists():
        # Fallback to local development path
        config_path = "./agentsystems-config.yml"

    if not Path(config_path).exists():
        raise FileNotFoundError(
            f"AgentSystems config not found at {config_path}. "
            "Ensure the config file is mounted or AGENTSYSTEMS_CONFIG_PATH is set."
        )

    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}")

    # Extract model connections
    model_connections = config.get("model_connections", {})

    if model_id not in model_connections:
        available = ", ".join(model_connections.keys()) if model_connections else "none"
        raise ValueError(
            f"No connection configured for model '{model_id}'. "
            f"Available models: {available}. "
            "Configure model connections via the AgentSystems UI."
        )

    connection = model_connections[model_id]

    # Validate connection is enabled
    if not connection.get("enabled", True):
        raise ValueError(f"Model connection '{model_id}' is disabled")

    return connection  # type: ignore[no-any-return]


def list_available_models() -> dict[str, dict[str, Any]]:
    """List all configured model connections.

    Returns:
        Dictionary mapping model IDs to their connection info

    Example:
        >>> models = list_available_models()
        >>> print(models.keys())
        ['claude-sonnet-4', 'gpt-4o']
    """
    try:
        # Load all model connections
        config_path = os.getenv(
            "AGENTSYSTEMS_CONFIG_PATH", "/etc/agentsystems/agentsystems-config.yml"
        )

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config.get("model_connections", {})  # type: ignore[no-any-return]

    except (FileNotFoundError, yaml.YAMLError):
        return {}


def validate_model_dependencies(required_models: list[str]) -> dict[str, bool]:
    """Validate that all required models are configured.

    Args:
        required_models: List of model IDs this agent requires

    Returns:
        Dictionary mapping model IDs to availability status

    Example:
        >>> status = validate_model_dependencies(['claude-sonnet-4', 'gpt-4o'])
        >>> print(status)
        {'claude-sonnet-4': True, 'gpt-4o': False}
    """
    available = list_available_models()

    return {
        model_id: (model_id in available and available[model_id].get("enabled", True))
        for model_id in required_models
    }
