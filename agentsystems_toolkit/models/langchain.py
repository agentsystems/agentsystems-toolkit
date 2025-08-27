"""LangChain model implementations for AgentSystems model routing.

This module handles the instantiation of LangChain model objects based on
user-configured model connections, supporting multiple hosting providers
for each model type.
"""

import os
from typing import Any


def get_langchain_model(
    model_id: str, connection: dict[str, Any], **kwargs: Any
) -> Any:
    """Create a LangChain model instance from connection configuration.

    Args:
        model_id: Abstract model identifier (e.g., 'claude-sonnet-4')
        connection: Model connection configuration from agentsystems-config.yml
        **kwargs: Additional arguments passed to the LangChain model constructor

    Returns:
        LangChain model instance (ChatAnthropic, ChatOpenAI, ChatBedrock, etc.)

    Raises:
        ValueError: If hosting provider not supported or credentials missing
        ImportError: If required LangChain package not installed
    """
    hosting_provider = connection["hosting_provider"]
    hosting_provider_model_id = connection["hosting_provider_model_id"]
    auth = connection.get("auth", {})

    # Route to hosting provider-specific implementation
    if hosting_provider == "anthropic":
        return _create_anthropic_model(hosting_provider_model_id, auth, **kwargs)

    elif hosting_provider == "amazon_bedrock":
        return _create_bedrock_model(hosting_provider_model_id, auth, **kwargs)

    elif hosting_provider == "openai":
        return _create_openai_model(hosting_provider_model_id, auth, **kwargs)

    elif hosting_provider == "ollama":
        return _create_ollama_model(hosting_provider_model_id, auth, **kwargs)

    else:
        providers = "anthropic, amazon_bedrock, openai, ollama"
        raise ValueError(
            f"Hosting provider '{hosting_provider}' not supported for LangChain. "
            f"Supported providers: {providers}"
        )


def _create_anthropic_model(
    hosting_provider_model_id: str, auth: dict[str, Any], **kwargs: Any
) -> Any:
    """Create ChatAnthropic instance."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "langchain-anthropic not installed. "
            "Install with: pip install langchain-anthropic"
        )

    api_key_env = auth.get("api_key_env")
    if not api_key_env:
        raise ValueError("Anthropic connection missing api_key_env")

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Environment variable '{api_key_env}' not set")

    return ChatAnthropic(model=hosting_provider_model_id, api_key=api_key, **kwargs)


def _create_bedrock_model(
    hosting_provider_model_id: str, auth: dict[str, Any], **kwargs: Any
) -> Any:
    """Create ChatBedrock instance."""
    try:
        from langchain_aws import ChatBedrock
    except ImportError:
        raise ImportError(
            "langchain-aws not installed. " "Install with: pip install langchain-aws"
        )

    # Extract AWS credentials from auth config
    access_key_env = auth.get("aws_access_key_env")
    secret_key_env = auth.get("aws_secret_key_env")
    region_env = auth.get("aws_region")
    region_prefix = auth.get("region_prefix")

    if not access_key_env or not secret_key_env or not region_env:
        msg = (
            "AWS Bedrock connection missing aws_access_key_env, "
            "aws_secret_key_env, or aws_region"
        )
        raise ValueError(msg)

    access_key = os.getenv(access_key_env)
    secret_key = os.getenv(secret_key_env)
    region = os.getenv(region_env)

    if not access_key or not secret_key or not region:
        missing = []
        if not access_key:
            missing.append(access_key_env)
        if not secret_key:
            missing.append(secret_key_env)
        if not region:
            missing.append(region_env)
        raise ValueError(f"Environment variables not set: {', '.join(missing)}")

    # Apply region prefix to model ID if specified
    final_model_id = hosting_provider_model_id
    if region_prefix:
        final_model_id = f"{region_prefix}.{hosting_provider_model_id}"

    return ChatBedrock(
        model_id=final_model_id,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        **kwargs,
    )


def _create_openai_model(
    hosting_provider_model_id: str, auth: dict[str, Any], **kwargs: Any
) -> Any:
    """Create ChatOpenAI instance."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai not installed. "
            "Install with: pip install langchain-openai"
        )

    api_key_env = auth.get("api_key_env")
    if not api_key_env:
        raise ValueError("OpenAI connection missing api_key_env")

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Environment variable '{api_key_env}' not set")

    return ChatOpenAI(model=hosting_provider_model_id, api_key=api_key, **kwargs)


def _create_ollama_model(
    hosting_provider_model_id: str, auth: dict[str, Any], **kwargs: Any
) -> Any:
    """Create ChatOllama instance with optional authentication."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError(
            "langchain-ollama not installed. "
            "Install with: pip install langchain-ollama"
        )

    base_url = auth.get("base_url", "http://localhost:11434")
    api_key_env = auth.get("api_key_env")

    # Build client kwargs for optional authentication
    client_kwargs = {}
    if api_key_env:
        api_key = os.getenv(api_key_env)
        if api_key:
            client_kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}

    return ChatOllama(
        model=hosting_provider_model_id,
        base_url=base_url,
        client_kwargs=client_kwargs if client_kwargs else None,
        **kwargs,
    )
