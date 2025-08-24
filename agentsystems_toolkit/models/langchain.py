"""LangChain model implementations for AgentSystems model routing.

This module handles the instantiation of LangChain model objects based on
user-configured model connections, supporting multiple hosting providers
for each model type.
"""

import os
from typing import Any, Dict, Optional

def get_langchain_model(model_id: str, connection: Dict[str, Any], **kwargs) -> Any:
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
    hosting_provider = connection['hosting_provider']
    provider_model_id = connection['hosting_provider_model_id']
    auth = connection.get('auth', {})
    
    # Route to hosting provider-specific implementation
    if hosting_provider == 'anthropic':
        return _create_anthropic_model(provider_model_id, auth, **kwargs)
        
    elif hosting_provider == 'aws_bedrock':
        return _create_bedrock_model(provider_model_id, auth, **kwargs)
        
    elif hosting_provider == 'gcp_vertex':
        return _create_vertex_model(provider_model_id, auth, **kwargs)
        
    elif hosting_provider == 'openai':
        return _create_openai_model(provider_model_id, auth, **kwargs)
        
    elif hosting_provider == 'azure_openai':
        return _create_azure_openai_model(provider_model_id, auth, connection.get('endpoint'), **kwargs)
        
    else:
        raise ValueError(
            f"Hosting provider '{hosting_provider}' not supported for LangChain. "
            f"Supported providers: anthropic, aws_bedrock, gcp_vertex, openai, azure_openai"
        )


def _create_anthropic_model(model_id: str, auth: Dict[str, Any], **kwargs) -> Any:
    """Create ChatAnthropic instance."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "langchain-anthropic not installed. "
            "Install with: pip install langchain-anthropic"
        )
    
    api_key_env = auth.get('api_key_env')
    if not api_key_env:
        raise ValueError("Anthropic connection missing api_key_env")
        
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Environment variable '{api_key_env}' not set")
    
    return ChatAnthropic(
        model=model_id,
        api_key=api_key,
        **kwargs
    )


def _create_bedrock_model(model_id: str, auth: Dict[str, Any], **kwargs) -> Any:
    """Create ChatBedrock instance."""
    try:
        from langchain_aws import ChatBedrock
    except ImportError:
        raise ImportError(
            "langchain-aws not installed. "
            "Install with: pip install langchain-aws"
        )
    
    # Extract AWS credentials from auth config
    access_key_env = auth.get('aws_access_key_env')
    secret_key_env = auth.get('aws_secret_key_env')
    region = auth.get('aws_region', 'us-east-1')
    
    if not access_key_env or not secret_key_env:
        raise ValueError("AWS Bedrock connection missing aws_access_key_env or aws_secret_key_env")
    
    access_key = os.getenv(access_key_env)
    secret_key = os.getenv(secret_key_env)
    
    if not access_key or not secret_key:
        missing = []
        if not access_key: missing.append(access_key_env)
        if not secret_key: missing.append(secret_key_env)
        raise ValueError(f"Environment variables not set: {', '.join(missing)}")
    
    return ChatBedrock(
        model_id=model_id,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        **kwargs
    )


def _create_vertex_model(model_id: str, auth: Dict[str, Any], **kwargs) -> Any:
    """Create ChatVertexAI instance."""
    try:
        from langchain_google_vertexai import ChatVertexAI
    except ImportError:
        raise ImportError(
            "langchain-google-vertexai not installed. "
            "Install with: pip install langchain-google-vertexai"
        )
    
    # Extract GCP credentials from auth config
    service_account_env = auth.get('gcp_service_account_key_env')
    project_id = auth.get('gcp_project_id')
    region = auth.get('gcp_region', 'us-central1')
    
    if not service_account_env or not project_id:
        raise ValueError("GCP Vertex connection missing gcp_service_account_key_env or gcp_project_id")
    
    service_account_key = os.getenv(service_account_env)
    if not service_account_key:
        raise ValueError(f"Environment variable '{service_account_env}' not set")
    
    return ChatVertexAI(
        model_name=model_id,
        project=project_id,
        location=region,
        credentials=service_account_key,  # May need JSON parsing
        **kwargs
    )


def _create_openai_model(model_id: str, auth: Dict[str, Any], **kwargs) -> Any:
    """Create ChatOpenAI instance."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai not installed. "
            "Install with: pip install langchain-openai"
        )
    
    api_key_env = auth.get('api_key_env')
    if not api_key_env:
        raise ValueError("OpenAI connection missing api_key_env")
        
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Environment variable '{api_key_env}' not set")
    
    return ChatOpenAI(
        model=model_id,
        api_key=api_key,
        **kwargs
    )


def _create_azure_openai_model(model_id: str, auth: Dict[str, Any], endpoint: Optional[str], **kwargs) -> Any:
    """Create AzureChatOpenAI instance."""
    try:
        from langchain_openai import AzureChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai not installed. "
            "Install with: pip install langchain-openai"
        )
    
    api_key_env = auth.get('api_key_env')
    if not api_key_env:
        raise ValueError("Azure OpenAI connection missing api_key_env")
        
    if not endpoint:
        raise ValueError("Azure OpenAI connection missing endpoint")
        
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Environment variable '{api_key_env}' not set")
    
    return AzureChatOpenAI(
        model=model_id,
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version="2024-02-01",  # Could be configurable
        **kwargs
    )