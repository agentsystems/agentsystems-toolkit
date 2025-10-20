# AgentSystems Toolkit

[![GitHub stars](https://img.shields.io/github/stars/agentsystems/agentsystems?style=flat-square&logo=github)](https://github.com/agentsystems/agentsystems/stargazers)

> [!NOTE]
> **Pre-Release Software** - AgentSystems is in active development. Join our [Discord](https://discord.com/invite/JsxDxQ5zfV) for updates and early access.
> ⭐ [**Star the main repository**](https://github.com/agentsystems/agentsystems) to show your support!

> This is the **development toolkit** for AgentSystems. See the [main repository](https://github.com/agentsystems/agentsystems) for platform overview and documentation.

Development toolkit for building AgentSystems agents with model routing, progress tracking, and agent framework integration.

## Features

- **Model Routing**: Abstract model layer supporting OpenAI, Anthropic, AWS Bedrock, Ollama
- **Progress Tracking**: Progress reporting utilities for long-running tasks
- **LangChain Integration**: Pre-configured providers with credential management

## Installation

```bash
pip install agentsystems-toolkit
```

## Quick Start

```python
from agentsystems_toolkit import get_model

# Get a model - automatically routes to configured provider
model = get_model("claude-sonnet-4", "langchain", temperature=0)

# Use with any LangChain workflow
response = model.invoke("Hello, world!")
```

## License

Licensed under the [Apache-2.0 license](./LICENSE).
