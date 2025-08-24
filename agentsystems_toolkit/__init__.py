"""AgentSystems Toolkit - Development utilities for building agents."""

from importlib import metadata as _metadata

__version__ = (
    _metadata.version(__name__.replace("_", "-")) if __name__ != "__main__" else "0.0.0"
)

# Main API exports
from .models import getModel

__all__ = ["__version__", "getModel"]