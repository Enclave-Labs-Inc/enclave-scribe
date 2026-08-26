"""EnclaveScribe agentic OCR pipeline.

Public API:
    from scribe.agent import Agent, ModelRegistry

Usage:
    agent = Agent()
    result = agent.parse("path/to/doc.pdf")
    print(result.markdown)
"""
from .models import ModelRegistry
from .orchestrator import Agent, DocumentResult, PageResult

__all__ = ["Agent", "ModelRegistry", "DocumentResult", "PageResult"]
