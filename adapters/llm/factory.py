from __future__ import annotations

from adapters.llm.base import BaseLLMAdapter
from adapters.llm.langchain_binding import LangChainProviderBinding


def build_provider_binding(adapter: BaseLLMAdapter) -> LangChainProviderBinding:
    return LangChainProviderBinding(adapter=adapter, model_name=getattr(adapter, "model", None))
