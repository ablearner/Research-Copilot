try:  # pragma: no cover - optional dependency guard
    from adapters.llm.provider_binding import ProviderBinding, ProviderBindingError, build_provider_binding
except Exception:  # noqa: BLE001 - keep package importable without optional deps
    ProviderBinding = None  # type: ignore[assignment]
    ProviderBindingError = RuntimeError  # type: ignore[assignment]

    def build_provider_binding(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("Provider binding requires optional LLM dependencies.")


try:  # pragma: no cover - optional dependency guard
    from adapters.llm.langchain_adapter import LangChainLLMAdapter
except Exception:  # noqa: BLE001 - keep package importable without optional deps
    LangChainLLMAdapter = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    from adapters.llm.openai_relay_adapter import OpenAIRelayAdapter
except Exception:  # noqa: BLE001 - keep package importable without optional deps
    OpenAIRelayAdapter = None  # type: ignore[assignment]

from adapters.llm.error_classifier import ClassifiedError, FailureReason, classify_llm_error
from adapters.llm.fallback_adapter import FallbackLLMAdapter

__all__ = [
    "ClassifiedError",
    "FailureReason",
    "FallbackLLMAdapter",
    "LangChainLLMAdapter",
    "OpenAIRelayAdapter",
    "ProviderBinding",
    "ProviderBindingError",
    "build_provider_binding",
    "classify_llm_error",
]
