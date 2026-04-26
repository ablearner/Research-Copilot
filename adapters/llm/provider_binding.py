from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from core.config import Settings


class ProviderBindingError(RuntimeError):
    """Raised when a provider-backed LangChain model cannot be configured."""


@dataclass(slots=True)
class ProviderBinding:
    provider: str
    chat_model: BaseChatModel | None = None
    vision_model: BaseChatModel | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 90.0
    max_retries: int = 2

    def require_chat_model(self) -> BaseChatModel:
        if self.chat_model is None:
            raise ProviderBindingError(f"Provider '{self.provider}' does not expose a chat model")
        return self.chat_model

    def require_vision_model(self) -> BaseChatModel:
        if self.vision_model is not None:
            return self.vision_model
        return self.require_chat_model()


def file_to_data_uri(file_path: str, default_mime_type: str = "application/octet-stream") -> str:
    path = Path(file_path)
    mime_type = mimetypes.guess_type(path.name)[0] or default_mime_type
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def normalize_openai_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return base_url
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    for suffix in ("/chat/completions", "/responses"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def build_provider_binding(settings: Settings) -> ProviderBinding:
    provider = settings.llm_provider.lower()
    if provider == "local":
        return ProviderBinding(provider="local")
    if provider == "dashscope":
        if not settings.dashscope_api_key:
            raise ProviderBindingError("DASHSCOPE_API_KEY is required when LLM_PROVIDER=dashscope")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ProviderBindingError("langchain-openai is required for DashScope provider binding") from exc
        chat_model = ChatOpenAI(
            api_key=settings.dashscope_api_key,
            model=settings.llm_model,
            base_url=settings.dashscope_base_url,
            temperature=0.1,
            timeout=settings.dashscope_timeout_seconds,
            max_retries=settings.dashscope_max_retries,
        )
        vision_model = ChatOpenAI(
            api_key=settings.dashscope_api_key,
            model=settings.vision_model or settings.llm_model,
            base_url=settings.dashscope_base_url,
            temperature=0.1,
            timeout=settings.dashscope_timeout_seconds,
            max_retries=settings.dashscope_max_retries,
        )
        return ProviderBinding(
            provider="dashscope",
            chat_model=chat_model,
            vision_model=vision_model,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            timeout_seconds=settings.dashscope_timeout_seconds,
            max_retries=settings.dashscope_max_retries,
        )
    if provider == "openai":
        if not settings.openai_api_key:
            raise ProviderBindingError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ProviderBindingError("langchain-openai is required for OpenAI provider binding") from exc
        common_kwargs: dict[str, Any] = {
            "api_key": settings.openai_api_key,
            "temperature": 0.1,
            "timeout": settings.openai_timeout_seconds,
            "max_retries": settings.openai_max_retries,
        }
        if settings.openai_base_url:
            common_kwargs["base_url"] = normalize_openai_base_url(settings.openai_base_url)

        chat_model = ChatOpenAI(model=settings.llm_model, **common_kwargs)
        vision_model = ChatOpenAI(model=settings.vision_model or settings.llm_model, **common_kwargs)
        return ProviderBinding(
            provider="openai",
            chat_model=chat_model,
            vision_model=vision_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )
    if provider == "gemini":
        if not settings.google_api_key:
            raise ProviderBindingError("GOOGLE_API_KEY is required when LLM_PROVIDER=gemini")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ProviderBindingError(
                "langchain-google-genai is required for Gemini provider binding"
            ) from exc
        chat_model = ChatGoogleGenerativeAI(
            google_api_key=settings.google_api_key,
            model=settings.llm_model,
            temperature=0.1,
        )
        vision_model = ChatGoogleGenerativeAI(
            google_api_key=settings.google_api_key,
            model=settings.vision_model or settings.llm_model,
            temperature=0.1,
        )
        return ProviderBinding(
            provider="gemini",
            chat_model=chat_model,
            vision_model=vision_model,
            api_key=settings.google_api_key,
        )
    raise ProviderBindingError(f"Unsupported LLM provider binding: {settings.llm_provider}")


def supports_structured_output(model: BaseChatModel) -> bool:
    return callable(getattr(model, "with_structured_output", None))


def as_runnable_structured_output(
    *,
    model: BaseChatModel,
    schema: type,
    include_raw: bool = False,
) -> Any:
    structured_builder = getattr(model, "with_structured_output", None)
    if not callable(structured_builder):
        raise ProviderBindingError(
            f"Model '{model.__class__.__name__}' does not support with_structured_output"
        )
    return structured_builder(schema, include_raw=include_raw)
