from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field

from adapters.llm.base import BaseLLMAdapter


def _normalize_content(content: Any) -> Any:
    if isinstance(content, list):
        return [_normalize_content(item) for item in content]
    if isinstance(content, dict):
        return {key: _normalize_content(value) for key, value in content.items()}
    return content


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "input_text"}:
                    chunks.append(str(item.get("text", "")))
                elif item.get("type") in {"image_url", "input_image"}:
                    chunks.append("[image]")
                elif item.get("type") in {"file", "input_file"}:
                    chunks.append("[file]")
                else:
                    chunks.append(json.dumps(item, ensure_ascii=False))
            else:
                chunks.append(str(item))
        return "\n".join(part for part in chunks if part)
    return str(content)


class AdapterChatModel(BaseChatModel):
    """Minimal LangChain chat model backed by the project's provider binding."""

    adapter: BaseLLMAdapter = Field(exclude=True)
    model_name: str = "adapter-chat-model"

    @property
    def _llm_type(self) -> str:
        return "provider-binding-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("Use async generation for this runtime.")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not hasattr(self.adapter, "generate_text"):
            prompt = "\n".join(f"{message.type}: {_stringify_content(message.content)}" for message in messages)
            message = AIMessage(content=prompt)
            return ChatResult(generations=[ChatGeneration(message=message)])

        content = await self.adapter.generate_text(messages=messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


class LangChainProviderBinding:
    """Provider binding exposed to the business layer as LangChain-first primitives."""

    def __init__(self, adapter: BaseLLMAdapter, model_name: str | None = None) -> None:
        self.adapter = adapter
        self.chat_model = AdapterChatModel(
            adapter=adapter,
            model_name=model_name or getattr(adapter, "model", adapter.__class__.__name__),
        )

    async def ainvoke_structured(
        self,
        *,
        messages: list[BaseMessage],
        response_model: type[BaseModel],
        metadata: dict[str, Any] | None = None,
    ) -> BaseModel:
        system_prompt, payload = self._split_messages(messages)
        return await self.adapter.generate_structured(
            prompt=system_prompt,
            input_data=self._to_input_data(payload, metadata),
            response_model=response_model,
        )

    async def ainvoke_image_structured(
        self,
        *,
        messages: list[BaseMessage],
        image_path: str,
        response_model: type[BaseModel],
        metadata: dict[str, Any] | None = None,
    ) -> BaseModel:
        system_prompt, payload = self._split_messages(messages)
        prompt = system_prompt
        if payload:
            prompt = f"{system_prompt}\n\nAdditional user context:\n{json.dumps(payload, ensure_ascii=False)}"
        return await self.adapter.analyze_image_structured(
            prompt=prompt,
            image_path=image_path,
            response_model=response_model,
        )

    async def ainvoke_pdf_structured(
        self,
        *,
        messages: list[BaseMessage],
        file_path: str,
        response_model: type[BaseModel],
        metadata: dict[str, Any] | None = None,
    ) -> BaseModel:
        system_prompt, payload = self._split_messages(messages)
        prompt = system_prompt
        if payload:
            prompt = f"{system_prompt}\n\nAdditional user context:\n{json.dumps(payload, ensure_ascii=False)}"
        return await self.adapter.analyze_pdf_structured(
            prompt=prompt,
            file_path=file_path,
            response_model=response_model,
        )

    async def ainvoke_graph_structured(
        self,
        *,
        messages: list[BaseMessage],
        response_model: type[BaseModel],
        metadata: dict[str, Any] | None = None,
    ) -> BaseModel:
        system_prompt, payload = self._split_messages(messages)
        return await self.adapter.extract_graph_triples(
            prompt=system_prompt,
            input_data=self._to_input_data(payload, metadata),
            response_model=response_model,
        )

    def _split_messages(self, messages: list[BaseMessage]) -> tuple[str, list[dict[str, Any]]]:
        if not messages:
            return "", []
        system_chunks: list[str] = []
        payload: list[dict[str, Any]] = []
        for message in messages:
            normalized_content = _normalize_content(message.content)
            content = _stringify_content(normalized_content)
            if message.type == "system":
                system_chunks.append(content)
            else:
                payload.append({"role": message.type, "content": normalized_content})
        return "\n\n".join(chunk for chunk in system_chunks if chunk), payload

    def _to_input_data(
        self,
        payload: list[dict[str, Any]],
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        effective_metadata = dict(metadata or {})
        raw_input_data = effective_metadata.pop("_input_data_override", None)
        if isinstance(raw_input_data, dict):
            return {**raw_input_data, "metadata": effective_metadata}
        if len(payload) == 1 and isinstance(payload[0].get("content"), dict):
            return {**payload[0]["content"], "metadata": effective_metadata}
        return {"messages": payload, "metadata": effective_metadata}


def ensure_provider_binding(binding_or_adapter: Any) -> LangChainProviderBinding:
    if isinstance(binding_or_adapter, LangChainProviderBinding):
        return binding_or_adapter
    if isinstance(binding_or_adapter, BaseLLMAdapter):
        return LangChainProviderBinding(binding_or_adapter)
    raise TypeError(f"Unsupported LLM binding type: {type(binding_or_adapter)!r}")


def file_to_data_uri(file_path: str, default_mime_type: str = "application/octet-stream") -> str:
    import base64
    import mimetypes

    path = Path(file_path)
    mime_type = mimetypes.guess_type(path.name)[0] or default_mime_type
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
