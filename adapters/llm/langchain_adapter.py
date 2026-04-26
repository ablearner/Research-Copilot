from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from adapters.llm.base import BaseLLMAdapter, TModel
from adapters.llm.provider_binding import (
    ProviderBinding,
    as_runnable_structured_output,
    file_to_data_uri,
)


class LangChainLLMAdapter(BaseLLMAdapter):
    """Compatibility adapter backed by LangChain chat models."""

    def __init__(
        self,
        provider_binding: ProviderBinding,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.5,
    ) -> None:
        super().__init__(max_retries=max_retries, retry_delay_seconds=retry_delay_seconds)
        self.provider_binding = provider_binding
        self.model = getattr(provider_binding.chat_model, "model_name", provider_binding.provider)
        self.vision_model = getattr(provider_binding.vision_model, "model_name", self.model)

    async def _generate_structured(
        self,
        prompt: str,
        input_data: dict[str, Any],
        response_model: type[TModel],
    ) -> TModel:
        runnable = as_runnable_structured_output(
            model=self.provider_binding.require_chat_model(),
            schema=response_model,
        )
        return await runnable.ainvoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content=(
                        "Return valid JSON that matches the requested structured output schema.\n\n"
                        f"{json.dumps(input_data, ensure_ascii=False, indent=2)}"
                    )
                ),
            ]
        )

    async def _analyze_image_structured(
        self,
        prompt: str,
        image_path: str,
        response_model: type[TModel],
    ) -> TModel:
        runnable = as_runnable_structured_output(
            model=self.provider_binding.require_vision_model(),
            schema=response_model,
        )
        return await runnable.ainvoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Analyze the image and return the requested schema."},
                        {
                            "type": "image_url",
                            "image_url": {"url": file_to_data_uri(image_path, "image/png")},
                        },
                    ]
                ),
            ]
        )

    async def _analyze_pdf_structured(
        self,
        prompt: str,
        file_path: str,
        response_model: type[TModel],
    ) -> TModel:
        runnable = as_runnable_structured_output(
            model=self.provider_binding.require_vision_model(),
            schema=response_model,
        )
        return await runnable.ainvoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Analyze the PDF and return the requested schema."},
                        {
                            "type": "file",
                            "source_type": "base64",
                            "mime_type": "application/pdf",
                            "data": file_to_data_uri(file_path, "application/pdf").split(",", 1)[1],
                        },
                    ]
                ),
            ]
        )

    async def _extract_graph_triples(
        self,
        prompt: str,
        input_data: dict[str, Any],
        response_model: type[TModel],
    ) -> TModel:
        return await self._generate_structured(prompt, input_data, response_model)
