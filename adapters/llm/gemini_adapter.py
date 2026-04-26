import base64
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any

from adapters.llm.base import BaseLLMAdapter, LLMAdapterError, TModel

logger = logging.getLogger(__name__)


class GeminiLLMAdapter(BaseLLMAdapter):
    def __init__(
        self,
        client: Any,
        model: str,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.5,
    ) -> None:
        super().__init__(max_retries=max_retries, retry_delay_seconds=retry_delay_seconds)
        self.client = client
        self.model = model

    async def _generate_structured(
        self,
        prompt: str,
        input_data: dict[str, Any],
        response_model: type[TModel],
    ) -> TModel:
        raw = await self._generate_content(
            [
                prompt,
                json.dumps(input_data, ensure_ascii=False),
                f"Return JSON matching this schema: {json.dumps(response_model.model_json_schema())}",
            ]
        )
        return self._validate_response(self._parse_json(raw), response_model)

    async def _analyze_image_structured(
        self,
        prompt: str,
        image_path: str,
        response_model: type[TModel],
    ) -> TModel:
        raw = await self._generate_content(
            [
                prompt,
                self._file_part(image_path),
                f"Return JSON matching this schema: {json.dumps(response_model.model_json_schema())}",
            ]
        )
        return self._validate_response(self._parse_json(raw), response_model)

    async def _analyze_pdf_structured(
        self,
        prompt: str,
        file_path: str,
        response_model: type[TModel],
    ) -> TModel:
        raw = await self._generate_content(
            [
                prompt,
                self._file_part(file_path, default_mime_type="application/pdf"),
                f"Return JSON matching this schema: {json.dumps(response_model.model_json_schema())}",
            ]
        )
        return self._validate_response(self._parse_json(raw), response_model)

    async def _extract_graph_triples(
        self,
        prompt: str,
        input_data: dict[str, Any],
        response_model: type[TModel],
    ) -> TModel:
        return await self._generate_structured(prompt, input_data, response_model)

    async def _generate_content(self, contents: list[Any]) -> str:
        try:
            model = self.client.models if hasattr(self.client, "models") else self.client
            response = await model.generate_content(model=self.model, contents=contents)
            text = getattr(response, "text", None)
            if text:
                return text
        except Exception as exc:
            logger.exception("Gemini LLM adapter call failed")
            raise LLMAdapterError("Gemini LLM adapter call failed") from exc
        raise LLMAdapterError("Gemini response did not include text content")

    def _file_part(
        self,
        file_path: str,
        default_mime_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        path = Path(file_path)
        mime_type = mimetypes.guess_type(path.name)[0] or default_mime_type
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"inline_data": {"mime_type": mime_type, "data": data}}

    def _parse_json(self, raw_text: str) -> dict[str, Any]:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)
