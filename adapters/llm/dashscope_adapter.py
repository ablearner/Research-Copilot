import base64
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any

import httpx

from adapters.llm.base import BaseLLMAdapter, LLMAdapterError, TModel

logger = logging.getLogger(__name__)


class DashScopeLLMAdapter(BaseLLMAdapter):
    _REQUIRED_FALLBACKS = {
        "id": "chart",
        "document_id": "document",
        "page_id": "page",
        "page_number": 1,
        "chart_type": "unknown",
    }

    def __init__(
        self,
        api_key: str,
        model: str = "qwen-plus",
        vision_model: str | None = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout_seconds: float = 90.0,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.5,
    ) -> None:
        super().__init__(max_retries=max_retries, retry_delay_seconds=retry_delay_seconds)
        self.api_key = api_key
        self.model = model
        self.vision_model = vision_model or model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def _generate_structured(
        self,
        prompt: str,
        input_data: dict[str, Any],
        response_model: type[TModel],
    ) -> TModel:
        schema = response_model.model_json_schema()
        content = (
            f"{json.dumps(input_data, ensure_ascii=False)}\n\n"
            "Return only valid JSON that matches this JSON Schema:\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )
        payload = await self._chat_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            json_mode=False,
        )
        return self._validate_response(self._extract_structured_payload(payload, response_model), response_model)

    async def _analyze_image_structured(
        self,
        prompt: str,
        image_path: str,
        response_model: type[TModel],
    ) -> TModel:
        output_contract = self._image_output_contract(response_model)
        payload = await self._chat_completion(
            model=self.vision_model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze the image itself. Return one JSON object for the observed chart, "
                                "not a JSON Schema and not markdown. Use this output contract and preserve "
                                "the caller-provided ids when present:\n"
                                f"{json.dumps(output_contract, ensure_ascii=False, indent=2)}"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": self._to_data_uri(image_path)}},
                    ],
                },
            ],
            json_mode=False,
        )
        return self._validate_response(self._extract_structured_payload(payload, response_model), response_model)

    def _image_output_contract(self, response_model: type[TModel]) -> dict[str, Any]:
        if getattr(response_model, "__name__", "") != "ChartSchema":
            return {"instruction": "Return one valid JSON object matching the requested response model."}
        return {
            "type": "ChartSchema instance, not JSON Schema",
            "required": {
                "id": "Use caller chart_id if provided, otherwise a short chart id string.",
                "document_id": "Use caller document_id if provided.",
                "page_id": "Use caller page_id if provided.",
                "page_number": "Use caller page_number if provided, integer >= 1.",
                "chart_type": "One of: bar, line, scatter, pie, table, mixed, unknown.",
            },
            "optional": {
                "title": "Visible title string or null.",
                "caption": "Visible caption string or null.",
                "bbox": "Bounding box object or null.",
                "x_axis": "Axis object or null.",
                "y_axis": "Axis object or null.",
                "series": "Array of series objects; use [] if unreadable.",
                "summary": "Concise visual summary based only on the image.",
                "confidence": "Number between 0 and 1.",
                "metadata": "Small object for notes; use {} if none.",
            },
            "example": {
                "id": "chart_1",
                "document_id": "doc_1",
                "page_id": "page_1",
                "page_number": 1,
                "chart_type": "mixed",
                "title": None,
                "caption": None,
                "bbox": None,
                "x_axis": {"name": None, "label": None, "unit": None, "scale": "unknown", "min_value": None, "max_value": None, "categories": []},
                "y_axis": {"name": None, "label": None, "unit": None, "scale": "unknown", "min_value": None, "max_value": None, "categories": []},
                "series": [],
                "summary": "The image contains a chart; describe only visible trends and labels.",
                "confidence": 0.6,
                "metadata": {},
            },
            "hard_rules": [
                "Do not return keys such as $defs, properties, required, or type from a JSON Schema.",
                "Do not invent exact data points when they are not readable.",
                "Use null or [] for unreadable fields.",
            ],
        }

    async def _analyze_pdf_structured(
        self,
        prompt: str,
        file_path: str,
        response_model: type[TModel],
    ) -> TModel:
        # DashScope OpenAI-compatible chat is used here for text/vision payloads. PDF
        # parsing should normally happen before the LLM call in DocumentAgent.
        raise LLMAdapterError("DashScope PDF structured analysis is not configured; parse PDF first")

    async def _extract_graph_triples(
        self,
        prompt: str,
        input_data: dict[str, Any],
        response_model: type[TModel],
    ) -> TModel:
        return await self._generate_structured(prompt, input_data, response_model)

    async def _chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        json_mode: bool = False,
    ) -> str:
        payload: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0.1}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return self._normalize_content(content)
        except Exception as exc:
            logger.warning("DashScope chat completion call failed: %s", exc.__class__.__name__)
            raise LLMAdapterError("DashScope chat completion call failed") from exc

    def _normalize_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text is not None:
                        chunks.append(str(text))
                elif item is not None:
                    chunks.append(str(item))
            return "\n".join(chunk for chunk in chunks if chunk).strip()
        if isinstance(content, dict):
            if "text" in content:
                return str(content["text"])
            return json.dumps(content, ensure_ascii=False)
        return str(content)

    def _to_data_uri(self, file_path: str) -> str:
        path = Path(file_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _extract_structured_payload(self, raw_text: str, response_model: type[TModel]) -> Any:
        payload = self._parse_json(raw_text)
        if isinstance(payload, dict) and self._looks_like_json_schema(payload):
            raise LLMAdapterError("Provider returned a JSON Schema instead of a structured model instance")
        elif isinstance(payload, dict):
            payload = self._fill_required_defaults(payload)
        return payload

    def _looks_like_json_schema(self, payload: dict[str, Any]) -> bool:
        return "properties" in payload and "type" in payload and ("$defs" in payload or "title" in payload)

    def _fill_required_defaults(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = self._normalize_chart_payload(payload)
        for key, value in self._REQUIRED_FALLBACKS.items():
            payload.setdefault(key, value)
        if payload.get("chart_type") not in {"bar", "line", "scatter", "pie", "table", "mixed", "unknown"}:
            raw_chart_type = payload.get("chart_type")
            payload["chart_type"] = self._normalize_chart_type(str(raw_chart_type or ""))
            payload.setdefault("metadata", {})
            if isinstance(payload["metadata"], dict):
                payload["metadata"].setdefault("raw_chart_type", raw_chart_type)
        return payload

    def _normalize_chart_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        for axis_key in ("x_axis", "y_axis"):
            axis_value = normalized.get(axis_key)
            if isinstance(axis_value, list):
                normalized[axis_key] = axis_value[0] if axis_value else None
        series = normalized.get("series")
        if isinstance(series, list):
            normalized_series = []
            for item in series:
                if not isinstance(item, dict):
                    continue
                series_item = dict(item)
                if series_item.get("points") is None:
                    series_item["points"] = []
                normalized_series.append(series_item)
            normalized["series"] = normalized_series
        return normalized

    def _normalize_chart_type(self, raw_chart_type: str) -> str:
        lowered = raw_chart_type.lower()
        for candidate in ("bar", "line", "scatter", "pie", "table"):
            if candidate in lowered:
                return candidate
        if any(token in lowered for token in ("mixed", "network", "chart")):
            return "mixed"
        return "unknown"

    def _parse_json(self, raw_text: str) -> Any:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                return json.loads(cleaned[start : end + 1])
            raise
