from __future__ import annotations

import base64
import json
import logging
import mimetypes
import re
from urllib.parse import urlparse
from pathlib import Path
from typing import Any

import httpx

from adapters.llm.base import BaseLLMAdapter, ImageNotVisibleLLMAdapterError, LLMAdapterError, TModel
from adapters.llm.provider_binding import normalize_openai_base_url
from context.prompt_caching import apply_anthropic_cache_control

logger = logging.getLogger(__name__)

IMAGE_NOT_VISIBLE_SENTINEL = "IMAGE_NOT_VISIBLE_SENTINEL"


class OpenAIRelayAdapter(BaseLLMAdapter):
    _REQUIRED_FALLBACKS = {
        "id": "chart",
        "document_id": "document",
        "page_id": "page",
        "page_number": 1,
        "chart_type": "unknown",
    }

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        vision_model: str | None = None,
        base_url: str = "https://gpt-agent.cc/v1/chat/completions",
        timeout_seconds: float = 90.0,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.5,
    ) -> None:
        super().__init__(max_retries=max_retries, retry_delay_seconds=retry_delay_seconds)
        self.api_key = api_key
        self.model = model
        self.vision_model = vision_model or model
        normalized = normalize_openai_base_url(base_url) or base_url.rstrip("/")
        self.base_url = normalized.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._timeout = httpx.Timeout(
            connect=30.0,
            read=timeout_seconds,
            write=max(timeout_seconds, 120.0),
            pool=timeout_seconds,
        )
        self._uploaded_file_url_cache: dict[str, str] = {}
        self._trust_env = self._should_trust_env_for_base_url(self.base_url)

    async def _generate_structured(
        self,
        prompt: str,
        input_data: dict[str, Any],
        response_model: type[TModel],
    ) -> TModel:
        schema = response_model.model_json_schema()
        payload = await self._chat_completion(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"{json.dumps(input_data, ensure_ascii=False)}\n\n"
                        "Return only valid JSON that matches this JSON Schema:\n"
                        f"{json.dumps(schema, ensure_ascii=False)}"
                    ),
                },
            ],
            json_mode=True,
        )
        return self._validate_response(self._extract_structured_payload(payload, response_model), response_model)

    async def _analyze_image_structured(
        self,
        prompt: str,
        image_path: str,
        response_model: type[TModel],
    ) -> TModel:
        image_descriptor = self._describe_image_input(image_path)
        output_contract = self._image_output_contract(response_model)
        logger.info(
            "OpenAI relay vision request prepared: path=%s exists=%s size=%s mime=%s transport=%s",
            image_descriptor["image_path"],
            image_descriptor["image_exists"],
            image_descriptor["image_size_bytes"],
            image_descriptor["image_mime_type"],
            image_descriptor["image_transport"],
            extra=image_descriptor,
        )
        payload = await self._request_chart_payload(
            prompt=prompt,
            image_url=self._to_data_uri(image_path),
            output_contract=output_contract,
        )
        if self._indicates_missing_image(payload):
            uploaded_url = await self._upload_file_and_get_url(image_path)
            if uploaded_url:
                retry_descriptor = {**image_descriptor, "image_transport": "uploaded_file_url"}
                logger.info(
                    "Retrying OpenAI relay vision request with uploaded file url: path=%s url=%s",
                    retry_descriptor["image_path"],
                    uploaded_url,
                    extra=retry_descriptor,
                )
                payload = await self._request_chart_payload(
                    prompt=prompt,
                    image_url=uploaded_url,
                    output_contract=output_contract,
                )
                if not self._indicates_missing_image(payload):
                    return self._validate_response(
                        self._extract_structured_payload(payload, response_model),
                        response_model,
                    )
            logger.warning(
                "OpenAI relay accepted request but did not see the image: path=%s exists=%s size=%s mime=%s transport=%s",
                image_descriptor["image_path"],
                image_descriptor["image_exists"],
                image_descriptor["image_size_bytes"],
                image_descriptor["image_mime_type"],
                image_descriptor["image_transport"],
                extra=image_descriptor,
            )
            raise ImageNotVisibleLLMAdapterError("Relay provider reported that the image was not visible")
        return self._validate_response(self._extract_structured_payload(payload, response_model), response_model)

    async def answer_image_question(
        self,
        *,
        image_path: str,
        question: str,
        system_prompt: str | None = None,
        context: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        context_text = json.dumps(context or {}, ensure_ascii=False, indent=2)
        history_text = json.dumps(history or [], ensure_ascii=False, indent=2)
        prompt = (
            f"Conversation history:\n{history_text}\n\n"
            f"Additional chart context:\n{context_text}\n\n"
            f"User question:\n{question}\n\n"
            "Answer the user's question based on the attached image. "
            "If the question asks for exact values that are not readable, say that the exact value is not readable and provide the visible trend if possible. "
            "Use the same language as the user question when possible. "
            "If you include mathematical expressions, always use LaTeX delimiters like \\( ... \\) for inline math and \\[ ... \\] for display math, never plain parentheses around formulas."
        )
        return await self._chat_completion(
            model=self.vision_model,
            messages=[
                {"role": "system", "content": system_prompt or "You are a scientific chart image QA assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self._to_data_uri(image_path)}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            json_mode=False,
        )

    async def _analyze_pdf_structured(
        self,
        prompt: str,
        file_path: str,
        response_model: type[TModel],
    ) -> TModel:
        raise LLMAdapterError("OpenAI relay PDF structured analysis is not configured")

    async def _extract_graph_triples(
        self,
        prompt: str,
        input_data: dict[str, Any],
        response_model: type[TModel],
    ) -> TModel:
        return await self._generate_structured(prompt, input_data, response_model)

    def _is_anthropic_provider(self) -> bool:
        """Detect if the relay target is an Anthropic model."""
        model_lower = self.model.lower()
        return any(tag in model_lower for tag in ("claude", "anthropic"))

    async def _chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        json_mode: bool = False,
    ) -> str:
        if self._is_anthropic_provider():
            messages = apply_anthropic_cache_control(messages)
        payload: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0.0 if json_mode else 0.1}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            return await self._post_chat_completion(payload)
        except Exception as exc:
            if json_mode and self._supports_json_mode_fallback(exc):
                logger.info("OpenAI relay rejected response_format; retrying without response_format")
                fallback_payload = dict(payload)
                fallback_payload.pop("response_format", None)
                try:
                    return await self._post_chat_completion(fallback_payload)
                except Exception as retry_exc:
                    logger.warning("OpenAI relay chat completion fallback call failed: %s", retry_exc.__class__.__name__)
                    raise LLMAdapterError(self._format_relay_error("OpenAI relay chat completion call failed", retry_exc)) from retry_exc
            logger.warning("OpenAI relay chat completion call failed: %s", exc.__class__.__name__)
            raise LLMAdapterError(self._format_relay_error("OpenAI relay chat completion call failed", exc)) from exc

    async def _post_chat_completion(self, payload: dict[str, Any]) -> str:
        async with httpx.AsyncClient(timeout=self._timeout, trust_env=self._trust_env) as client:
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
            normalized = self._normalize_content(content)
            if not normalized:
                reason = data["choices"][0].get("finish_reason")
                logger.warning("OpenAI relay returned empty content", extra={"finish_reason": reason})
                raise LLMAdapterError(f"OpenAI relay returned empty content; finish_reason={reason}")
            return normalized

    def _format_relay_error(self, message: str, exc: Exception) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            response = exc.response
            status = response.status_code
            body = response.text.strip().replace("\n", " ")
            if len(body) > 240:
                body = f"{body[:237]}..."
            if body:
                return f"{message} (status={status}; body={body})"
            return f"{message} (status={status})"
        if isinstance(exc, httpx.RequestError):
            detail = str(exc).strip().replace("\n", " ")
            return f"{message} ({exc.__class__.__name__}: {detail})" if detail else f"{message} ({exc.__class__.__name__})"
        detail = str(exc).strip().replace("\n", " ")
        return f"{message} ({exc.__class__.__name__}: {detail})" if detail else f"{message} ({exc.__class__.__name__})"

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

    async def _request_chart_payload(
        self,
        *,
        prompt: str,
        image_url: str,
        output_contract: dict[str, Any],
    ) -> str:
        return await self._chat_completion(
            model=self.vision_model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are analyzing a chart image for a scientific chart assistant. "
                                f"If the image itself is missing or truly not visible to you, reply with exactly: {IMAGE_NOT_VISIBLE_SENTINEL}. "
                                "If you can see the image but it is not actually a chart, plot, table, or scientific figure, do not use the sentinel; instead return a lightweight JSON result with chart_type set to unknown and explain what the image appears to show.\n\n"
                                "Otherwise return exactly one JSON object for the observed chart. "
                                "Do not return markdown fences. Do not return a JSON Schema.\n"
                                f"{json.dumps(output_contract, ensure_ascii=False, indent=2)}"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                },
            ],
            json_mode=False,
        )

    async def _upload_file_and_get_url(self, file_path: str) -> str | None:
        path = Path(file_path)
        cache_key = str(path.resolve())
        cached = self._uploaded_file_url_cache.get(cache_key)
        if cached:
            return cached
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        try:
            async with httpx.AsyncClient(timeout=self._timeout, trust_env=self._trust_env) as client:
                response = await client.post(
                    f"{self.base_url}/files",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={"file": (path.name, path.read_bytes(), mime_type)},
                )
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            logger.warning("OpenAI relay file upload fallback failed: %s", exc.__class__.__name__)
            return None
        uploaded_url = data.get("url")
        if isinstance(uploaded_url, str) and uploaded_url.strip():
            self._uploaded_file_url_cache[cache_key] = uploaded_url.strip()
            return uploaded_url.strip()
        logger.warning("OpenAI relay file upload fallback returned no url")
        return None

    def _should_trust_env_for_base_url(self, base_url: str) -> bool:
        hostname = (urlparse(base_url).hostname or "").lower()
        # The BLTCY relay is reachable directly in this environment, while the system
        # proxy intermittently breaks CONNECT/TLS for long-lived LLM calls.
        if hostname == "api.bltcy.ai":
            return False
        return True

    def _describe_image_input(self, file_path: str) -> dict[str, Any]:
        path = Path(file_path)
        exists = path.exists()
        size_bytes = path.stat().st_size if exists else None
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        return {
            "image_path": str(path),
            "image_exists": exists,
            "image_size_bytes": size_bytes,
            "image_mime_type": mime_type,
            "image_transport": "data_uri:image_url",
        }

    def _image_output_contract(self, response_model: type[TModel]) -> dict[str, Any]:
        if getattr(response_model, "__name__", "") != "ChartSchema":
            return {"instruction": "Return one valid JSON object matching the requested response model."}
        return {
            "type": "Lightweight chart JSON; the backend will map it to ChartSchema",
            "required": {
                "chart_type": "One of: bar, line, scatter, pie, table, mixed, network, unknown.",
                "summary": "One or two concise sentences describing visible chart content.",
            },
            "optional": {
                "title": "Visible title string or null.",
                "caption": "Visible caption string or null.",
                "x_axis_label": "Visible x-axis label or null.",
                "y_axis_label": "Visible y-axis label or null.",
                "series_names": "Array of visible series names or color-based names; use [] if none.",
                "visible_trends": "Array of short visible trends or observations; use [] if none.",
                "confidence": "Number between 0 and 1.",
                "metadata": "Small object; use {} if none.",
            },
            "example": {
                "chart_type": "mixed",
                "title": None,
                "caption": None,
                "x_axis_label": "node number",
                "y_axis_label": "variance",
                "series_names": ["blue line", "red line"],
                "visible_trends": ["The image contains a network graph and two line charts."],
                "summary": "The image shows a green node-link network on the left and two line charts on the right, with blue and red series plotted against node number.",
                "confidence": 0.85,
                "metadata": {},
            },
            "hard_rules": [
                f"If the image is not visible, reply exactly with {IMAGE_NOT_VISIBLE_SENTINEL}.",
                "If the image is visible but not a chart, return chart_type as unknown and describe the visible content in summary.",
                "Do not return markdown fences.",
                "Do not return a JSON Schema.",
                "Do not estimate exact data points unless they are clearly readable.",
                "Use null or [] for unreadable fields.",
            ],
        }

    def _indicates_missing_image(self, text: str) -> bool:
        normalized = text.strip().lower()
        if not normalized:
            return False
        patterns = (
            IMAGE_NOT_VISIBLE_SENTINEL.lower(),
            "i don't see",
            "i do not see",
            "no image was provided",
            "image is missing",
            "image not visible",
            "cannot see the image",
            "can't see the image",
            "do not have access to the image",
        )
        return any(pattern in normalized for pattern in patterns)

    def _extract_structured_payload(self, raw_text: str, response_model: type[TModel]) -> dict[str, Any]:
        payload = self._parse_json(raw_text)
        if not isinstance(payload, dict):
            raise LLMAdapterError("Relay provider did not return a JSON object")
        if self._looks_like_json_schema(payload):
            raise LLMAdapterError("Relay provider returned a JSON Schema instead of a model instance")
        if getattr(response_model, "__name__", "") != "ChartSchema":
            return payload
        return self._fill_required_defaults(payload)

    def _looks_like_json_schema(self, payload: dict[str, Any]) -> bool:
        return "properties" in payload and "type" in payload and ("$defs" in payload or "title" in payload)

    def _fill_required_defaults(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize_lightweight_chart_payload(dict(payload))
        for key, value in self._REQUIRED_FALLBACKS.items():
            normalized.setdefault(key, value)
        if normalized.get("chart_type") not in {"bar", "line", "scatter", "pie", "table", "mixed", "unknown"}:
            raw_chart_type = normalized.get("chart_type")
            normalized["chart_type"] = self._normalize_chart_type(str(raw_chart_type or ""))
            normalized.setdefault("metadata", {})
            if isinstance(normalized["metadata"], dict):
                normalized["metadata"].setdefault("raw_chart_type", raw_chart_type)
        return normalized

    def _normalize_lightweight_chart_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = self._flatten_contract_sections(dict(payload))
        if self._is_full_chart_payload(payload):
            return self._normalize_chart_payload(payload)

        normalized = dict(payload)
        metadata = normalized.get("metadata") if isinstance(normalized.get("metadata"), dict) else {}
        visible_trends = normalized.pop("visible_trends", [])
        if visible_trends:
            metadata["visible_trends"] = visible_trends
        confidence = normalized.get("confidence")
        if isinstance(confidence, str):
            lowered = confidence.strip().lower()
            confidence_map = {
                "high": 0.9,
                "medium": 0.6,
                "low": 0.3,
            }
            normalized["confidence"] = confidence_map.get(lowered)
        if "x_axis_label" in normalized and "x_axis" not in normalized:
            x_label = normalized.pop("x_axis_label")
            normalized["x_axis"] = {"label": x_label, "name": x_label} if x_label else None
        if "y_axis_label" in normalized and "y_axis" not in normalized:
            y_label = normalized.pop("y_axis_label")
            normalized["y_axis"] = {"label": y_label, "name": y_label} if y_label else None
        series_names = normalized.pop("series_names", None)
        if series_names is not None and "series" not in normalized:
            if isinstance(series_names, str):
                series_names = [series_names]
            if isinstance(series_names, list):
                normalized["series"] = [
                    {"name": str(name), "chart_role": "unknown", "points": []}
                    for name in series_names
                    if str(name).strip()
                ]
        normalized.pop("id", None)
        normalized.pop("document_id", None)
        normalized.pop("page_id", None)
        normalized.pop("page_number", None)
        normalized["metadata"] = metadata
        return self._normalize_chart_payload(normalized)

    def _flatten_contract_sections(self, payload: dict[str, Any]) -> dict[str, Any]:
        flattened = dict(payload)
        for section in ("required", "optional"):
            section_value = flattened.pop(section, None)
            if isinstance(section_value, dict):
                for key, value in section_value.items():
                    flattened.setdefault(key, value)
        flattened.pop("type", None)
        return flattened

    def _is_full_chart_payload(self, payload: dict[str, Any]) -> bool:
        return any(key in payload for key in ("id", "document_id", "page_id", "page_number", "x_axis", "y_axis", "series"))

    def _normalize_chart_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        for axis_key in ("x_axis", "y_axis"):
            axis_value = payload.get(axis_key)
            if isinstance(axis_value, list):
                payload[axis_key] = axis_value[0] if axis_value else None
        series = payload.get("series")
        if isinstance(series, list):
            normalized_series = []
            for item in series:
                if not isinstance(item, dict):
                    continue
                series_item = dict(item)
                if series_item.get("points") is None:
                    series_item["points"] = []
                normalized_series.append(series_item)
            payload["series"] = normalized_series
        return payload

    def _normalize_chart_type(self, raw_chart_type: str) -> str:
        lowered = raw_chart_type.lower()
        for candidate in ("bar", "line", "scatter", "pie", "table"):
            if candidate in lowered:
                return candidate
        if any(token in lowered for token in ("mixed", "network", "chart")):
            return "mixed"
        return "unknown"

    def _parse_json(self, raw_text: str) -> Any:
        last_error: json.JSONDecodeError | None = None
        for candidate in self._iter_json_candidates(raw_text):
            try:
                return self._load_json_candidate(candidate)
            except json.JSONDecodeError as exc:
                last_error = exc
                repaired = self._repair_json_candidate(candidate, exc)
                if repaired == candidate:
                    continue
                try:
                    payload = self._load_json_candidate(repaired)
                except json.JSONDecodeError as repaired_exc:
                    last_error = repaired_exc
                    continue
                logger.info("Recovered malformed JSON from OpenAI relay response")
                return payload

        if last_error is not None:
            raise last_error
        raise json.JSONDecodeError("OpenAI relay response did not contain JSON", raw_text, 0)

    def _iter_json_candidates(self, raw_text: str) -> list[str]:
        cleaned = self._strip_code_fences(raw_text.strip())
        candidates: list[str] = []
        for candidate in (
            cleaned,
            self._slice_from_first_json_token(cleaned),
            self._extract_balanced_json_fragment(cleaned),
        ):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def _strip_code_fences(self, raw_text: str) -> str:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned, count=1)
            cleaned = re.sub(r"\s*```$", "", cleaned, count=1)
        return cleaned.strip()

    def _slice_from_first_json_token(self, raw_text: str) -> str | None:
        start_indexes = [index for index in (raw_text.find("{"), raw_text.find("[")) if index >= 0]
        if not start_indexes:
            return None
        start = min(start_indexes)
        return raw_text[start:].strip()

    def _extract_balanced_json_fragment(self, raw_text: str) -> str | None:
        candidate = self._slice_from_first_json_token(raw_text)
        if not candidate:
            return None

        stack: list[str] = []
        in_string = False
        escape = False
        for index, char in enumerate(candidate):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue
            if char in "{[":
                stack.append(char)
                continue
            if char in "}]":
                if not stack:
                    break
                opener = stack.pop()
                if (opener == "{" and char != "}") or (opener == "[" and char != "]"):
                    break
                if not stack:
                    return candidate[: index + 1].strip()
        return None

    def _load_json_candidate(self, candidate: str) -> Any:
        cleaned = candidate.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            payload, end_index = json.JSONDecoder().raw_decode(cleaned)
            if cleaned[end_index:].strip():
                logger.info("Ignoring trailing non-JSON text in OpenAI relay response")
            return payload

    def _repair_json_candidate(self, candidate: str, first_error: json.JSONDecodeError) -> str:
        repaired = candidate.strip().replace("\ufeff", "")
        repaired = repaired.translate(
            str.maketrans(
                {
                    "\u2018": "'",
                    "\u2019": "'",
                    "\u201c": '"',
                    "\u201d": '"',
                }
            )
        )
        repaired = self._remove_trailing_commas(repaired)
        repaired = self._close_open_json_structures(repaired)
        repaired = self._insert_missing_commas(repaired)
        repaired = self._remove_trailing_commas(repaired)
        return repaired

    def _remove_trailing_commas(self, candidate: str) -> str:
        return re.sub(r",(\s*[}\]])", r"\1", candidate)

    def _insert_missing_commas(
        self,
        candidate: str,
        *,
        first_error: json.JSONDecodeError | None = None,
        max_repairs: int = 8,
    ) -> str:
        repaired = candidate
        current_error = first_error
        for _ in range(max_repairs):
            if current_error is None:
                try:
                    json.loads(repaired)
                    return repaired
                except json.JSONDecodeError as exc:
                    current_error = exc

            if current_error is None or "Expecting ',' delimiter" not in current_error.msg:
                break

            insert_at = current_error.pos
            if insert_at <= 0 or insert_at > len(repaired):
                break
            repaired = repaired[:insert_at] + "," + repaired[insert_at:]
            current_error = None
        return repaired

    def _close_open_json_structures(self, candidate: str) -> str:
        closers = {"{": "}", "[": "]"}
        stack: list[str] = []
        in_string = False
        escape = False

        for char in candidate:
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue
            if char in closers:
                stack.append(char)
                continue
            if char in "}]":
                if stack and closers[stack[-1]] == char:
                    stack.pop()

        repaired = candidate
        if in_string:
            repaired += '"'
        if stack:
            repaired += "".join(closers[opener] for opener in reversed(stack))
        return repaired

    def _supports_json_mode_fallback(self, exc: Exception) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code
            response_text = exc.response.text.lower()
            if status_code in {400, 404, 415, 422} and any(
                token in response_text
                for token in (
                    "response_format",
                    "json_object",
                    "unsupported",
                    "not support",
                    "not supported",
                    "invalid parameter",
                )
            ):
                return True

        message = str(exc).lower()
        return any(
            token in message
            for token in (
                "response_format",
                "json_object",
                "not support",
                "not supported",
                "unsupported",
            )
        )
