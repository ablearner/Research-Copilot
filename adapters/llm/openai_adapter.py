import base64
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from adapters.llm.base import BaseLLMAdapter, LLMAdapterError, TModel

logger = logging.getLogger(__name__)


class OpenAILLMAdapter(BaseLLMAdapter):
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
        payload = await self._parse_with_responses_api(
            prompt=prompt,
            input_content=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(input_data, ensure_ascii=False)},
            ],
            response_model=response_model,
        )
        return self._validate_response(payload, response_model)

    async def _analyze_image_structured(
        self,
        prompt: str,
        image_path: str,
        response_model: type[TModel],
    ) -> TModel:
        image_uri = self._to_data_uri(image_path)
        payload = await self._parse_with_responses_api(
            prompt=prompt,
            input_content=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Analyze the image and return structured JSON."},
                        {"type": "input_image", "image_url": image_uri},
                    ],
                },
            ],
            response_model=response_model,
        )
        return self._validate_response(payload, response_model)

    async def _analyze_pdf_structured(
        self,
        prompt: str,
        file_path: str,
        response_model: type[TModel],
    ) -> TModel:
        pdf_uri = self._to_data_uri(file_path, default_mime_type="application/pdf")
        payload = await self._parse_with_responses_api(
            prompt=prompt,
            input_content=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Analyze the PDF and return structured JSON."},
                        {"type": "input_file", "filename": Path(file_path).name, "file_data": pdf_uri},
                    ],
                },
            ],
            response_model=response_model,
        )
        return self._validate_response(payload, response_model)

    async def _extract_graph_triples(
        self,
        prompt: str,
        input_data: dict[str, Any],
        response_model: type[TModel],
    ) -> TModel:
        return await self._generate_structured(prompt, input_data, response_model)

    async def _parse_with_responses_api(
        self,
        prompt: str,
        input_content: list[dict[str, Any]],
        response_model: type[BaseModel],
    ) -> Any:
        try:
            if hasattr(self.client, "responses"):
                response = await self.client.responses.parse(
                    model=self.model,
                    input=input_content,
                    text_format=response_model,
                )
                parsed = getattr(response, "output_parsed", None)
                if parsed is not None:
                    return parsed
                output_text = getattr(response, "output_text", None)
                if output_text:
                    return json.loads(output_text)
            if hasattr(self.client, "beta") and hasattr(self.client.beta, "chat"):
                completion = await self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=input_content,
                    response_format=response_model,
                )
                return completion.choices[0].message.parsed
        except Exception as exc:
            logger.exception("OpenAI LLM adapter call failed")
            raise LLMAdapterError("OpenAI LLM adapter call failed") from exc
        raise LLMAdapterError("OpenAI client does not expose a supported structured output API")

    def _to_data_uri(self, file_path: str, default_mime_type: str = "application/octet-stream") -> str:
        path = Path(file_path)
        mime_type = mimetypes.guess_type(path.name)[0] or default_mime_type
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
