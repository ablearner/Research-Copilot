import logging
from typing import Any

import httpx

from adapters.embedding.base import BaseEmbeddingAdapter, EmbeddingAdapterError
from adapters.llm.provider_binding import normalize_openai_base_url
from domain.schemas.embedding import EmbeddingVector

logger = logging.getLogger(__name__)


class DashScopeEmbeddingAdapter(BaseEmbeddingAdapter):
    def __init__(
        self,
        api_key: str,
        text_model: str = "text-embedding-v4",
        multimodal_model: str | None = None,
        text_batch_size: int = 16,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout_seconds: float = 120.0,
        connect_timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        super().__init__(max_retries=max_retries, retry_delay_seconds=retry_delay_seconds)
        self.api_key = api_key
        self.text_model = text_model
        self.multimodal_model = multimodal_model or text_model
        self.text_batch_size = max(1, text_batch_size)
        normalized_base_url = normalize_openai_base_url(base_url) or base_url.rstrip("/")
        self.base_url = normalized_base_url.rstrip("/")
        self.timeout = httpx.Timeout(
            timeout=timeout_seconds,
            connect=connect_timeout_seconds,
            read=timeout_seconds,
            write=60.0,
            pool=30.0,
        )
        self._client: httpx.AsyncClient | None = None

    async def _embed_text(self, text: str) -> EmbeddingVector:
        return (await self._embed_texts([text]))[0]

    async def _embed_texts(self, texts: list[str]) -> list[EmbeddingVector]:
        if not texts:
            return []
        vectors: list[EmbeddingVector] = []
        for batch in self._batches(texts, size=self.text_batch_size):
            payload = await self._create_embedding(self.text_model, batch)
            vectors.extend(
                EmbeddingVector(
                    model=self.text_model,
                    dimensions=len(item["embedding"]),
                    values=list(item["embedding"]),
                )
                for item in payload.get("data", [])
            )
        return vectors

    async def _embed_image(self, image_path: str) -> EmbeddingVector:
        # The default business runtime stores image/chart summaries as text embeddings.
        return await self._embed_text(f"image:{image_path}")

    async def _embed_page(self, page_image_path: str, page_text: str) -> EmbeddingVector:
        return await self._embed_text(f"page_image:{page_image_path}\n{page_text}")

    async def _embed_chart(self, chart_image_path: str, chart_summary: str) -> EmbeddingVector:
        return await self._embed_text(f"chart_image:{chart_image_path}\n{chart_summary}")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def _reset_client(self) -> None:
        client = self._client
        self._client = None
        if client is not None and not client.is_closed:
            await client.aclose()

    async def _create_embedding(self, model: str, inputs: list[str]) -> dict[str, Any]:
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/embeddings",
                json={"model": model, "input": inputs},
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            await self._reset_client()
            logger.warning(
                "Embedding API call timed out: %s (url=%s, model=%s, input_count=%d)",
                exc.__class__.__name__,
                self.base_url,
                model,
                len(inputs),
            )
            raise EmbeddingAdapterError(
                f"Embedding API call timed out ({exc.__class__.__name__})"
            ) from exc
        except httpx.TransportError as exc:
            await self._reset_client()
            logger.warning(
                "Embedding API transport failed: %s (url=%s, model=%s, input_count=%d)",
                exc.__class__.__name__,
                self.base_url,
                model,
                len(inputs),
            )
            raise EmbeddingAdapterError(
                f"Embedding API transport failed ({exc.__class__.__name__})"
            ) from exc
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500] if exc.response is not None else ""
            logger.warning(
                "Embedding API call failed with HTTP status",
                extra={"status_code": exc.response.status_code if exc.response else None, "response_body": body},
            )
            raise EmbeddingAdapterError("Embedding API call failed") from exc
        except Exception as exc:
            logger.warning("Embedding API call failed: %s", exc.__class__.__name__)
            raise EmbeddingAdapterError("Embedding API call failed") from exc

    async def close(self) -> None:
        """Close the underlying HTTP client to release connections."""
        await self._reset_client()

    def _batches(self, texts: list[str], size: int) -> list[list[str]]:
        return [texts[index : index + size] for index in range(0, len(texts), size)]
