import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from domain.schemas.embedding import EmbeddingVector

logger = logging.getLogger(__name__)


class EmbeddingAdapterError(RuntimeError):
    """Raised when an embedding adapter call fails."""


class BaseEmbeddingAdapter(ABC):
    def __init__(self, max_retries: int = 3, retry_delay_seconds: float = 1.0) -> None:
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

    async def embed_text(self, text: str) -> EmbeddingVector:
        return await self._run_with_retries("embed_text", lambda: self._embed_text(text))

    async def embed_texts(self, texts: list[str]) -> list[EmbeddingVector]:
        return await self._run_with_retries("embed_texts", lambda: self._embed_texts(texts))

    async def embed_image(self, image_path: str) -> EmbeddingVector:
        return await self._run_with_retries("embed_image", lambda: self._embed_image(image_path))

    async def embed_page(self, page_image_path: str, page_text: str) -> EmbeddingVector:
        return await self._run_with_retries(
            "embed_page",
            lambda: self._embed_page(page_image_path, page_text),
        )

    async def embed_chart(self, chart_image_path: str, chart_summary: str) -> EmbeddingVector:
        return await self._run_with_retries(
            "embed_chart",
            lambda: self._embed_chart(chart_image_path, chart_summary),
        )

    async def _run_with_retries(
        self,
        operation: str,
        call: Callable[[], Awaitable[EmbeddingVector | list[EmbeddingVector]]],
    ) -> EmbeddingVector | list[EmbeddingVector]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await call()
            except (EmbeddingAdapterError, OSError, ValueError, TimeoutError) as exc:
                last_error = exc
                delay = self.retry_delay_seconds * (2 ** attempt)
                logger.warning(
                    "Embedding adapter operation failed (attempt %d/%d, retrying in %.1fs): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    delay if attempt < self.max_retries else 0,
                    str(exc)[:200],
                    extra={"operation": operation, "attempt": attempt + 1},
                )
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(delay)
        raise EmbeddingAdapterError(f"Embedding adapter operation failed after {self.max_retries + 1} attempts: {operation}") from last_error

    @abstractmethod
    async def _embed_text(self, text: str) -> EmbeddingVector:
        raise NotImplementedError

    @abstractmethod
    async def _embed_texts(self, texts: list[str]) -> list[EmbeddingVector]:
        raise NotImplementedError

    @abstractmethod
    async def _embed_image(self, image_path: str) -> EmbeddingVector:
        raise NotImplementedError

    @abstractmethod
    async def _embed_page(self, page_image_path: str, page_text: str) -> EmbeddingVector:
        raise NotImplementedError

    @abstractmethod
    async def _embed_chart(self, chart_image_path: str, chart_summary: str) -> EmbeddingVector:
        raise NotImplementedError
