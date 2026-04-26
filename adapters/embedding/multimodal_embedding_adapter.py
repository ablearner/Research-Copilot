import base64
import logging
from pathlib import Path
from typing import Any

from adapters.embedding.base import BaseEmbeddingAdapter, EmbeddingAdapterError
from domain.schemas.embedding import EmbeddingVector

logger = logging.getLogger(__name__)


class MultimodalEmbeddingAdapter(BaseEmbeddingAdapter):
    def __init__(
        self,
        client: Any,
        text_model: str,
        multimodal_model: str | None = None,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.5,
    ) -> None:
        super().__init__(max_retries=max_retries, retry_delay_seconds=retry_delay_seconds)
        self.client = client
        self.text_model = text_model
        self.multimodal_model = multimodal_model or text_model

    async def _embed_text(self, text: str) -> EmbeddingVector:
        return (await self._embed_texts([text]))[0]

    async def _embed_texts(self, texts: list[str]) -> list[EmbeddingVector]:
        if not texts:
            return []
        try:
            response = await self.client.embeddings.create(model=self.text_model, input=texts)
            return [
                EmbeddingVector(
                    model=self.text_model,
                    dimensions=len(item.embedding),
                    values=list(item.embedding),
                )
                for item in response.data
            ]
        except Exception as exc:
            logger.exception("Text embedding adapter call failed")
            raise EmbeddingAdapterError("Text embedding adapter call failed") from exc

    async def _embed_image(self, image_path: str) -> EmbeddingVector:
        return await self._embed_multimodal(
            text="Represent this image for semantic retrieval.",
            image_path=image_path,
        )

    async def _embed_page(self, page_image_path: str, page_text: str) -> EmbeddingVector:
        return await self._embed_multimodal(
            text=f"Represent this document page for semantic retrieval.\n\nPage text:\n{page_text}",
            image_path=page_image_path,
        )

    async def _embed_chart(self, chart_image_path: str, chart_summary: str) -> EmbeddingVector:
        return await self._embed_multimodal(
            text=f"Represent this chart for semantic retrieval.\n\nChart summary:\n{chart_summary}",
            image_path=chart_image_path,
        )

    async def _embed_multimodal(self, text: str, image_path: str) -> EmbeddingVector:
        try:
            if not hasattr(self.client, "embeddings"):
                raise EmbeddingAdapterError("Client does not expose an embeddings API")
            image_payload = {
                "type": "input_image",
                "image_url": self._to_data_uri(image_path),
            }
            response = await self.client.embeddings.create(
                model=self.multimodal_model,
                input=[{"type": "input_text", "text": text}, image_payload],
            )
            values = list(response.data[0].embedding)
            return EmbeddingVector(
                model=self.multimodal_model,
                dimensions=len(values),
                values=values,
            )
        except Exception as exc:
            logger.exception("Multimodal embedding adapter call failed")
            raise EmbeddingAdapterError("Multimodal embedding adapter call failed") from exc

    def _to_data_uri(self, image_path: str) -> str:
        encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
