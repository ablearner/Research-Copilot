from __future__ import annotations

import asyncio
import contextlib
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from domain.schemas.retrieval import RetrievalHit


@contextlib.contextmanager
def _silence_model_load_logs() -> object:
    logger_names = (
        "transformers",
        "transformers.modeling_utils",
        "transformers.utils.loading_report",
        "sentence_transformers",
    )
    previous: list[tuple[logging.Logger, int, bool]] = []
    try:
        for name in logger_names:
            logger = logging.getLogger(name)
            previous.append((logger, logger.level, logger.propagate))
            logger.setLevel(logging.ERROR)
            logger.propagate = False
        yield
    finally:
        for logger, level, propagate in previous:
            logger.setLevel(level)
            logger.propagate = propagate


class BaseCrossEncoderReranker(Protocol):
    async def score(self, query: str, documents: list[str]) -> list[float]:
        """Return a relevance score for each query-document pair."""


@dataclass(slots=True)
class SentenceTransformersCrossEncoderReranker:
    model_name: str
    batch_size: int = 16
    max_length: int = 512
    allow_download: bool = False
    cache_dir: str | None = None
    _model: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "sentence-transformers is required for Cross-Encoder reranking. "
                "Install project dependencies with the reranker extras available."
            ) from exc
        cache_folder = str(Path(self.cache_dir).expanduser()) if self.cache_dir else None
        try:
            with _silence_model_load_logs(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self._model = CrossEncoder(
                    self.model_name,
                    max_length=self.max_length,
                    cache_folder=cache_folder,
                    local_files_only=not self.allow_download,
                )
        except Exception as exc:
            if not self.allow_download:
                raise RuntimeError(
                    f"Cross-Encoder model '{self.model_name}' is not available locally. "
                    "Set RERANKER_ALLOW_DOWNLOAD=true to fetch it, "
                    "or set RERANKER_UNAVAILABLE_POLICY=heuristic to allow fallback."
                ) from exc
            raise RuntimeError(
                f"Cross-Encoder model '{self.model_name}' could not be loaded."
            ) from exc

    async def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        pairs = [(query, document) for document in documents]
        return await asyncio.to_thread(
            self._predict,
            pairs,
        )

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(score) for score in scores]


@dataclass(slots=True)
class HeuristicFallbackReranker:
    reason: str = "cross_encoder_unavailable"

    async def score(self, query: str, documents: list[str]) -> list[float]:
        del query
        return [0.0 for _ in documents]

    async def rerank_hits(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        del query
        reranked = []
        for hit in hits:
            fallback_score = float(
                hit.merged_score
                or hit.vector_score
                or hit.graph_score
                or hit.sparse_score
                or 0.0
            )
            reranked.append(
                hit.model_copy(
                    update={
                        "merged_score": fallback_score,
                        "metadata": {
                            **hit.metadata,
                            "reranker_fallback": self.reason,
                        },
                    }
                )
            )
        return sorted(reranked, key=lambda item: item.merged_score or 0.0, reverse=True)
