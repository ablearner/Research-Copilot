"""Provider failover chain adapter for Kepler.

Wraps a primary LLM adapter with an ordered fallback chain.
On provider-level failures (rate limit, billing, auth, server error),
automatically switches to the next available fallback.

Supports per-turn primary restoration so that a transient failure
does not permanently pin the system onto a fallback provider.
"""

from __future__ import annotations

import logging
from typing import Any

from adapters.llm.base import BaseLLMAdapter, TModel
from adapters.llm.error_classifier import classify_llm_error

logger = logging.getLogger(__name__)


class FallbackLLMAdapter(BaseLLMAdapter):
    """Wraps a primary adapter with an ordered list of fallback adapters."""

    def __init__(
        self,
        primary: BaseLLMAdapter,
        fallbacks: list[BaseLLMAdapter],
        *,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.5,
    ) -> None:
        super().__init__(max_retries=max_retries, retry_delay_seconds=retry_delay_seconds)
        self._primary = primary
        self._fallbacks = fallbacks
        self._active: BaseLLMAdapter = primary
        self._is_fallback_active = False
        self._fallback_index = 0

    @property
    def active_adapter(self) -> BaseLLMAdapter:
        return self._active

    def restore_primary(self) -> None:
        """Restore primary adapter at the start of a new turn."""
        if self._is_fallback_active:
            logger.info("Restoring primary LLM adapter for new turn")
            self._active = self._primary
            self._is_fallback_active = False
            self._fallback_index = 0

    def _activate_next_fallback(self) -> BaseLLMAdapter | None:
        """Switch to the next available fallback adapter."""
        while self._fallback_index < len(self._fallbacks):
            fb = self._fallbacks[self._fallback_index]
            self._fallback_index += 1
            if not fb._is_provider_circuit_open():
                self._active = fb
                self._is_fallback_active = True
                logger.info(
                    "Activated fallback LLM adapter #%d: %s",
                    self._fallback_index,
                    getattr(fb, "model", "unknown"),
                )
                return fb
        return None

    async def _try_with_fallback(self, operation: str, method_name: str, *args: Any) -> Any:
        """Try active adapter, fall back on provider-level errors."""
        try:
            method = getattr(self._active, method_name)
            return await method(*args)
        except Exception as exc:
            classified = classify_llm_error(exc)
            if classified.should_fallback:
                logger.warning(
                    "Primary adapter failed (%s), attempting fallback: %s",
                    classified.reason.value,
                    exc,
                )
                next_adapter = self._activate_next_fallback()
                if next_adapter is not None:
                    method = getattr(next_adapter, method_name)
                    return await method(*args)
            raise

    async def _generate_structured(
        self,
        prompt: str,
        input_data: dict[str, Any],
        response_model: type[TModel],
    ) -> TModel:
        return await self._try_with_fallback(
            "generate_structured",
            "_generate_structured",
            prompt, input_data, response_model,
        )

    async def _analyze_image_structured(
        self,
        prompt: str,
        image_path: str,
        response_model: type[TModel],
    ) -> TModel:
        return await self._try_with_fallback(
            "analyze_image_structured",
            "_analyze_image_structured",
            prompt, image_path, response_model,
        )

    async def _analyze_pdf_structured(
        self,
        prompt: str,
        file_path: str,
        response_model: type[TModel],
    ) -> TModel:
        return await self._try_with_fallback(
            "analyze_pdf_structured",
            "_analyze_pdf_structured",
            prompt, file_path, response_model,
        )

    async def _extract_graph_triples(
        self,
        prompt: str,
        input_data: dict[str, Any],
        response_model: type[TModel],
    ) -> TModel:
        return await self._try_with_fallback(
            "extract_graph_triples",
            "_extract_graph_triples",
            prompt, input_data, response_model,
        )
