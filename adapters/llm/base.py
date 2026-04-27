import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)


class LLMAdapterError(RuntimeError):
    """Raised when an LLM adapter call fails."""

    def __init__(self, message: str = "", *, classified=None):
        super().__init__(message)
        self.classified = classified


class ImageNotVisibleLLMAdapterError(LLMAdapterError):
    """Raised when the provider accepted a vision request but could not see the image."""


def is_expected_provider_error(exc: Exception) -> bool:
    if isinstance(exc, (LLMAdapterError, ValidationError, ValueError)):
        return True

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and 400 <= status_code < 500 and status_code not in {408, 409, 429}:
        return True

    error_name = exc.__class__.__name__
    if error_name in {"AuthenticationError", "PermissionDeniedError", "BadRequestError", "NotFoundError"}:
        return True

    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "allocationquota",
            "free tier",
            "free-tier",
            "insufficient_quota",
            "quota exceeded",
            "permission denied",
            "authentication",
            "unauthorized",
            "forbidden",
        )
    )


def format_llm_error(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    status_fragment = f" status={status_code}" if status_code is not None else ""
    message = str(exc).strip().replace("\n", " ")
    if len(message) > 220:
        message = f"{message[:217]}..."
    return f"{exc.__class__.__name__}{status_fragment}: {message}" if message else exc.__class__.__name__


def should_open_provider_circuit(exc: Exception) -> bool:
    if isinstance(exc, LLMAdapterError):
        cause = exc.__cause__
        if isinstance(cause, Exception):
            return should_open_provider_circuit(cause)
        return False

    status_code = getattr(exc, "status_code", None)
    if status_code in {401, 403}:
        return True

    error_name = exc.__class__.__name__
    if error_name in {"AuthenticationError", "PermissionDeniedError"}:
        return True

    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "allocationquota",
            "free tier",
            "free-tier",
            "insufficient_quota",
            "quota exceeded",
            "permission denied",
            "authentication",
            "unauthorized",
            "forbidden",
        )
    )


class BaseLLMAdapter(ABC):
    def __init__(
        self,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.5,
        provider_error_cooldown_seconds: float = 45.0,
    ) -> None:
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.provider_error_cooldown_seconds = max(provider_error_cooldown_seconds, 0.0)
        self._provider_circuit_open_until = 0.0
        self._provider_circuit_error: Exception | None = None

    async def generate_structured(
        self,
        prompt: str,
        input_data: dict[str, Any],
        response_model: type[TModel],
    ) -> TModel:
        return await self._run_with_retries(
            "generate_structured",
            lambda: self._generate_structured(prompt, input_data, response_model),
        )

    async def analyze_image_structured(
        self,
        prompt: str,
        image_path: str,
        response_model: type[TModel],
    ) -> TModel:
        return await self._run_with_retries(
            "analyze_image_structured",
            lambda: self._analyze_image_structured(prompt, image_path, response_model),
        )

    async def analyze_pdf_structured(
        self,
        prompt: str,
        file_path: str,
        response_model: type[TModel],
    ) -> TModel:
        return await self._run_with_retries(
            "analyze_pdf_structured",
            lambda: self._analyze_pdf_structured(prompt, file_path, response_model),
        )

    async def extract_graph_triples(
        self,
        prompt: str,
        input_data: dict[str, Any],
        response_model: type[TModel],
    ) -> TModel:
        return await self._run_with_retries(
            "extract_graph_triples",
            lambda: self._extract_graph_triples(prompt, input_data, response_model),
        )

    async def _run_with_retries(
        self,
        operation: str,
        call: Callable[[], Awaitable[TModel]],
    ) -> TModel:
        self._raise_if_provider_circuit_open(operation)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await call()
            except asyncio.CancelledError:
                raise
            except (LLMAdapterError, ValidationError, OSError, ValueError, Exception) as exc:
                last_error = exc
                if is_expected_provider_error(exc):
                    logger.warning(
                        "LLM adapter operation failed: %s",
                        format_llm_error(exc),
                        extra={"operation": operation, "attempt": attempt + 1},
                    )
                else:
                    logger.warning(
                        "LLM adapter operation failed",
                        extra={"operation": operation, "attempt": attempt + 1},
                        exc_info=exc,
                    )
                if should_open_provider_circuit(exc):
                    self._open_provider_circuit(exc, operation=operation)
                if attempt >= self.max_retries or not self._should_retry_exception(exc):
                    break
                await asyncio.sleep(self.retry_delay_seconds * (2**attempt))
        if isinstance(last_error, LLMAdapterError):
            raise last_error
        error_name = last_error.__class__.__name__ if last_error is not None else "UnknownError"
        from adapters.llm.error_classifier import classify_llm_error
        classified = classify_llm_error(last_error) if last_error is not None else None
        raise LLMAdapterError(
            f"LLM adapter operation failed: {operation} ({error_name})",
            classified=classified,
        ) from last_error

    def _should_retry_exception(self, exc: Exception) -> bool:
        if is_expected_provider_error(exc):
            return False

        return True

    def _raise_if_provider_circuit_open(self, operation: str) -> None:
        if not self._is_provider_circuit_open():
            return
        reason = format_llm_error(self._provider_circuit_error) if self._provider_circuit_error is not None else "provider error"
        raise LLMAdapterError(f"LLM provider temporarily short-circuited: {operation} ({reason})") from self._provider_circuit_error

    def _is_provider_circuit_open(self) -> bool:
        return self.provider_error_cooldown_seconds > 0 and time.monotonic() < self._provider_circuit_open_until

    def _open_provider_circuit(self, exc: Exception, *, operation: str) -> None:
        if self.provider_error_cooldown_seconds <= 0:
            return
        self._provider_circuit_error = exc
        self._provider_circuit_open_until = time.monotonic() + self.provider_error_cooldown_seconds
        logger.warning(
            "Opening LLM provider circuit breaker for %.1fs after %s",
            self.provider_error_cooldown_seconds,
            operation,
            extra={"operation": operation},
        )

    def _validate_response(self, payload: Any, response_model: type[TModel]) -> TModel:
        if isinstance(payload, response_model):
            return payload
        return response_model.model_validate(payload)

    @abstractmethod
    async def _generate_structured(
        self,
        prompt: str,
        input_data: dict[str, Any],
        response_model: type[TModel],
    ) -> TModel:
        raise NotImplementedError

    @abstractmethod
    async def _analyze_image_structured(
        self,
        prompt: str,
        image_path: str,
        response_model: type[TModel],
    ) -> TModel:
        raise NotImplementedError

    @abstractmethod
    async def _analyze_pdf_structured(
        self,
        prompt: str,
        file_path: str,
        response_model: type[TModel],
    ) -> TModel:
        raise NotImplementedError

    @abstractmethod
    async def _extract_graph_triples(
        self,
        prompt: str,
        input_data: dict[str, Any],
        response_model: type[TModel],
    ) -> TModel:
        raise NotImplementedError
