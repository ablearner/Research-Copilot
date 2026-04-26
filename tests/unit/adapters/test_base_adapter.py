import pytest
from pydantic import BaseModel

from adapters.llm.base import BaseLLMAdapter, LLMAdapterError


class DummyResponse(BaseModel):
    value: str


class PermissionDeniedError(Exception):
    def __init__(self, message: str = "free tier exhausted", status_code: int = 403) -> None:
        super().__init__(message)
        self.status_code = status_code


class DummyAdapter(BaseLLMAdapter):
    def __init__(self) -> None:
        super().__init__(max_retries=2, retry_delay_seconds=0)
        self.calls = 0

    async def _generate_structured(self, prompt: str, input_data: dict, response_model: type[BaseModel]):
        self.calls += 1
        raise PermissionDeniedError()

    async def _analyze_image_structured(self, prompt: str, image_path: str, response_model: type[BaseModel]):
        raise NotImplementedError

    async def _analyze_pdf_structured(self, prompt: str, file_path: str, response_model: type[BaseModel]):
        raise NotImplementedError

    async def _extract_graph_triples(self, prompt: str, input_data: dict, response_model: type[BaseModel]):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_generate_structured_wraps_non_retryable_provider_error() -> None:
    adapter = DummyAdapter()

    with pytest.raises(LLMAdapterError, match="PermissionDeniedError"):
        await adapter.generate_structured(
            prompt="Return JSON",
            input_data={"question": "what happened"},
            response_model=DummyResponse,
        )

    assert adapter.calls == 1


@pytest.mark.asyncio
async def test_generate_structured_short_circuits_after_provider_permission_error() -> None:
    adapter = DummyAdapter()

    with pytest.raises(LLMAdapterError, match="PermissionDeniedError"):
        await adapter.generate_structured(
            prompt="Return JSON",
            input_data={"question": "first call"},
            response_model=DummyResponse,
        )

    with pytest.raises(LLMAdapterError, match="temporarily short-circuited"):
        await adapter.generate_structured(
            prompt="Return JSON",
            input_data={"question": "second call"},
            response_model=DummyResponse,
        )

    assert adapter.calls == 1
