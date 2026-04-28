from adapters.embedding.dashscope_adapter import DashScopeEmbeddingAdapter
from adapters.embedding.openai import OpenAIEmbeddingAdapter
from apps.api.runtime import _build_embedding_adapter
from core.config import Settings


def test_build_embedding_adapter_normalizes_dashscope_relay_base_url() -> None:
    settings = Settings(
        embedding_provider="dashscope",
        embedding_model="text-embedding-3-small",
        embedding_text_batch_size=24,
        dashscope_api_key="relay-key",
        dashscope_base_url="https://api.bltcy.ai/",
    )

    adapter = _build_embedding_adapter(settings)

    assert isinstance(adapter, DashScopeEmbeddingAdapter)
    assert adapter.text_model == "text-embedding-3-small"
    assert adapter.text_batch_size == 24
    assert adapter.base_url == "https://api.bltcy.ai/v1"


def test_build_embedding_adapter_uses_openai_relay_for_openai_provider() -> None:
    settings = Settings(
        embedding_provider="openai_compatible",
        embedding_model="text-embedding-3-small",
        embedding_text_batch_size=20,
        openai_api_key="openai-key",
        openai_base_url="https://api.bltcy.ai/",
    )

    adapter = _build_embedding_adapter(settings)

    assert isinstance(adapter, OpenAIEmbeddingAdapter)
    assert adapter.text_model == "text-embedding-3-small"
    assert adapter.text_batch_size == 20
    assert adapter.base_url == "https://api.bltcy.ai/v1"


def test_build_embedding_adapter_openai_provider_can_fallback_to_dashscope_relay_credentials() -> None:
    settings = Settings(
        _env_file=None,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        openai_api_key="",
        dashscope_api_key="relay-key",
        dashscope_base_url="https://api.bltcy.ai/",
    )

    adapter = _build_embedding_adapter(settings)

    assert isinstance(adapter, OpenAIEmbeddingAdapter)
    assert adapter.api_key == "relay-key"
    assert adapter.base_url == "https://api.bltcy.ai/v1"
