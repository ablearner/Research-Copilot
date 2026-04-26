from adapters.llm.openai_relay_adapter import OpenAIRelayAdapter
from apps.api.runtime import _build_llm_adapter
from core.config import Settings


def test_build_llm_adapter_uses_openai_relay_for_openai_compatible_provider() -> None:
    settings = Settings(
        llm_provider="openai_compatible",
        llm_model="claude-opus-4-6",
        openai_api_key="openai-key",
        openai_base_url="https://gpt-agent.cc/v1",
    )

    adapter = _build_llm_adapter(settings)

    assert isinstance(adapter, OpenAIRelayAdapter)
    assert adapter.model == "claude-opus-4-6"
    assert adapter.base_url == "https://gpt-agent.cc/v1"
