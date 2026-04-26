from core.config import Settings
from apps.api.runtime import _build_chart_vision_adapter
from adapters.local_runtime import LocalLLMAdapter
from adapters.llm.openai_relay_adapter import OpenAIRelayAdapter


def test_build_chart_vision_adapter_uses_openai_override() -> None:
    settings = Settings(
        llm_provider="dashscope",
        llm_model="qwen-plus",
        vision_model="qwen-vl-plus",
        dashscope_api_key="dashscope-key",
        chart_vision_provider="openai",
        chart_vision_model="gpt-4o",
        chart_vision_api_key="openai-key",
        chart_vision_base_url="https://api.bltcy.ai/",
        chart_vision_max_retries=2,
    )

    adapter = _build_chart_vision_adapter(settings, default_adapter=LocalLLMAdapter())

    assert isinstance(adapter, OpenAIRelayAdapter)
    assert adapter.model == "gpt-4o"
    assert adapter.vision_model == "gpt-4o"
    assert adapter.base_url == "https://api.bltcy.ai/v1"


def test_build_chart_vision_adapter_defaults_to_main_adapter() -> None:
    settings = Settings(llm_provider="dashscope", dashscope_api_key="dashscope-key", chart_vision_provider="dashscope")
    default = LocalLLMAdapter()

    adapter = _build_chart_vision_adapter(settings, default_adapter=default)

    assert adapter is default
