from adapters.embedding.dashscope_adapter import DashScopeEmbeddingAdapter
from adapters.llm.provider_binding import normalize_openai_base_url


class OpenAIEmbeddingAdapter(DashScopeEmbeddingAdapter):
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-3-small",
        multimodal_model: str | None = None,
        text_batch_size: int = 16,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 120.0,
        connect_timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        normalized_base_url = normalize_openai_base_url(base_url) or base_url.rstrip("/")
        super().__init__(
            api_key=api_key,
            text_model=model,
            multimodal_model=multimodal_model,
            text_batch_size=text_batch_size,
            base_url=normalized_base_url,
            timeout_seconds=timeout_seconds,
            connect_timeout_seconds=connect_timeout_seconds,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
        )
