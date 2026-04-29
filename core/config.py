from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    app_name: str = "Research-Copilot"
    app_env: str = "local"
    log_level: str = "INFO"
    runtime_backend: str = "local"
    api_key_enabled: bool = False
    api_key: str | None = None
    audit_log_enabled: bool = True

    dashscope_api_key: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_timeout_seconds: float = 60.0
    openai_max_retries: int = 2
    google_api_key: str | None = None
    llm_provider: str = "local"
    llm_model: str = "qwen-plus"
    llm_fallback_providers: str = ""
    llm_fallback_models: str = ""
    llm_fallback_api_keys: str = ""
    llm_fallback_base_urls: str = ""
    embedding_provider: str = "local"
    embedding_model: str = "local-hash-embedding"
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_text_batch_size: int = 16
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = 16
    reranker_max_length: int = 512
    reranker_allow_download: bool = False
    reranker_cache_dir: str | None = None
    reranker_unavailable_policy: str = "error"
    vision_model: str | None = None
    chart_vision_provider: str | None = None
    chart_vision_model: str | None = None
    chart_vision_api_key: str | None = None
    chart_vision_base_url: str | None = None
    chart_vision_max_retries: int | None = None
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_timeout_seconds: float = 60.0
    chart_vision_timeout_seconds: float = 180.0
    dashscope_max_retries: int = 2
    dashscope_retry_delay_seconds: float = 2.0
    cors_allow_origins: str = "http://127.0.0.1:3000,http://localhost:3000,http://127.0.0.1:3001,http://localhost:3001"
    rate_limit_max_requests: int = 60
    rate_limit_window_seconds: int = 60
    json_log_format: bool = False
    storage_provider: str = "json"
    research_storage_root: str = ".data/research"
    research_sqlite_db_path: str = ".data/research/kepler.db"
    research_reset_on_startup: bool = False
    research_http_timeout_seconds: float = 10.0
    research_import_concurrency: int = 2
    research_cli_poll_initial_seconds: float = 1.5
    research_cli_poll_steady_seconds: float = 3.0
    research_cli_heartbeat_seconds: float = 10.0
    research_state_cache_ttl_seconds: float = 2.0
    research_trajectory_cache_ttl_seconds: float = 3.0
    research_default_days_back: int = 30
    research_default_max_papers: int = 5
    research_default_ranking_mode: str = "heuristic"
    research_contact_email: str | None = None
    arxiv_api_base_url: str = "https://export.arxiv.org/api/query"
    openalex_api_base_url: str = "https://api.openalex.org"
    semantic_scholar_api_base_url: str = "https://api.semanticscholar.org/graph/v1"
    semantic_scholar_api_key: str | None = None
    ieee_api_base_url: str = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
    ieee_api_key: str | None = None
    zotero_api_base_url: str = "https://api.zotero.org"
    zotero_api_key: str | None = None
    zotero_library_type: str | None = None
    zotero_library_id: str | None = None
    zotero_local_enabled: bool = False
    zotero_local_base_url: str = "http://127.0.0.1:23119"
    zotero_local_user_id: str = "0"
    zotero_local_timeout_seconds: float = 20.0

    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None
    neo4j_database: str | None = None

    vector_store_provider: str = "milvus"
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str | None = None
    milvus_db_name: str | None = None
    milvus_collection_name: str = "multimodal_embeddings"
    milvus_dimension: int | None = None
    milvus_metric_type: str = "COSINE"
    milvus_index_type: str = "HNSW"
    graph_store_provider: str = "memory"

    session_memory_provider: str = "sqlite"
    research_working_memory_turns: int = 10
    research_session_memory_dir: str = ".data/research/session_memory"
    research_long_term_memory_dir: str = ".data/research/long_term_memory"
    research_paper_knowledge_dir: str = ".data/research/paper_knowledge"
    long_term_memory_provider: str = "sqlite"
    long_term_memory_max_records: int = 5000
    local_code_execution_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("LOCAL_CODE_EXECUTION_ENABLED", "MCP_CODE_EXECUTION_ENABLED"),
    )

    mcp_servers: dict = Field(default_factory=dict)
    knowledge_dir: str = ".data/knowledge"

    # ── 检索配置 ──
    retrieval_top_k: int = 10
    retrieval_mode: str = "hybrid"
    graph_query_mode: str = "auto"
    enable_graph_summary: bool = True

    # ── 回答配置 ──
    answer_language: str = "zh-CN"
    answer_detail_level: str = "normal"
    answer_tone: str = "factual"

    # ── Prompt 路径 ──
    answer_prompt_path: str = "prompts/document/answer_question_with_hybrid_rag.txt"
    rewrite_prompt_path: str = "prompts/retrieval/rewrite_query.txt"

    # ── MCP 白名单 ──
    mcp_allowed_tools: str = ""

    local_storage_root: str = ".data/storage"
    upload_dir: str = ".data/uploads"
    upload_max_bytes: int = 25 * 1024 * 1024

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        populate_by_name=True,
        extra="ignore",
    )

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    def resolve_path(self, path_value: str) -> Path:
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (self.project_root / path).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
