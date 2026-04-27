import logging

from adapters.local_runtime import (
    InMemoryGraphStore,
    InMemoryVectorStore,
    LocalDocumentParser,
    LocalHashEmbeddingAdapter,
    LocalLLMAdapter,
)
from adapters.embedding.dashscope_adapter import DashScopeEmbeddingAdapter
from adapters.embedding.openai import OpenAIEmbeddingAdapter
from adapters.graph_store.neo4j_adapter import Neo4jGraphStore
from adapters.llm import FallbackLLMAdapter, LangChainLLMAdapter, OpenAIRelayAdapter, build_provider_binding
from adapters.llm.dashscope_adapter import DashScopeLLMAdapter
from adapters.vector_store.milvus_adapter import MilvusVectorStore
from adapters.vector_store.pgvector_adapter import PgVectorStore
from core.config import Settings
from core.prompt_resolver import PromptResolver
from rag_runtime.checkpoint import build_checkpointer
from rag_runtime.memory import GraphSessionMemory, MySQLSessionMemoryStore
from rag_runtime.runtime import GraphRuntime, RagRuntime
from reasoning import CoTReasoningAgent, PlanAndSolveReasoningAgent, ReActReasoningAgent, ReasoningStrategySet
from retrieval.graph_retriever import GraphRetriever
from retrieval.graph_summary_retriever import GraphSummaryRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.cross_encoder import SentenceTransformersCrossEncoderReranker
from retrieval.cross_encoder import HeuristicFallbackReranker
from retrieval.sparse_retriever import SparseRetriever
from retrieval.vector_retriever import VectorRetriever
from rag_runtime.services.embedding_index_service import EmbeddingIndexService
from rag_runtime.services.graph_index_service import GraphIndexService
from skills.loader import SkillLoader
from skills.research import build_core_research_skill_profiles
from skills.registry import SkillRegistry
from tooling.executor import ToolExecutor
from tooling.registry import ToolRegistry
from tooling.schemas import (
    AnswerWithEvidenceToolInput,
    HybridRetrieveToolInput,
    IndexDocumentToolInput,
    ParseDocumentToolInput,
    QueryGraphSummaryToolInput,
    TOOL_OUTPUT_SCHEMAS,
    ToolSpec,
    UnderstandChartToolInput,
)
from tools.answer_toolkit import AnswerTools
from tools.chart_toolkit import ChartTools
from tools.document_toolkit import DocumentTools
from tools.graph_extraction_toolkit import GraphExtractionTools
from tools.retrieval_toolkit import RetrievalTools

logger = logging.getLogger(__name__)
_OPENAI_COMPATIBLE_PROVIDERS = {"openai", "openai_relay", "openai_compatible", "relay"}


def build_rag_runtime(settings: Settings) -> RagRuntime:
    llm_adapter = _build_llm_adapter(settings)
    chart_llm_adapter = _build_chart_vision_adapter(settings, default_adapter=llm_adapter)
    embedding_adapter = _build_embedding_adapter(settings)
    vector_store = _build_vector_store(settings, embedding_adapter)
    graph_store = _build_graph_store(settings)
    document_parser = LocalDocumentParser(storage_root=settings.resolve_path(settings.local_storage_root))

    vector_retriever = VectorRetriever(embedding_adapter, vector_store)
    sparse_retriever = SparseRetriever(vector_store)
    graph_retriever = GraphRetriever(graph_store)
    graph_summary_retriever = GraphSummaryRetriever()
    reranker = _build_reranker(settings)
    hybrid_retriever = HybridRetriever(
        graph_retriever=graph_retriever,
        vector_retriever=vector_retriever,
        sparse_retriever=sparse_retriever,
        graph_summary_retriever=graph_summary_retriever,
        reranker=reranker,
    )

    session_memory = _build_session_memory(settings)
    prompt_resolver = _build_prompt_resolver(settings)
    skill_registry = _build_skill_registry(settings)
    tool_registry = ToolRegistry()
    tool_executor = ToolExecutor(tool_registry)
    reasoning_strategies = ReasoningStrategySet(
        answer_synthesis=CoTReasoningAgent(llm_adapter=llm_adapter),
        query_planning=PlanAndSolveReasoningAgent(llm_adapter=llm_adapter),
        tool_reasoning=ReActReasoningAgent(
            llm_adapter=llm_adapter,
            tool_registry=tool_registry,
            tool_executor=tool_executor,
        ),
    )
    graph_runtime = RagRuntime(
        document_tools=DocumentTools(
            pdf_service=document_parser,
            ocr_service=document_parser,
            layout_service=document_parser,
            llm_adapter=llm_adapter,
        ),
        chart_tools=ChartTools(
            llm_adapter=chart_llm_adapter,
            vision_timeout_seconds=settings.chart_vision_timeout_seconds,
        ),
        graph_extraction_tools=GraphExtractionTools(llm_adapter=llm_adapter),
        retrieval_tools=RetrievalTools(hybrid_retriever),
        answer_tools=AnswerTools(
            llm_adapter=llm_adapter,
            prompt_resolver=prompt_resolver,
            reasoning_strategies=reasoning_strategies,
        ),
        graph_index_service=GraphIndexService(graph_store),
        embedding_index_service=EmbeddingIndexService(embedding_adapter, vector_store),
        checkpointer=build_checkpointer(),
        session_memory=session_memory,
        llm_adapter=llm_adapter,
        prompt_resolver=prompt_resolver,
        skill_registry=skill_registry,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        reasoning_strategies=reasoning_strategies,
        cot_reasoning_agent=None,
        plan_and_solve_reasoning_agent=None,
        react_reasoning_agent=None,
    )
    _register_runtime_tools(graph_runtime)
    graph_runtime.reasoning_strategies = reasoning_strategies
    graph_runtime.cot_reasoning_agent = reasoning_strategies.cot_reasoning_agent
    graph_runtime.plan_and_solve_reasoning_agent = reasoning_strategies.plan_and_solve_reasoning_agent
    graph_runtime.react_reasoning_agent = reasoning_strategies.react_reasoning_agent
    graph_runtime.answer_tools.reasoning_strategies = reasoning_strategies
    graph_runtime.answer_tools.cot_reasoning_agent = reasoning_strategies.cot_reasoning_agent
    logger.info(
        "Built RAG API runtime",
        extra={
            "runtime_backend": settings.runtime_backend,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "chart_vision_provider": settings.chart_vision_provider or settings.llm_provider,
            "chart_vision_model": settings.chart_vision_model or settings.vision_model or settings.llm_model,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
        },
    )
    return graph_runtime


def build_graph_runtime(settings: Settings) -> GraphRuntime:
    return build_rag_runtime(settings)


def _build_reranker(settings: Settings):
    policy = str(settings.reranker_unavailable_policy or "error").strip().lower()
    try:
        reranker = SentenceTransformersCrossEncoderReranker(
            model_name=settings.reranker_model,
            batch_size=settings.reranker_batch_size,
            max_length=settings.reranker_max_length,
            allow_download=settings.reranker_allow_download,
            cache_dir=settings.reranker_cache_dir,
        )
        logger.info(
            "Built Cross-Encoder reranker",
            extra={
                "reranker_model": settings.reranker_model,
                "reranker_policy": policy,
                "reranker_allow_download": settings.reranker_allow_download,
            },
        )
        return reranker
    except Exception as exc:
        if policy in {"heuristic", "fallback"}:
            logger.warning(
                "Cross-Encoder reranker unavailable; falling back to heuristic reranking: %s",
                exc,
            )
            return HeuristicFallbackReranker(reason=f"cross_encoder_unavailable:{exc.__class__.__name__}")
        raise RuntimeError(
            "Cross-Encoder reranker initialization failed. "
            "Install sentence-transformers/torch/transformers, make sure the reranker model is present locally, "
            "set RERANKER_ALLOW_DOWNLOAD=true to fetch it automatically, "
            "or set RERANKER_UNAVAILABLE_POLICY=heuristic to allow fallback."
        ) from exc


def _build_prompt_resolver(settings: Settings) -> PromptResolver:
    return PromptResolver(mapping_path=settings.resolve_path("prompts/skill_prompt_mapping.yaml"))


def _build_skill_registry(settings: Settings) -> SkillRegistry:
    registry = SkillRegistry()
    loader = SkillLoader(specs_dir=settings.resolve_path("skills/specs"))
    registry.register_many(loader.load_from_directory(), replace=True)
    registry.register_many(build_core_research_skill_profiles(), replace=True)
    return registry


def _register_runtime_tools(graph_runtime: GraphRuntime) -> None:
    graph_runtime.tool_registry.register_many(
        [
            ToolSpec(
                name="parse_document",
                description="Parse a document into a structured ParsedDocument.",
                input_schema=ParseDocumentToolInput,
                output_schema=TOOL_OUTPUT_SCHEMAS["parse_document"],
                handler=graph_runtime.handle_parse_document,
                tags=["document", "parse"],
            ),
            ToolSpec(
                name="index_document",
                description="Index a parsed document into graph and embedding stores.",
                input_schema=IndexDocumentToolInput,
                output_schema=TOOL_OUTPUT_SCHEMAS["index_document"],
                handler=graph_runtime.handle_index_document,
                tags=["document", "index"],
            ),
            ToolSpec(
                name="understand_chart",
                description="Analyze a chart image and return structured chart information.",
                input_schema=UnderstandChartToolInput,
                output_schema=TOOL_OUTPUT_SCHEMAS["understand_chart"],
                handler=graph_runtime.handle_understand_chart,
                tags=["chart", "vision"],
            ),
            ToolSpec(
                name="hybrid_retrieve",
                description="Retrieve vector, graph, and graph-summary evidence for a question.",
                input_schema=HybridRetrieveToolInput,
                output_schema=TOOL_OUTPUT_SCHEMAS["hybrid_retrieve"],
                handler=graph_runtime.retrieval_tools.tool_hybrid_retrieve,
                tags=["retrieval", "search"],
            ),
            ToolSpec(
                name="query_graph_summary",
                description="Query graph community summaries for a question.",
                input_schema=QueryGraphSummaryToolInput,
                output_schema=TOOL_OUTPUT_SCHEMAS["query_graph_summary"],
                handler=graph_runtime.query_graph_summary,
                tags=["retrieval", "graph", "summary"],
            ),
            ToolSpec(
                name="answer_with_evidence",
                description="Generate a grounded answer from the supplied evidence bundle.",
                input_schema=AnswerWithEvidenceToolInput,
                output_schema=TOOL_OUTPUT_SCHEMAS["answer_with_evidence"],
                handler=graph_runtime.answer_tools.answer_with_evidence,
                tags=["answer", "generation"],
            ),
        ],
        replace=True,
    )


def _build_session_memory(settings: Settings) -> GraphSessionMemory:
    provider = settings.session_memory_provider.lower()
    if provider in {"memory", "local", "inmemory"}:
        return GraphSessionMemory()
    if provider not in {"auto", "mysql"}:
        raise RuntimeError(f"Unsupported SESSION_MEMORY_PROVIDER: {settings.session_memory_provider}")
    if all([settings.mysql_host, settings.mysql_user, settings.mysql_database]):
        return GraphSessionMemory(
            MySQLSessionMemoryStore(
                host=settings.mysql_host or "localhost",
                port=settings.mysql_port,
                user=settings.mysql_user or "root",
                password=settings.mysql_password or "",
                database=settings.mysql_database or "ecs",
                charset=settings.mysql_charset,
                table_name=settings.mysql_session_memory_table,
            )
        )
    return GraphSessionMemory()


async def initialize_rag_runtime(graph_runtime: RagRuntime) -> None:
    vector_store = graph_runtime.embedding_index_service.vector_store
    ensure_schema = getattr(vector_store, "ensure_schema", None)
    if ensure_schema:
        await ensure_schema()
    connect = getattr(graph_runtime.graph_index_service.graph_store, "connect", None)
    if connect:
        await connect()


async def close_rag_runtime(graph_runtime: RagRuntime) -> None:
    vector_store = graph_runtime.embedding_index_service.vector_store
    close_vector = getattr(vector_store, "close", None)
    if close_vector:
        await close_vector()
    close_graph = getattr(graph_runtime.graph_index_service.graph_store, "close", None)
    if close_graph:
        await close_graph()


async def initialize_graph_runtime(graph_runtime: GraphRuntime) -> None:
    await initialize_rag_runtime(graph_runtime)


async def close_graph_runtime(graph_runtime: GraphRuntime) -> None:
    await close_rag_runtime(graph_runtime)


def _build_llm_adapter(settings: Settings):
    primary = _build_primary_llm_adapter(settings)
    fallbacks = _build_fallback_adapters(settings)
    if fallbacks:
        logger.info("LLM fallback chain configured: %d fallback(s)", len(fallbacks))
        return FallbackLLMAdapter(primary, fallbacks)
    return primary


def _build_primary_llm_adapter(settings: Settings):
    provider = settings.llm_provider.lower()
    if provider == "dashscope":
        if not settings.dashscope_api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required when LLM_PROVIDER=dashscope")
        provider_binding = _build_provider_binding(settings)
        if provider_binding is not None and provider_binding.chat_model is not None:
            return LangChainLLMAdapter(provider_binding=provider_binding)
        return DashScopeLLMAdapter(
            api_key=settings.dashscope_api_key,
            model=settings.llm_model,
            vision_model=settings.vision_model,
            base_url=settings.dashscope_base_url,
            timeout_seconds=settings.dashscope_timeout_seconds,
            max_retries=settings.dashscope_max_retries,
            retry_delay_seconds=settings.dashscope_retry_delay_seconds,
        )
    if provider in _OPENAI_COMPATIBLE_PROVIDERS:
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when LLM_PROVIDER uses an OpenAI-compatible relay"
            )
        return OpenAIRelayAdapter(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            vision_model=settings.vision_model,
            base_url=settings.openai_base_url or "https://gpt-agent.cc/v1/chat/completions",
            timeout_seconds=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
            retry_delay_seconds=0.5,
        )
    provider_binding = _build_provider_binding(settings)
    if provider_binding is not None and provider_binding.chat_model is not None:
        return LangChainLLMAdapter(provider_binding=provider_binding)
    if settings.runtime_backend.lower() == "business":
        raise RuntimeError(f"Unsupported business LLM_PROVIDER: {settings.llm_provider}")
    return LocalLLMAdapter()


def _build_fallback_adapters(settings: Settings) -> list:
    """Build fallback adapters from comma-separated config fields."""
    providers = [p.strip() for p in settings.llm_fallback_providers.split(",") if p.strip()]
    if not providers:
        return []
    models = [m.strip() for m in settings.llm_fallback_models.split(",")]
    api_keys = [k.strip() for k in settings.llm_fallback_api_keys.split(",")]
    base_urls = [u.strip() for u in settings.llm_fallback_base_urls.split(",")]
    adapters = []
    for i, prov in enumerate(providers):
        model = models[i] if i < len(models) and models[i] else settings.llm_model
        api_key = api_keys[i] if i < len(api_keys) and api_keys[i] else settings.openai_api_key
        base_url = base_urls[i] if i < len(base_urls) and base_urls[i] else None
        if not api_key:
            logger.warning("Skipping fallback provider %s — no API key", prov)
            continue
        prov_lower = prov.lower()
        if prov_lower in _OPENAI_COMPATIBLE_PROVIDERS:
            adapters.append(OpenAIRelayAdapter(
                api_key=api_key,
                model=model,
                base_url=base_url or "https://api.openai.com/v1/chat/completions",
                timeout_seconds=settings.openai_timeout_seconds,
                max_retries=settings.openai_max_retries,
                retry_delay_seconds=0.5,
            ))
        elif prov_lower == "dashscope":
            adapters.append(DashScopeLLMAdapter(
                api_key=api_key,
                model=model,
                base_url=base_url or settings.dashscope_base_url,
                timeout_seconds=settings.dashscope_timeout_seconds,
                max_retries=settings.dashscope_max_retries,
                retry_delay_seconds=settings.dashscope_retry_delay_seconds,
            ))
        else:
            logger.warning("Unsupported fallback provider: %s", prov)
    return adapters


def _build_provider_binding(settings: Settings):
    try:
        return build_provider_binding(settings)
    except Exception:
        if settings.runtime_backend.lower() == "business" and settings.llm_provider.lower() != "local":
            raise
        logger.info(
            "Falling back to legacy adapter-only runtime for LLM provider binding",
            extra={"llm_provider": settings.llm_provider},
        )
        return None


def _build_chart_vision_adapter(settings: Settings, default_adapter):
    provider = (settings.chart_vision_provider or settings.llm_provider).lower()
    if provider in _OPENAI_COMPATIBLE_PROVIDERS:
        api_key = settings.chart_vision_api_key or settings.openai_api_key
        if not api_key:
            raise RuntimeError("CHART_VISION_API_KEY or OPENAI_API_KEY is required when chart vision uses openai")
        return OpenAIRelayAdapter(
            api_key=api_key,
            model=settings.chart_vision_model or settings.vision_model or settings.llm_model,
            vision_model=settings.chart_vision_model or settings.vision_model or settings.llm_model,
            base_url=settings.chart_vision_base_url or settings.openai_base_url or "https://api.bltcy.ai/",
            timeout_seconds=settings.openai_timeout_seconds,
            max_retries=settings.chart_vision_max_retries or settings.openai_max_retries,
            retry_delay_seconds=0.5,
        )
    return default_adapter


def _build_embedding_adapter(settings: Settings):
    provider = settings.embedding_provider.lower()
    if provider == "dashscope":
        api_key = settings.embedding_api_key or settings.dashscope_api_key
        if not api_key:
            raise RuntimeError("EMBEDDING_API_KEY or DASHSCOPE_API_KEY is required when EMBEDDING_PROVIDER=dashscope")
        return DashScopeEmbeddingAdapter(
            api_key=api_key,
            text_model=settings.embedding_model,
            text_batch_size=settings.embedding_text_batch_size,
            base_url=settings.embedding_base_url or settings.dashscope_base_url,
            timeout_seconds=settings.dashscope_timeout_seconds,
            max_retries=settings.dashscope_max_retries,
            retry_delay_seconds=settings.dashscope_retry_delay_seconds,
        )
    if provider in _OPENAI_COMPATIBLE_PROVIDERS:
        api_key = settings.embedding_api_key or settings.openai_api_key or settings.dashscope_api_key
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY or DASHSCOPE_API_KEY is required when EMBEDDING_PROVIDER uses an OpenAI-compatible relay"
            )
        return OpenAIEmbeddingAdapter(
            api_key=api_key,
            model=settings.embedding_model,
            text_batch_size=settings.embedding_text_batch_size,
            base_url=settings.embedding_base_url or settings.openai_base_url or settings.dashscope_base_url or "https://api.openai.com/v1",
            timeout_seconds=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
            retry_delay_seconds=0.5,
        )
    if settings.runtime_backend.lower() == "business":
        raise RuntimeError(f"Unsupported business EMBEDDING_PROVIDER: {settings.embedding_provider}")
    return LocalHashEmbeddingAdapter(model=settings.embedding_model)


def _build_vector_store(settings: Settings, embedding_adapter):
    provider = settings.vector_store_provider.lower()
    if provider in {"milvus", "zilliz"}:
        return MilvusVectorStore(
            collection_name=settings.milvus_collection_name,
            uri=settings.milvus_uri,
            token=settings.milvus_token,
            db_name=settings.milvus_db_name,
            dimension=settings.milvus_dimension,
            metric_type=settings.milvus_metric_type,
            index_type=settings.milvus_index_type,
            embedding_adapter=embedding_adapter,
        )
    if provider == "pgvector":
        if not settings.postgres_dsn:
            raise RuntimeError("POSTGRES_DSN is required when VECTOR_STORE_PROVIDER=pgvector")
        return PgVectorStore(
            dsn=settings.postgres_dsn,
            table_name=settings.pgvector_table,
            embedding_adapter=embedding_adapter,
        )
    if provider in {"memory", "local"} and settings.runtime_backend.lower() != "business":
        return InMemoryVectorStore()
    raise RuntimeError(f"Unsupported VECTOR_STORE_PROVIDER: {settings.vector_store_provider}")


def _build_graph_store(settings: Settings):
    provider = settings.graph_store_provider.lower()
    if provider == "neo4j":
        if not settings.neo4j_uri or not settings.neo4j_user or not settings.neo4j_password:
            raise RuntimeError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD are required")
        return Neo4jGraphStore(
            uri=settings.neo4j_uri,
            username=settings.neo4j_user,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
    if provider in {"memory", "local"} and settings.runtime_backend.lower() != "business":
        return InMemoryGraphStore()
    raise RuntimeError(f"Unsupported GRAPH_STORE_PROVIDER: {settings.graph_store_provider}")
