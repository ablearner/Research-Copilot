from __future__ import annotations

import socket
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from core.config import Settings
from domain.schemas.research import (
    CreateResearchConversationRequest,
    ImportPapersRequest,
    ResearchTaskAskRequest,
    ResearchAgentRunRequest,
    ResearchAgentRunResponse,
    ResearchConversationResponse,
    ResearchMessage,
)
from memory.long_term_memory import (
    InMemoryLongTermMemoryStore,
    JsonLongTermMemoryStore,
    LongTermMemory,
    SQLiteLongTermMemoryStore,
)
from memory.memory_manager import MemoryManager
from memory.paper_knowledge_memory import JsonPaperKnowledgeStore, PaperKnowledgeMemory
from memory.session_memory import JsonSessionMemoryStore, SessionMemory
from services.research.literature_research_service import LiteratureResearchService
from services.research.paper_import_service import PaperImportService
from services.research.paper_search_service import PaperSearchService
from services.research.research_report_service import ResearchReportService
from sdk.runtime_profile import RuntimeProfile, RuntimeProfileStore
from tools.research.arxiv_search_tool import ArxivSearchTool
from tools.research.ieee_metadata_search_tool import IEEEMetadataSearchTool
from tools.research.openalex_search_tool import OpenAlexSearchTool
from tools.research.semantic_scholar_search_tool import SemanticScholarSearchTool


PLUGIN_CATALOG: dict[str, dict[str, Any]] = {
    "academic_search": {
        "description": "Enable the built-in multi-source academic search stack for arXiv/OpenAlex/Semantic Scholar/IEEE.",
        "default_enabled": True,
    },
    "zotero_local_mcp": {
        "description": "Enable local Zotero MCP integration for library-backed paper discovery.",
        "default_enabled": False,
        "setting_overrides": {"zotero_local_enabled": True},
    },
    "local_code_execution": {
        "description": "Allow research runtime code execution tools when local execution is configured.",
        "default_enabled": False,
        "setting_overrides": {"local_code_execution_enabled": True},
    },
    "trajectory_inspector": {
        "description": "Expose trajectory, recent events, and context-compression inspection commands in the CLI.",
        "default_enabled": True,
    },
    "terminal_agent": {
        "description": "Enable interactive terminal agent mode with conversation-oriented status panels.",
        "default_enabled": True,
    },
}


def _preferred_answer_language_from_text(text: str | None) -> str:
    value = str(text or "").strip()
    if not value:
        return "zh-CN"
    cjk_count = sum(1 for char in value if "\u4e00" <= char <= "\u9fff")
    latin_count = sum(1 for char in value if ("a" <= char.lower() <= "z"))
    if cjk_count > 0 and cjk_count >= max(1, latin_count // 2):
        return "zh-CN"
    return "en-US"


def _build_long_term_memory(settings: Settings) -> LongTermMemory:
    provider = str(settings.long_term_memory_provider or "json").strip().lower()
    if provider in {"json", "file"}:
        return LongTermMemory(
            JsonLongTermMemoryStore(
                settings.resolve_path(settings.research_long_term_memory_dir)
            )
        )
    if provider == "sqlite":
        return LongTermMemory(
            SQLiteLongTermMemoryStore(
                db_path=settings.resolve_path(settings.research_sqlite_db_path),
                max_records=settings.long_term_memory_max_records,
            )
        )
    return LongTermMemory(InMemoryLongTermMemoryStore())


def _can_connect(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class ResearchCopilotSDK:
    def __init__(
        self,
        *,
        settings: Settings,
        service: LiteratureResearchService,
        profile_store: RuntimeProfileStore,
        profile: RuntimeProfile,
    ) -> None:
        self.settings = settings
        self.service = service
        self.profile_store = profile_store
        self.profile = profile
        self._graph_runtime: Any | None = None
        self._agent_service: LiteratureResearchService | None = None
        self._state_cache: OrderedDict[tuple[str, bool], tuple[float, dict[str, Any]]] = OrderedDict()
        self._trajectory_cache: OrderedDict[tuple[str, int, int], tuple[float, dict[str, Any]]] = OrderedDict()

    def _invalidate_conversation_cache(self, conversation_id: str) -> None:
        self._state_cache = OrderedDict(
            (key, value) for key, value in self._state_cache.items() if key[0] != conversation_id
        )
        self._trajectory_cache = OrderedDict(
            (key, value) for key, value in self._trajectory_cache.items() if key[0] != conversation_id
        )

    def _prune_cache(
        self,
        cache: OrderedDict[Any, tuple[float, Any]],
        *,
        ttl_seconds: float,
        max_entries: int = 32,
    ) -> None:
        now = time.monotonic()
        expired = [key for key, (created_at, _) in cache.items() if now - created_at >= ttl_seconds]
        for key in expired:
            cache.pop(key, None)
        while len(cache) > max_entries:
            cache.popitem(last=False)

    def _cache_get(
        self,
        cache: OrderedDict[Any, tuple[float, Any]],
        key: Any,
        *,
        ttl_seconds: float,
    ) -> Any | None:
        self._prune_cache(cache, ttl_seconds=ttl_seconds)
        item = cache.get(key)
        if item is None:
            return None
        created_at, value = item
        if time.monotonic() - created_at >= ttl_seconds:
            cache.pop(key, None)
            return None
        cache.move_to_end(key)
        return value

    def _cache_put(
        self,
        cache: OrderedDict[Any, tuple[float, Any]],
        key: Any,
        value: Any,
        *,
        ttl_seconds: float,
    ) -> Any:
        cache[key] = (time.monotonic(), value)
        cache.move_to_end(key)
        self._prune_cache(cache, ttl_seconds=ttl_seconds)
        return value

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "ResearchCopilotSDK":
        settings = settings or Settings()
        storage_root = settings.resolve_path(settings.research_storage_root)
        profile_store = RuntimeProfileStore(storage_root / "cli" / "runtime_profile.json")
        profile = profile_store.load()
        report_service = ResearchReportService(storage_root)
        search_service = PaperSearchService(
            arxiv_tool=ArxivSearchTool(
                base_url=settings.arxiv_api_base_url,
                app_name=settings.app_name,
                timeout_seconds=settings.research_http_timeout_seconds,
            ),
            openalex_tool=OpenAlexSearchTool(
                base_url=settings.openalex_api_base_url,
                timeout_seconds=settings.research_http_timeout_seconds,
                mailto=settings.research_contact_email,
            ),
            semantic_scholar_tool=SemanticScholarSearchTool(
                base_url=settings.semantic_scholar_api_base_url,
                timeout_seconds=settings.research_http_timeout_seconds,
                api_key=settings.semantic_scholar_api_key,
            ),
            ieee_tool=IEEEMetadataSearchTool(
                base_url=settings.ieee_api_base_url,
                timeout_seconds=settings.research_http_timeout_seconds,
                api_key=settings.ieee_api_key,
            ),
            ranking_mode=settings.research_default_ranking_mode,
        )
        memory_manager = MemoryManager(
            session_memory=SessionMemory(
                JsonSessionMemoryStore(settings.resolve_path(settings.research_session_memory_dir))
            ),
            long_term_memory=_build_long_term_memory(settings),
            paper_knowledge_memory=PaperKnowledgeMemory(
                JsonPaperKnowledgeStore(settings.resolve_path(settings.research_paper_knowledge_dir))
            ),
        )
        service = LiteratureResearchService(
            paper_search_service=search_service,
            report_service=report_service,
            paper_import_service=PaperImportService(
                upload_dir=settings.resolve_path(settings.upload_dir),
                timeout_seconds=settings.research_http_timeout_seconds,
            ),
            memory_manager=memory_manager,
            import_concurrency=settings.research_import_concurrency,
        )
        return cls(settings=settings, service=service, profile_store=profile_store, profile=profile)

    def list_conversations(self):
        return self.service.list_conversations()

    def load_user_profile(self):
        return self.service.memory_manager.load_user_profile()

    def update_user_profile(
        self,
        *,
        topic: str | None = None,
        sources: list[str] | None = None,
        keywords: list[str] | None = None,
        reasoning_style: str | None = None,
        answer_language: str | None = None,
        note: str | None = None,
    ):
        profile = self.service.memory_manager.update_user_profile(
            topic=topic,
            sources=sources,
            keywords=keywords,
            reasoning_style=reasoning_style,
            answer_language=answer_language,
            note=note,
        )
        if self._agent_service is not None:
            self._agent_service.memory_manager = self.service.memory_manager
        return profile

    def clear_user_profile(self):
        profile = self.service.memory_manager.clear_user_profile()
        if self._agent_service is not None:
            self._agent_service.memory_manager = self.service.memory_manager
        return profile

    def remove_user_profile_topics(self, topics: list[str]):
        profile = self.service.memory_manager.remove_user_profile_topics(topics=topics)
        if self._agent_service is not None:
            self._agent_service.memory_manager = self.service.memory_manager
        return profile

    def create_conversation(
        self,
        *,
        title: str | None = None,
        topic: str | None = None,
        days_back: int | None = None,
        max_papers: int | None = None,
        sources: list[str] | None = None,
    ) -> ResearchConversationResponse:
        return self.service.create_conversation(
            CreateResearchConversationRequest(
                title=title,
                topic=topic,
                days_back=days_back if days_back is not None else self.settings.research_default_days_back,
                max_papers=max_papers if max_papers is not None else self.settings.research_default_max_papers,
                sources=sources or ["arxiv", "openalex"],
            )
        )

    def get_conversation(self, conversation_id: str) -> ResearchConversationResponse:
        return self.service.get_conversation(conversation_id)

    def clear_conversation_memory(self, conversation_id: str) -> None:
        self.service.delete_conversation(conversation_id)
        self._invalidate_conversation_cache(conversation_id)
        graph_session_memory = getattr(self._graph_runtime, "session_memory", None)
        if graph_session_memory is not None and hasattr(graph_session_memory, "clear"):
            graph_session_memory.clear(conversation_id)

    def conversation_state(
        self,
        conversation_id: str,
        *,
        include_papers: bool = True,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        cache_key = (conversation_id, include_papers)
        ttl_seconds = max(float(self.settings.research_state_cache_ttl_seconds), 0.0)
        if use_cache and ttl_seconds > 0:
            cached = self._cache_get(self._state_cache, cache_key, ttl_seconds=ttl_seconds)
            if cached is not None:
                return cached
        response = self.get_conversation(conversation_id)
        conversation = response.conversation
        snapshot = conversation.snapshot
        task = snapshot.task_result.task if snapshot.task_result is not None else None
        task_id = task.task_id if task is not None else conversation.task_id
        papers = []
        if include_papers:
            papers = self.service.report_service.load_papers(task_id) if task_id else []
            if not papers and snapshot.task_result is not None:
                papers = list(snapshot.task_result.papers)
        payload = {
            "conversation": conversation,
            "messages": response.messages,
            "task": task,
            "task_id": task_id,
            "papers": papers,
            "selected_paper_ids": list(snapshot.selected_paper_ids),
            "must_read_paper_ids": list(snapshot.workspace.must_read_paper_ids),
            "ingest_candidate_ids": list(snapshot.workspace.ingest_candidate_ids),
            "sources": list(snapshot.sources),
        }
        if use_cache and ttl_seconds > 0:
            return self._cache_put(self._state_cache, cache_key, payload, ttl_seconds=ttl_seconds)
        return payload

    def list_candidate_papers(self, conversation_id: str) -> list[dict[str, Any]]:
        state = self.conversation_state(conversation_id, include_papers=True)
        selected = set(state["selected_paper_ids"])
        must_read = set(state["must_read_paper_ids"])
        ingest = set(state["ingest_candidate_ids"])
        rows: list[dict[str, Any]] = []
        for index, paper in enumerate(state["papers"], start=1):
            rows.append(
                {
                    "index": index,
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "source": paper.source,
                    "year": paper.year,
                    "selected": paper.paper_id in selected,
                    "must_read": paper.paper_id in must_read,
                    "ingest_candidate": paper.paper_id in ingest,
                    "ingest_status": paper.ingest_status,
                    "document_id": paper.metadata.get("document_id"),
                }
            )
        return rows

    def resolve_paper_selection(self, conversation_id: str, selectors: list[str]) -> list[str]:
        state = self.conversation_state(conversation_id, include_papers=True)
        papers = state["papers"]
        by_id = {paper.paper_id: paper.paper_id for paper in papers}
        must_read = list(state["must_read_paper_ids"])
        ingest = list(state["ingest_candidate_ids"])
        selected = list(state["selected_paper_ids"])
        if not selectors:
            return selected
        resolved: list[str] = []
        for selector in selectors:
            value = str(selector).strip()
            if not value:
                continue
            lowered = value.lower()
            if lowered == "all":
                resolved.extend([paper.paper_id for paper in papers])
                continue
            if lowered in {"selected", "current"}:
                resolved.extend(selected)
                continue
            if lowered in {"mustread", "must-read"}:
                resolved.extend(must_read)
                continue
            if lowered in {"ingest", "candidates"}:
                resolved.extend(ingest)
                continue
            if value.isdigit():
                index = int(value)
                if 1 <= index <= len(papers):
                    resolved.append(papers[index - 1].paper_id)
                continue
            if value in by_id:
                resolved.append(value)
        return list(dict.fromkeys(resolved))

    def list_jobs(self, *, conversation_id: str | None = None, task_id: str | None = None):
        return self.service.report_service.list_jobs(conversation_id=conversation_id, task_id=task_id)

    def doctor(self) -> dict[str, Any]:
        settings = self._effective_settings()
        checks = {
            "research_storage_root": settings.resolve_path(settings.research_storage_root).exists(),
            "upload_dir_parent": settings.resolve_path(settings.upload_dir).parent.exists(),
            "neo4j_bolt": _can_connect("127.0.0.1", 7687) if settings.neo4j_uri else False,
            "milvus_http": _can_connect("127.0.0.1", 19530) if settings.milvus_uri else False,
            "llm_api_key_present": bool(settings.dashscope_api_key or settings.openai_api_key),
        }
        return {
            "app_env": settings.app_env,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "checks": checks,
            "all_required_passed": checks["research_storage_root"] and checks["upload_dir_parent"],
        }

    def describe_runtime(self) -> dict[str, Any]:
        settings = self._effective_settings()
        return {
            "app_name": settings.app_name,
            "app_env": settings.app_env,
            "storage_root": str(settings.resolve_path(settings.research_storage_root)),
            "llm": {"provider": settings.llm_provider, "model": settings.llm_model},
            "embedding": {"provider": settings.embedding_provider, "model": settings.embedding_model},
            "chart_vision": {
                "provider": settings.chart_vision_provider or settings.llm_provider,
                "model": settings.chart_vision_model or settings.vision_model or settings.llm_model,
            },
            "memory": {
                "session_provider": settings.session_memory_provider,
                "long_term_provider": settings.long_term_memory_provider,
            },
            "vector_store_provider": settings.vector_store_provider,
            "graph_store_provider": settings.graph_store_provider,
            "plugins": self.list_plugins(),
        }

    def conversation_trajectory(
        self,
        conversation_id: str,
        *,
        message_limit: int = 12,
        event_limit: int = 12,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        cache_key = (conversation_id, message_limit, event_limit)
        ttl_seconds = max(float(self.settings.research_trajectory_cache_ttl_seconds), 0.0)
        if use_cache and ttl_seconds > 0:
            cached = self._cache_get(self._trajectory_cache, cache_key, ttl_seconds=ttl_seconds)
            if cached is not None:
                return cached
        conversation = self.service.report_service.load_conversation(conversation_id)
        if conversation is None:
            raise KeyError(conversation_id)
        messages = self.service.report_service.load_messages(conversation_id)
        payload = {
            "conversation": conversation.model_dump(mode="json"),
            "context_summary": conversation.snapshot.context_summary.model_dump(mode="json"),
            "recent_events": [
                event.model_dump(mode="json") for event in conversation.snapshot.recent_events[-event_limit:]
            ],
            "messages": [message.model_dump(mode="json") for message in messages[-message_limit:]],
        }
        if use_cache and ttl_seconds > 0:
            return self._cache_put(self._trajectory_cache, cache_key, payload, ttl_seconds=ttl_seconds)
        return payload

    def list_plugins(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name, metadata in PLUGIN_CATALOG.items():
            rows.append(
                {
                    "name": name,
                    "enabled": self._plugin_enabled(name),
                    "description": metadata["description"],
                }
            )
        return rows

    def set_plugin_enabled(self, name: str, enabled: bool) -> dict[str, Any]:
        if name not in PLUGIN_CATALOG:
            raise KeyError(name)
        self.profile = self.profile_store.set_plugin_enabled(name, enabled)
        self._graph_runtime = None
        self._agent_service = None
        return {"name": name, "enabled": enabled}

    def update_model_profile(self, **updates: str | None) -> dict[str, Any]:
        self._write_model_settings_to_env(**updates)
        self.settings = Settings()
        self.profile = self.profile_store.clear_models()
        self._graph_runtime = None
        self._agent_service = None
        return self.describe_runtime()["llm"] | {"embedding": self.describe_runtime()["embedding"]}

    async def run_agent_message(
        self,
        *,
        message: str,
        conversation_id: str | None = None,
        mode: str = "auto",
        task_id: str | None = None,
        topic: str | None = None,
        days_back: int = 180,
        max_papers: int = 12,
        sources: list[str] | None = None,
        skill_name: str | None = None,
        selected_paper_ids: list[str] | None = None,
        selected_document_ids: list[str] | None = None,
    ) -> tuple[ResearchAgentRunResponse, ResearchConversationResponse | None, int]:
        agent_service = self._ensure_agent_service()
        resolved_conversation_id = conversation_id
        if resolved_conversation_id is None:
            conversation = self.create_conversation(
                title=topic or message,
                topic=topic,
                days_back=days_back,
                max_papers=max_papers,
                sources=sources or ["arxiv", "openalex"],
            )
            resolved_conversation_id = conversation.conversation.conversation_id
        prior_message_count = 0
        if resolved_conversation_id:
            conversation_before = self.service.report_service.load_conversation(resolved_conversation_id)
            if conversation_before is not None:
                prior_message_count = conversation_before.message_count
        request = ResearchAgentRunRequest(
            message=message,
            mode=mode,
            task_id=task_id,
            conversation_id=resolved_conversation_id,
            days_back=days_back,
            max_papers=max_papers,
            sources=sources or ["arxiv", "openalex", "semantic_scholar"],
            selected_paper_ids=selected_paper_ids or [],
            selected_document_ids=selected_document_ids or [],
            skill_name=skill_name or "research_report",
            metadata={
                "source": "sdk_terminal_agent",
                "user_message": message,
                "context": {
                    "answer_language": _preferred_answer_language_from_text(message),
                    "active_paper_ids": (
                        conversation_before.snapshot.active_paper_ids
                        if resolved_conversation_id and conversation_before is not None
                        else []
                    ),
                },
            },
        )
        self.update_user_profile(
            topic=topic,
            answer_language=_preferred_answer_language_from_text(message),
        )
        response = await agent_service.run_agent(request, graph_runtime=self._graph_runtime)
        conversation_response = None
        if resolved_conversation_id:
            conversation_response = agent_service.record_agent_turn(
                resolved_conversation_id,
                request=request,
                response=response,
            )
            self._invalidate_conversation_cache(resolved_conversation_id)
        return response, conversation_response, prior_message_count

    async def import_papers_for_conversation(
        self,
        *,
        conversation_id: str,
        paper_ids: list[str],
        include_graph: bool = True,
        include_embeddings: bool = True,
        fast_mode: bool = True,
        skill_name: str | None = None,
    ):
        agent_service = self._ensure_agent_service()
        state = self.conversation_state(
            conversation_id,
            include_papers=False,
            use_cache=False,
        )
        task_id = state["task_id"]
        if not task_id:
            raise ValueError("No active research task exists for this conversation.")
        response = await agent_service.import_papers(
            ImportPapersRequest(
                task_id=task_id,
                paper_ids=paper_ids,
                papers=[],
                include_graph=include_graph,
                include_embeddings=include_embeddings,
                fast_mode=fast_mode,
                skill_name=skill_name or "research_report",
                conversation_id=conversation_id,
            ),
            graph_runtime=self._graph_runtime,
        )
        task_response = None
        if task_id:
            try:
                task_response = agent_service.get_task(task_id)
            except KeyError:
                task_response = None
        agent_service.record_import_turn(
            conversation_id,
            task_response=task_response,
            import_response=response,
            selected_paper_ids=paper_ids,
        )
        self._invalidate_conversation_cache(conversation_id)
        return response

    async def ask_conversation_collection(
        self,
        *,
        conversation_id: str,
        question: str,
        paper_ids: list[str] | None = None,
        document_ids: list[str] | None = None,
        top_k: int = 10,
        skill_name: str | None = None,
        reasoning_style: str | None = None,
    ):
        agent_service = self._ensure_agent_service()
        state = self.conversation_state(
            conversation_id,
            include_papers=False,
            use_cache=False,
        )
        task_id = state["task_id"]
        if not task_id:
            raise ValueError("No active research task exists for this conversation.")
        return await agent_service.ask_task_collection(
            task_id,
            ResearchTaskAskRequest(
                question=question,
                top_k=top_k,
                paper_ids=paper_ids or [],
                document_ids=document_ids or [],
                conversation_id=conversation_id,
                skill_name=skill_name or "research_report",
                reasoning_style=reasoning_style or "cot",
            ),
            graph_runtime=self._graph_runtime,
        )

    async def sync_papers_to_zotero(
        self,
        *,
        conversation_id: str,
        paper_ids: list[str],
        collection_name: str | None = None,
    ) -> list[dict[str, Any]]:
        agent_service = self._ensure_agent_service()
        function_service = getattr(self._graph_runtime, "research_function_service", None)
        if function_service is None:
            raise RuntimeError("Research function service is not available in the current runtime.")
        state = self.conversation_state(conversation_id)
        papers_by_id = {paper.paper_id: paper for paper in state["papers"]}
        results: list[dict[str, Any]] = []
        for paper_id in paper_ids:
            paper = papers_by_id.get(paper_id)
            if paper is None:
                continue
            results.append(
                {
                    "paper_id": paper_id,
                    "title": paper.title,
                    **await function_service.sync_paper_to_zotero(
                        paper,
                        collection_name=collection_name,
                    ),
                }
            )
        return results

    def latest_assistant_message(
        self,
        conversation_response: ResearchConversationResponse | None,
        response: ResearchAgentRunResponse,
        *,
        prior_message_count: int = 0,
    ) -> str:
        if conversation_response is not None:
            new_messages = conversation_response.messages[prior_message_count:]
            for message in reversed(new_messages):
                if (
                    message.role == "assistant"
                    and (message.content or "").strip()
                    and message.kind != "welcome"
                ):
                    return message.content or message.title
            for message in reversed(new_messages):
                if (
                    message.role == "assistant"
                    and message.kind not in {"welcome", "candidates"}
                    and (message.title or "").strip()
                ):
                    return message.title
        for message in reversed(response.messages or []):
            if (
                isinstance(message, ResearchMessage)
                and message.role == "assistant"
                and (message.content or "").strip()
                and message.kind != "welcome"
            ):
                return message.content or message.title
        for message in reversed(response.messages or []):
            if (
                isinstance(message, ResearchMessage)
                and message.role == "assistant"
                and message.kind not in {"welcome", "candidates"}
                and (message.title or "").strip()
            ):
                return message.title
        if response.qa is not None:
            return response.qa.answer
        if response.report is not None:
            return response.report.markdown
        if response.papers:
            lines = ["找到这些候选论文："]
            for paper in response.papers[:5]:
                year = f" ({paper.year})" if paper.year else ""
                source = f" [{paper.source}]" if paper.source else ""
                lines.append(f"- {paper.title}{year}{source}")
            return "\n".join(lines)
        if response.import_result is not None:
            return (
                f"导入完成: imported={response.import_result.imported_count}, "
                f"skipped={response.import_result.skipped_count}, failed={response.import_result.failed_count}"
            )
        return "已完成本轮处理，但没有可展示的 assistant 内容。"

    def _plugin_enabled(self, name: str) -> bool:
        metadata = PLUGIN_CATALOG[name]
        stored = self.profile.plugins.get(name)
        if stored is not None:
            return stored.enabled
        return bool(metadata.get("default_enabled", False))

    def _effective_settings(self) -> Settings:
        profile = self.profile_store.load()
        self.profile = profile
        updates: dict[str, Any] = {}
        for plugin_name, metadata in PLUGIN_CATALOG.items():
            if not self._plugin_enabled(plugin_name):
                continue
            for key, value in metadata.get("setting_overrides", {}).items():
                updates[key] = value
        if not updates:
            return self.settings
        return self.settings.model_copy(update=updates)

    def _write_model_settings_to_env(self, **updates: str | None) -> None:
        env_path = self.settings.project_root / ".env"
        env_keys = OrderedDict(
            [
                ("llm_provider", "LLM_PROVIDER"),
                ("llm_model", "LLM_MODEL"),
                ("embedding_provider", "EMBEDDING_PROVIDER"),
                ("embedding_model", "EMBEDDING_MODEL"),
                ("chart_vision_provider", "CHART_VISION_PROVIDER"),
                ("chart_vision_model", "CHART_VISION_MODEL"),
            ]
        )
        desired = {
            env_key: str(value).strip()
            for field_name, env_key in env_keys.items()
            for value in [updates.get(field_name)]
            if value is not None and str(value).strip()
        }
        if not desired:
            return

        lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
        updated_lines: list[str] = []
        seen: set[str] = set()
        for raw_line in lines:
            replaced = False
            for env_key, env_value in desired.items():
                prefix = f"{env_key}="
                if raw_line.startswith(prefix):
                    updated_lines.append(f"{env_key}={env_value}")
                    seen.add(env_key)
                    replaced = True
                    break
            if not replaced:
                updated_lines.append(raw_line)
        for env_key, env_value in desired.items():
            if env_key not in seen:
                updated_lines.append(f"{env_key}={env_value}")
        env_path.write_text("\n".join(updated_lines).rstrip() + "\n", encoding="utf-8")

    def _register_optional_mcp_servers(self, graph_runtime: Any, settings: Settings) -> None:
        if not getattr(settings, "zotero_local_enabled", False):
            return
        from services.research.zotero_local_mcp import (
            ZoteroLocalServerConfig,
            build_zotero_local_mcp_client,
        )

        graph_runtime.external_tool_registry.register_server(
            "zotero-local",
            build_zotero_local_mcp_client(
                ZoteroLocalServerConfig(
                    base_url=getattr(settings, "zotero_local_base_url", "http://127.0.0.1:23119"),
                    user_id=getattr(settings, "zotero_local_user_id", "0"),
                    timeout_seconds=getattr(settings, "zotero_local_timeout_seconds", 20.0),
                )
            ),
            replace=True,
        )

    def _ensure_agent_service(self) -> LiteratureResearchService:
        if self._agent_service is not None and self._graph_runtime is not None:
            return self._agent_service
        from apps.api.research_runtime import (
            build_literature_research_service,
            register_research_runtime_extensions,
        )
        from apps.api.runtime import build_graph_runtime

        effective_settings = self._effective_settings()
        graph_runtime = build_graph_runtime(effective_settings)
        self._register_optional_mcp_servers(graph_runtime, effective_settings)
        from pathlib import Path as _Path
        mcp_config_path = _Path(effective_settings.resolve_path("mcp_servers.json"))
        graph_runtime.external_tool_registry.register_from_json_file(mcp_config_path)
        agent_service = build_literature_research_service(effective_settings, graph_runtime=graph_runtime)
        agent_service.memory_manager = self.service.memory_manager
        register_research_runtime_extensions(
            effective_settings,
            graph_runtime=graph_runtime,
            research_service=agent_service,
        )
        self._graph_runtime = graph_runtime
        self._agent_service = agent_service
        return agent_service
