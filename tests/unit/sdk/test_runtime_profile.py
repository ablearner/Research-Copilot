from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.config import Settings
from domain.schemas.research import ImportPapersResponse, ResearchAgentRunResponse, ResearchTask, ResearchTaskResponse
from sdk.client import ResearchCopilotSDK
from sdk.runtime_profile import RuntimeProfileStore


def test_runtime_profile_store_round_trip(tmp_path: Path) -> None:
    store = RuntimeProfileStore(tmp_path / "runtime_profile.json")
    profile = store.update_models(llm_provider="openai", llm_model="gpt-test")
    assert profile.models.llm_provider == "openai"
    assert profile.models.llm_model == "gpt-test"

    profile = store.set_plugin_enabled("zotero_local_mcp", True)
    assert profile.plugins["zotero_local_mcp"].enabled is True


def test_sdk_applies_profile_to_runtime_and_catalogs(tmp_path: Path) -> None:
    settings = Settings(
        research_storage_root=str(tmp_path / "research"),
        research_session_memory_dir=str(tmp_path / "session_memory"),
        research_paper_knowledge_dir=str(tmp_path / "paper_knowledge"),
        upload_dir=str(tmp_path / "uploads"),
    )
    sdk = ResearchCopilotSDK.from_settings(settings)
    sdk.update_model_profile(llm_provider="openai", llm_model="gpt-5.4-mini")
    sdk.set_plugin_enabled("zotero_local_mcp", True)

    runtime = sdk.describe_runtime()
    assert runtime["llm"]["provider"] == "openai"
    assert runtime["llm"]["model"] == "gpt-5.4-mini"

    plugins = {item["name"]: item for item in sdk.list_plugins()}
    assert plugins["zotero_local_mcp"]["enabled"] is True


def test_sdk_registers_zotero_local_mcp_server_when_plugin_enabled(tmp_path: Path) -> None:
    settings = Settings(
        research_storage_root=str(tmp_path / "research"),
        research_session_memory_dir=str(tmp_path / "session_memory"),
        research_paper_knowledge_dir=str(tmp_path / "paper_knowledge"),
        upload_dir=str(tmp_path / "uploads"),
    )
    sdk = ResearchCopilotSDK.from_settings(settings)
    sdk.set_plugin_enabled("zotero_local_mcp", True)

    sdk._ensure_agent_service()

    assert sdk._graph_runtime is not None
    assert "zotero-local" in sdk._graph_runtime.external_tool_registry.list_servers()


def test_sdk_clear_conversation_memory_invalidates_cached_state(tmp_path: Path) -> None:
    settings = Settings(
        research_storage_root=str(tmp_path / "research"),
        research_session_memory_dir=str(tmp_path / "session_memory"),
        research_paper_knowledge_dir=str(tmp_path / "paper_knowledge"),
        upload_dir=str(tmp_path / "uploads"),
    )
    sdk = ResearchCopilotSDK.from_settings(settings)
    conversation = sdk.create_conversation(topic="GraphRAG")
    conversation_id = conversation.conversation.conversation_id

    sdk.conversation_state(conversation_id)
    assert sdk._state_cache

    sdk.clear_conversation_memory(conversation_id)

    assert conversation_id not in [key[0] for key in sdk._state_cache]
    try:
        sdk.get_conversation(conversation_id)
    except KeyError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("Conversation should have been deleted.")


@pytest.mark.asyncio
async def test_sdk_run_agent_message_does_not_store_raw_user_message_as_topic(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        research_storage_root=str(tmp_path / "research"),
        research_session_memory_dir=str(tmp_path / "session_memory"),
        research_paper_knowledge_dir=str(tmp_path / "paper_knowledge"),
        upload_dir=str(tmp_path / "uploads"),
    )
    sdk = ResearchCopilotSDK.from_settings(settings)
    conversation = sdk.create_conversation(topic="GraphRAG")
    conversation_id = conversation.conversation.conversation_id
    captured: dict[str, object] = {}

    class AgentServiceStub:
        async def run_agent(self, request, graph_runtime=None):
            return ResearchAgentRunResponse(status="succeeded")

        def record_agent_turn(self, conversation_id, request, response):
            return None

    sdk.conversation_state(conversation_id)
    monkeypatch.setattr(sdk, "_ensure_agent_service", lambda: AgentServiceStub())
    monkeypatch.setattr(
        sdk.service.report_service,
        "load_conversation",
        lambda conversation_id: None,
    )

    def capture_update(**kwargs):
        captured.update(kwargs)
        return sdk.load_user_profile()

    monkeypatch.setattr(sdk, "update_user_profile", capture_update)
    sdk._graph_runtime = SimpleNamespace()

    await sdk.run_agent_message(
        message="科研助手里的长期记忆方向，有没有比较新的论文？",
        conversation_id=conversation_id,
        sources=["arxiv", "openalex"],
    )

    assert captured["topic"] is None
    assert "sources" not in captured
    assert conversation_id not in [key[0] for key in sdk._state_cache]


@pytest.mark.asyncio
async def test_sdk_import_papers_for_conversation_records_import_turn_and_invalidates_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = Settings(
        research_storage_root=str(tmp_path / "research"),
        research_session_memory_dir=str(tmp_path / "session_memory"),
        research_paper_knowledge_dir=str(tmp_path / "paper_knowledge"),
        upload_dir=str(tmp_path / "uploads"),
    )
    sdk = ResearchCopilotSDK.from_settings(settings)
    conversation = sdk.create_conversation(topic="GraphRAG")
    conversation_id = conversation.conversation.conversation_id
    captured: dict[str, object] = {}
    task_response = ResearchTaskResponse(
        task=ResearchTask(
            task_id="task-import-1",
            topic="GraphRAG",
            status="completed",
            created_at="2026-04-25T00:00:00+00:00",
            updated_at="2026-04-25T00:00:00+00:00",
            imported_document_ids=["doc-1"],
        ),
        papers=[],
        report=None,
        warnings=[],
    )

    class AgentServiceStub:
        async def import_papers(self, request, graph_runtime=None):
            captured["request"] = request
            return ImportPapersResponse(imported_count=1, skipped_count=0, failed_count=0, results=[])

        def get_task(self, task_id):
            captured["task_id"] = task_id
            return task_response

        def record_import_turn(self, conversation_id, task_response, import_response, selected_paper_ids):
            captured["conversation_id"] = conversation_id
            captured["selected_paper_ids"] = selected_paper_ids
            captured["recorded_task_response"] = task_response
            captured["import_response"] = import_response
            return None

    def conversation_state_stub(conversation_id_arg: str, **kwargs):
        captured["conversation_state_kwargs"] = kwargs
        assert conversation_id_arg == conversation_id
        return {"task_id": "task-import-1"}

    monkeypatch.setattr(sdk, "_ensure_agent_service", lambda: AgentServiceStub())
    monkeypatch.setattr(sdk, "conversation_state", conversation_state_stub)
    sdk._graph_runtime = SimpleNamespace()
    sdk._cache_put(
        sdk._state_cache,
        (conversation_id, False),
        {"task_id": "task-import-1"},
        ttl_seconds=999.0,
    )

    result = await sdk.import_papers_for_conversation(
        conversation_id=conversation_id,
        paper_ids=["paper-1"],
    )

    assert result.imported_count == 1
    assert captured["conversation_id"] == conversation_id
    assert captured["selected_paper_ids"] == ["paper-1"]
    assert captured["recorded_task_response"] == task_response
    assert captured["conversation_state_kwargs"]["use_cache"] is False
    assert conversation_id not in [key[0] for key in sdk._state_cache]


def test_sdk_profile_management_methods_round_trip(tmp_path: Path) -> None:
    settings = Settings(
        research_storage_root=str(tmp_path / "research"),
        research_session_memory_dir=str(tmp_path / "session_memory"),
        research_paper_knowledge_dir=str(tmp_path / "paper_knowledge"),
        upload_dir=str(tmp_path / "uploads"),
    )
    sdk = ResearchCopilotSDK.from_settings(settings)

    sdk.update_user_profile(topic="GraphRAG", keywords=["retrieval"])
    sdk.update_user_profile(topic="未命名研究会话")
    updated = sdk.remove_user_profile_topics(["未命名研究会话"])
    cleared = sdk.clear_user_profile()

    assert [item.topic_name for item in updated.interest_topics] == ["GraphRAG"]
    assert cleared.interest_topics == []
