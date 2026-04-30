from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _read(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_direct_discovery_helper_removed_from_research_service_and_paper_operations() -> None:
    literature_service = _read("services/research/literature_research_service.py")
    paper_operations = _read("services/research/paper_operations.py")

    assert "_run_direct_discovery" not in literature_service
    assert "_run_direct_discovery" not in paper_operations
    assert "research_discovery_capability.discover(" not in literature_service
    assert "research_discovery_capability.discover(" not in paper_operations


def test_high_level_research_modules_do_not_call_low_level_rag_handlers_directly() -> None:
    qa_router = _read("services/research/qa_router.py")
    paper_operations = _read("services/research/paper_operations.py")

    banned_markers = (
        "graph_runtime.handle_ask_document",
        "graph_runtime.handle_ask_fused",
        "graph_runtime.handle_parse_document",
        "graph_runtime.handle_index_document",
        "graph_runtime.handle_graph_backfill_document",
        "graph_runtime.handle_understand_chart",
    )
    for marker in banned_markers:
        assert marker not in qa_router
        assert marker not in paper_operations


def test_legacy_unified_bridge_methods_are_removed() -> None:
    runtime_core = _read("services/research/research_supervisor_graph_runtime_core.py")

    assert "_legacy_unified_delegates" not in runtime_core
    assert "_legacy_unified_execution_handlers" not in runtime_core
