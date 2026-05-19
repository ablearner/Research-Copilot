from __future__ import annotations

import sqlite3
from pathlib import Path

from core.config import Settings
from domain.schemas.paper_knowledge import PaperKnowledgeCard, PaperKnowledgeRecord
from domain.schemas.research_context import ResearchContext
from domain.schemas.research_memory import LongTermMemoryQuery
from memory.factory import build_memory_manager, resolve_memory_db_path


def test_factory_uses_single_sqlite_db_for_research_memory(tmp_path: Path) -> None:
    settings = Settings(
        research_storage_root=str(tmp_path / "research"),
        upload_dir=str(tmp_path / "uploads"),
    )
    manager = build_memory_manager(settings)

    manager.save_context(
        "session-1",
        ResearchContext(
            research_topic="GraphRAG memory",
            research_goals=["unify memory persistence"],
        ),
    )
    manager.update_paper_knowledge(
        PaperKnowledgeRecord(
            paper_id="paper-1",
            document_id="doc-1",
            title="GraphRAG Memory",
            knowledge_card=PaperKnowledgeCard(
                paper_id="paper-1",
                title="GraphRAG Memory",
                summary="Factory stores paper knowledge in the same SQLite database.",
            ),
        )
    )
    promoted = manager.promote_conclusion_to_long_term(
        "session-1",
        conclusion=(
            "GraphRAG memory persistence should use one configured SQLite database "
            "for session memory, long-term memory, and paper knowledge records."
        ),
        topic="GraphRAG memory",
        keywords=["GraphRAG", "memory"],
        related_paper_ids=["paper-1"],
        metadata={"evidence_count": 3, "confidence": 0.8},
    )

    db_path = resolve_memory_db_path(settings)
    assert db_path == tmp_path / "research" / "kepler.db"
    assert promoted is not None
    conn = sqlite3.connect(str(db_path))
    try:
        assert conn.execute("SELECT COUNT(*) FROM research_session_memory").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM research_paper_knowledge").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM long_term_memory").fetchone()[0] == 1
    finally:
        conn.close()


def test_quality_gate_blocks_low_quality_long_term_conclusion(tmp_path: Path) -> None:
    settings = Settings(
        research_storage_root=str(tmp_path / "research"),
        upload_dir=str(tmp_path / "uploads"),
    )
    manager = build_memory_manager(settings)

    rejected = manager.promote_conclusion_to_long_term(
        "session-1",
        conclusion="证据不足，当前无法稳定回答该问题。",
        topic="GraphRAG memory",
        keywords=["GraphRAG"],
        related_paper_ids=[],
        metadata={"evidence_count": 0, "confidence": 0.1},
    )
    result = manager.long_term_memory.search(
        LongTermMemoryQuery(query="GraphRAG memory", topic="GraphRAG memory")
    )

    assert rejected is None
    assert result.records == []


def test_quality_gate_records_decision_on_accepted_conclusion(tmp_path: Path) -> None:
    settings = Settings(
        research_storage_root=str(tmp_path / "research"),
        upload_dir=str(tmp_path / "uploads"),
    )
    manager = build_memory_manager(settings)

    accepted = manager.promote_conclusion_to_long_term(
        "session-1",
        conclusion=(
            "Across the evaluated papers, evidence-backed long-term memory should "
            "prefer compact conclusions with explicit topic, confidence, and paper links."
        ),
        topic="research memory",
        keywords=["memory", "quality"],
        related_paper_ids=["paper-1", "paper-2"],
        metadata={"evidence_count": 4, "confidence": 0.75},
    )

    assert accepted is not None
    assert accepted.metadata["quality_gate"]["allowed"] is True
    assert accepted.metadata["quality_gate"]["score"] >= settings.memory_min_quality_score
