from __future__ import annotations

from typing import Any, Callable

from domain.schemas.paper_knowledge import PaperKnowledgeRecord
from domain.schemas.research import PaperCandidate, ResearchReport, ResearchTask


class ResearchMemoryGateway:
    """Research-domain memory facade over MemoryManager and session memory."""

    def __init__(
        self,
        *,
        memory_manager: Any,
        research_context_manager: Any,
        paper_reading_skill: Any,
        compact_text: Callable[[str | None], str],
    ) -> None:
        self.memory_manager = memory_manager
        self.research_context_manager = research_context_manager
        self.paper_reading_skill = paper_reading_skill
        self.compact_text = compact_text

    def load_user_profile(self):
        return self.memory_manager.load_user_profile()

    def update_user_profile(
        self,
        *,
        topic: str,
        answer_language: str | None = None,
        note: str | None = None,
    ) -> Any:
        return self.memory_manager.update_user_profile(
            topic=topic,
            answer_language=answer_language,
            note=note,
        )

    def hydrate_context(
        self,
        session_id: str,
        *,
        base_context: Any,
    ) -> Any:
        return self.memory_manager.hydrate_context(
            session_id,
            base_context=base_context,
        )

    def save_context(self, session_id: str, research_context: Any) -> Any:
        return self.memory_manager.save_context(session_id, research_context)

    def set_active_papers(self, session_id: str, paper_ids: list[str]) -> Any:
        return self.memory_manager.set_active_papers(session_id, paper_ids)

    def record_turn(
        self,
        session_id: str,
        *,
        question: str,
        answer: str,
        selected_paper_ids: list[str],
        metadata: dict[str, Any],
    ) -> Any:
        return self.memory_manager.record_turn(
            session_id,
            question=question,
            answer=answer,
            selected_paper_ids=selected_paper_ids,
            metadata=metadata,
        )

    def append_intermediate_step(
        self,
        session_id: str,
        *,
        content: str,
        step_type: str,
        metadata: dict[str, Any],
    ) -> None:
        self.memory_manager.working_memory.append_intermediate_step(
            session_id=session_id,
            content=content,
            step_type=step_type,
            metadata=metadata,
        )

    def update_paper_knowledge(self, record: PaperKnowledgeRecord) -> Any:
        return self.memory_manager.update_paper_knowledge(record)

    def persist_research_update(
        self,
        *,
        session_id: str,
        conversation_id: str | None,
        graph_runtime: Any | None,
        task: ResearchTask | None,
        report: ResearchReport | None,
        papers: list[PaperCandidate],
        document_ids: list[str],
        selected_paper_ids: list[str],
        task_intent: str,
        question: str | None,
        answer: str | None,
        retrieval_summary: str | None,
        metadata_update: dict[str, Any] | None,
    ) -> None:
        session_memory = getattr(graph_runtime, "session_memory", None) if graph_runtime is not None else None
        report_summary = ""
        if report is not None and report.highlights:
            report_summary = self.compact_text("；".join(report.highlights[:2]))
        elif report is not None:
            report_summary = self.compact_text(report.markdown)
        cleaned_metadata = {
            "conversation_id": conversation_id,
            "task_id": task.task_id if task else None,
            "research_topic": task.topic if task else None,
            "paper_count": len(papers) if papers else (task.paper_count if task else None),
            "document_ids": document_ids[:12],
            "selected_paper_ids": selected_paper_ids[:8] or None,
            "report_highlights": report.highlights[:3] if report else None,
            "report_gaps": report.gaps[:2] if report else None,
            **(metadata_update or {}),
        }
        research_context = self.research_context_manager.build_from_artifacts(
            task=task,
            report=report,
            papers=papers,
            selected_paper_ids=selected_paper_ids,
            paper_summaries=self.research_context_manager.compress_papers(
                papers=papers,
                selected_paper_ids=selected_paper_ids,
                paper_reading_skill=self.paper_reading_skill,
            ),
            metadata=cleaned_metadata,
        )
        hydrated_context = self.hydrate_context(
            session_id,
            base_context=research_context,
        )
        merged_context = self.research_context_manager.update_context(
            current_context=hydrated_context,
            topic=research_context.research_topic,
            goals=research_context.research_goals,
            selected_papers=selected_paper_ids,
            imported_papers=research_context.imported_papers,
            known_conclusions=research_context.known_conclusions,
            open_questions=research_context.open_questions,
            paper_summaries=research_context.paper_summaries,
            current_task_plan=hydrated_context.current_task_plan,
            sub_manager_states=hydrated_context.sub_manager_states,
            metadata=cleaned_metadata,
        )
        self.save_context(session_id, merged_context)
        if retrieval_summary:
            self.append_intermediate_step(
                session_id,
                content=retrieval_summary,
                step_type="retrieve",
                metadata={
                    "task_intent": task_intent,
                    "document_ids": document_ids[:8],
                },
            )
        if question and answer:
            self.record_turn(
                session_id,
                question=question,
                answer=answer,
                selected_paper_ids=selected_paper_ids,
                metadata={
                    "task_id": task.task_id if task else None,
                    "conversation_id": conversation_id,
                    "document_ids": document_ids,
                    "task_intent": task_intent,
                    "paper_count": len(papers) if papers else (task.paper_count if task else 0),
                },
            )
        if session_memory is not None and hasattr(session_memory, "update_research_context"):
            session_memory.update_research_context(
                session_id=session_id,
                current_document_id=(document_ids or [None])[0],
                last_retrieval_summary=retrieval_summary,
                last_answer_summary=self.compact_text(answer) if answer else report_summary or None,
                current_task_intent=task_intent,
                metadata_update=cleaned_metadata,
            )
        if question and answer and session_memory is not None and hasattr(session_memory, "append_research_turn"):
            session_memory.append_research_turn(
                session_id=session_id,
                question=question,
                answer=answer,
                task_id=task.task_id if task else None,
                conversation_id=conversation_id,
                document_ids=document_ids,
                metadata={
                    "task_intent": task_intent,
                    "paper_count": len(papers) if papers else (task.paper_count if task else 0),
                },
            )
