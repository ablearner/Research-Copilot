from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from domain.schemas.paper_knowledge import PaperKnowledgeRecord
from domain.schemas.research_context import QAPair, ResearchContext
from domain.schemas.research_memory import LongTermMemoryQuery, LongTermMemoryRecord, SessionMemoryRecord
from domain.schemas.sub_manager import SubManagerState, TaskStep
from memory.long_term_memory import LongTermMemory
from memory.paper_knowledge_memory import PaperKnowledgeMemory
from memory.session_memory import SessionMemory
from memory.user_profile_memory import UserProfileMemory
from memory.working_memory import WorkingMemory


class MemoryManager:
    def __init__(
        self,
        *,
        working_memory: WorkingMemory | None = None,
        session_memory: SessionMemory | None = None,
        long_term_memory: LongTermMemory | None = None,
        paper_knowledge_memory: PaperKnowledgeMemory | None = None,
    ) -> None:
        self.working_memory = working_memory or WorkingMemory()
        self.session_memory = session_memory or SessionMemory()
        self.long_term_memory = long_term_memory or LongTermMemory()
        self.user_profile_memory = UserProfileMemory(self.long_term_memory)
        self.paper_knowledge_memory = paper_knowledge_memory
        self._session_snapshots: dict[str, dict] = {}

    def save_context(self, session_id: str, context: ResearchContext) -> SessionMemoryRecord:
        self.working_memory.sync_context(session_id, context)
        record = self.session_memory.load(session_id)
        return self.session_memory.save(
            record.model_copy(
                update={
                    "context": context,
                    "last_task_plan": list(context.current_task_plan),
                    "sub_manager_states": dict(context.sub_manager_states),
                }
            )
        )

    def hydrate_context(
        self,
        session_id: str,
        *,
        base_context: ResearchContext | None = None,
    ) -> ResearchContext:
        session_record = self.session_memory.load(session_id)
        working_state = self.working_memory.load(session_id)
        context = (base_context or session_record.context).model_copy(deep=True)
        context.selected_papers = list(
            dict.fromkeys([*context.selected_papers, *working_state.selected_paper_ids])
        )
        context.active_papers = list(
            dict.fromkeys([*context.active_papers, *working_state.active_paper_ids])
        )
        if working_state.current_task_plan:
            context.current_task_plan = list(working_state.current_task_plan)
        context.sub_manager_states = self._merge_sub_manager_states(
            context.sub_manager_states,
            session_record.sub_manager_states,
            working_state.sub_manager_states,
        )
        history = self._dedupe_history([*context.session_history, *working_state.recent_history])
        context.session_history = history[-context.user_preferences.max_history_turns :]

        frozen = self.get_frozen_prompt_block(session_id)
        if frozen is not None:
            context.metadata = {
                **context.metadata,
                "recalled_memories": frozen["recalled_memories"],
                "recalled_memory_ids": [m["memory_id"] for m in frozen["recalled_memories"]],
            }
        elif context.research_topic:
            if session_id not in self._session_snapshots:
                self.freeze_session_snapshot(session_id, context)
                frozen = self._session_snapshots.get(session_id)
            if frozen is not None:
                context.metadata = {
                    **context.metadata,
                    "recalled_memories": frozen["recalled_memories"],
                    "recalled_memory_ids": [m["memory_id"] for m in frozen["recalled_memories"]],
                }
            else:
                recall = self.long_term_memory.search(
                    LongTermMemoryQuery(
                        query=context.research_topic,
                        topic=context.research_topic,
                        keywords=context.research_goals[:5],
                        top_k=3,
                    )
                )
                context.metadata = {
                    **context.metadata,
                    "recalled_memory_ids": [record.memory_id for record in recall.records],
                    "recalled_memories": [
                        {
                            "memory_id": record.memory_id,
                            "memory_type": record.memory_type,
                            "content": record.content,
                            "score": record.score,
                        }
                        for record in recall.records
                    ],
                }
        return context

    def record_turn(
        self,
        session_id: str,
        *,
        question: str,
        answer: str,
        citations: list[str] | None = None,
        selected_paper_ids: list[str] | None = None,
        metadata: dict | None = None,
    ) -> SessionMemoryRecord:
        self.working_memory.push_turn(
            session_id,
            question=question,
            answer=answer,
            citations=citations,
            metadata=metadata,
        )
        if selected_paper_ids is not None:
            self.working_memory.set_selected_papers(session_id, selected_paper_ids)
            if selected_paper_ids:
                self.working_memory.set_active_papers(session_id, selected_paper_ids)
        session_record = self.session_memory.append_question(session_id, question)
        context = session_record.context.model_copy(deep=True)
        history = self._dedupe_history(
            [
                *context.session_history,
                QAPair(
                    question=question,
                    answer=answer,
                    citations=list(citations or []),
                    metadata=metadata or {},
                ),
            ]
        )
        context.session_history = history[-context.user_preferences.max_history_turns :]
        if selected_paper_ids:
            context.selected_papers = list(
                dict.fromkeys([*context.selected_papers, *selected_paper_ids])
            )
            context.active_papers = list(dict.fromkeys(selected_paper_ids))
        return self.save_context(session_id, context)

    def set_active_papers(self, session_id: str, paper_ids: list[str]) -> SessionMemoryRecord:
        active_ids = list(dict.fromkeys(paper_ids))
        self.working_memory.set_active_papers(session_id, active_ids)
        record = self.session_memory.load(session_id)
        context = record.context.model_copy(deep=True)
        context.active_papers = active_ids
        if active_ids:
            context.selected_papers = list(dict.fromkeys([*context.selected_papers, *active_ids]))
        return self.save_context(session_id, context)

    def record_conclusion(self, session_id: str, conclusion: str) -> SessionMemoryRecord:
        session_record = self.session_memory.append_conclusion(session_id, conclusion)
        context = session_record.context.model_copy(deep=True)
        context.known_conclusions = list(dict.fromkeys([*context.known_conclusions, conclusion]))
        return self.save_context(session_id, context)

    def promote_conclusion_to_long_term(
        self,
        session_id: str,
        *,
        conclusion: str,
        topic: str = "",
        keywords: list[str] | None = None,
        related_paper_ids: list[str] | None = None,
        metadata: dict | None = None,
    ) -> LongTermMemoryRecord:
        stable_id = uuid5(NAMESPACE_URL, f"{session_id}:{topic}:{conclusion.strip().lower()}")
        record = LongTermMemoryRecord(
            memory_id=f"conclusion:{stable_id}",
            memory_type="conclusion",
            topic=topic,
            content=conclusion,
            keywords=list(keywords or []),
            related_paper_ids=list(related_paper_ids or []),
            source_session_id=session_id,
            metadata={"source": "qa_answer", **(metadata or {})},
        )
        return self.long_term_memory.upsert(record)

    def record_read_paper(self, session_id: str, paper_id: str) -> SessionMemoryRecord:
        return self.session_memory.append_read_paper(session_id, paper_id)

    def record_task_plan(self, session_id: str, task_plan: list[TaskStep]) -> SessionMemoryRecord:
        self.working_memory.set_task_plan(session_id, task_plan)
        record = self.session_memory.load(session_id)
        context = record.context.model_copy(deep=True)
        context.current_task_plan = list(task_plan)
        return self.save_context(session_id, context)

    def update_sub_manager_state(
        self,
        session_id: str,
        *,
        name: str,
        state: SubManagerState,
    ) -> SessionMemoryRecord:
        record = self.session_memory.load(session_id)
        context = record.context.model_copy(deep=True)
        context.sub_manager_states = {
            **context.sub_manager_states,
            name: state,
        }
        self.working_memory.set_sub_manager_states(session_id, context.sub_manager_states)
        return self.save_context(session_id, context)

    def finalize_session(self, session_id: str) -> SessionMemoryRecord:
        finalized = self.session_memory.finalize_session(session_id)
        if finalized.summary and finalized.summary.summary_text:
            self.long_term_memory.upsert(
                LongTermMemoryRecord(
                    memory_id=f"session_summary:{session_id}",
                    memory_type="session_summary",
                    topic=finalized.context.research_topic,
                    content=finalized.summary.summary_text,
                    keywords=finalized.context.research_goals[:6],
                    related_paper_ids=finalized.read_paper_ids[:10],
                    source_session_id=session_id,
                    context_snapshot=finalized.context.model_dump(mode="json"),
                    metadata={"source": "session_summary"},
                )
            )
        return finalized

    def clear_session(self, session_id: str) -> None:
        self.working_memory.clear(session_id)
        self.session_memory.clear(session_id)

    def load_user_profile(self, *, user_id: str = "local-user"):
        return self.user_profile_memory.load_profile(user_id=user_id)

    def update_user_profile(
        self,
        *,
        user_id: str = "local-user",
        topic: str | None = None,
        sources: list[str] | None = None,
        keywords: list[str] | None = None,
        reasoning_style: str | None = None,
        answer_language: str | None = None,
        note: str | None = None,
        topics: list[str] | None = None,
        preferred_recency_days: int | None = None,
    ):
        return self.user_profile_memory.update_profile(
            user_id=user_id,
            topic=topic,
            sources=sources,
            keywords=keywords,
            reasoning_style=reasoning_style,
            answer_language=answer_language,
            note=note,
            topics=topics,
            preferred_recency_days=preferred_recency_days,
        )

    def clear_user_profile(self, *, user_id: str = "local-user"):
        return self.user_profile_memory.clear_profile(user_id=user_id)

    def remove_user_profile_topics(
        self,
        *,
        user_id: str = "local-user",
        topics: list[str],
    ):
        return self.user_profile_memory.remove_topics(user_id=user_id, topics=topics)

    def observe_user_query(
        self,
        *,
        user_id: str = "local-user",
        topics: list[str],
        sources: list[str] | None = None,
        keywords: list[str] | None = None,
        preferred_recency_days: int | None = None,
        signal_strength: float = 1.0,
        metadata: dict | None = None,
    ):
        return self.user_profile_memory.observe_query(
            user_id=user_id,
            topics=topics,
            sources=sources,
            keywords=keywords,
            preferred_recency_days=preferred_recency_days,
            signal_strength=signal_strength,
            metadata=metadata,
        )

    def record_user_recommendations(
        self,
        *,
        user_id: str = "local-user",
        topics_used: list[str],
        recommendation_ids: list[str],
        query: str,
    ):
        return self.user_profile_memory.record_recommendations(
            user_id=user_id,
            topics_used=topics_used,
            recommendation_ids=recommendation_ids,
            query=query,
        )

    def update_paper_knowledge(self, record: PaperKnowledgeRecord) -> PaperKnowledgeRecord:
        if self.paper_knowledge_memory is None:
            raise RuntimeError("PaperKnowledgeMemory is not configured")
        return self.paper_knowledge_memory.upsert(record)

    def freeze_session_snapshot(self, session_id: str, context: ResearchContext) -> None:
        """Capture a frozen memory snapshot for prompt injection.

        Called once at session start.  Subsequent hydrate_context() calls
        will use this snapshot for the system prompt portion, keeping the
        prompt prefix stable for Anthropic prompt caching.
        """
        recall = self.long_term_memory.search(
            LongTermMemoryQuery(
                query=context.research_topic or "",
                topic=context.research_topic or "",
                keywords=context.research_goals[:5],
                top_k=3,
            )
        )
        self._session_snapshots[session_id] = {
            "recalled_memories": [
                {
                    "memory_id": r.memory_id,
                    "memory_type": r.memory_type,
                    "content": r.content,
                    "score": r.score,
                }
                for r in recall.records
            ],
            "user_profile": self.load_user_profile(),
        }

    def get_frozen_prompt_block(self, session_id: str) -> dict | None:
        """Return the frozen snapshot for system prompt injection."""
        return self._session_snapshots.get(session_id)

    def _dedupe_history(self, history: list[QAPair]) -> list[QAPair]:
        deduped: list[QAPair] = []
        seen: set[tuple[str, str]] = set()
        for pair in history:
            marker = (pair.question.strip(), pair.answer.strip())
            if not marker[0] or not marker[1] or marker in seen:
                continue
            seen.add(marker)
            deduped.append(pair)
        return deduped

    def _merge_sub_manager_states(
        self,
        *layers: dict[str, SubManagerState],
    ) -> dict[str, SubManagerState]:
        merged = dict(ResearchContext().sub_manager_states)
        for layer in layers:
            for name, state in layer.items():
                merged[name] = state
        return merged
