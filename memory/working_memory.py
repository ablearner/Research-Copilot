from __future__ import annotations

from datetime import datetime, timezone

from domain.schemas.research_context import QAPair, ResearchContext
from domain.schemas.research_memory import WorkingMemoryState, WorkingMemoryStep
from domain.schemas.sub_manager import SubManagerState, TaskStep


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class WorkingMemory:
    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max(1, max_turns)
        self._states: dict[str, WorkingMemoryState] = {}

    def load(self, session_id: str) -> WorkingMemoryState:
        existing = self._states.get(session_id)
        if existing is not None:
            return existing
        state = WorkingMemoryState(session_id=session_id, max_turns=self.max_turns)
        self._states[session_id] = state
        return state

    def save(self, state: WorkingMemoryState) -> WorkingMemoryState:
        updated = state.model_copy(update={"updated_at": utc_now()})
        self._states[state.session_id] = updated
        return updated

    def push_turn(
        self,
        session_id: str,
        *,
        question: str,
        answer: str,
        citations: list[str] | None = None,
        metadata: dict | None = None,
    ) -> WorkingMemoryState:
        state = self.load(session_id)
        recent_history = list(state.recent_history)
        recent_history.append(
            QAPair(
                question=question,
                answer=answer,
                citations=list(citations or []),
                metadata=metadata or {},
            )
        )
        recent_history = recent_history[-state.max_turns :]
        return self.save(
            state.model_copy(
                update={
                    "recent_history": recent_history,
                }
            )
        )

    def set_selected_papers(self, session_id: str, paper_ids: list[str]) -> WorkingMemoryState:
        state = self.load(session_id)
        return self.save(
            state.model_copy(update={"selected_paper_ids": list(dict.fromkeys(paper_ids))})
        )

    def set_active_papers(self, session_id: str, paper_ids: list[str]) -> WorkingMemoryState:
        state = self.load(session_id)
        return self.save(
            state.model_copy(update={"active_paper_ids": list(dict.fromkeys(paper_ids))})
        )

    def set_task_plan(self, session_id: str, task_plan: list[TaskStep]) -> WorkingMemoryState:
        state = self.load(session_id)
        active_task_ids = [
            step.task_id
            for step in task_plan
            if step.status in {"planned", "queued", "running"}
        ]
        return self.save(
            state.model_copy(
                update={
                    "current_task_plan": list(task_plan),
                    "active_task_ids": active_task_ids,
                }
            )
        )

    def set_sub_manager_states(
        self,
        session_id: str,
        sub_manager_states: dict[str, SubManagerState],
    ) -> WorkingMemoryState:
        state = self.load(session_id)
        return self.save(state.model_copy(update={"sub_manager_states": dict(sub_manager_states)}))

    def sync_context(self, session_id: str, context: ResearchContext) -> WorkingMemoryState:
        state = self.load(session_id)
        active_task_ids = [
            step.task_id
            for step in context.current_task_plan
            if step.status in {"planned", "queued", "running"}
        ]
        return self.save(
            state.model_copy(
                update={
                    "selected_paper_ids": list(dict.fromkeys(context.selected_papers)),
                    "active_paper_ids": list(dict.fromkeys(context.active_papers)),
                    "current_task_plan": list(context.current_task_plan),
                    "sub_manager_states": dict(context.sub_manager_states),
                    "active_task_ids": active_task_ids,
                }
            )
        )

    def append_intermediate_step(
        self,
        session_id: str,
        *,
        content: str,
        step_type: str = "other",
        tool_name: str | None = None,
        metadata: dict | None = None,
    ) -> WorkingMemoryState:
        state = self.load(session_id)
        intermediate_steps = list(state.intermediate_steps)
        intermediate_steps.append(
            WorkingMemoryStep(
                step_type=step_type,
                content=content,
                tool_name=tool_name,
                metadata=metadata or {},
            )
        )
        intermediate_steps = intermediate_steps[-(state.max_turns * 3) :]
        return self.save(
            state.model_copy(
                update={
                    "intermediate_steps": intermediate_steps,
                }
            )
        )

    def clear(self, session_id: str) -> None:
        self._states.pop(session_id, None)
