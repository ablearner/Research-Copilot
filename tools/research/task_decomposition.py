from __future__ import annotations

from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, Field

from domain.schemas.research_context import ResearchContext
from domain.schemas.research_functions import DecomposeTaskFunctionOutput
from domain.schemas.sub_manager import TaskStep


class _TaskProposal(BaseModel):
    action_name: Literal["search_literature", "answer_question", "analyze_papers", "write_review"]
    instruction: str = Field(min_length=1)


# These are descriptive destinations in the returned plan, never executable objects.
_ACTIONS = {
    "search_literature": ("LiteratureScoutAgent", ["papers"]),
    "answer_question": ("ResearchQAAgent", ["answer"]),
    "analyze_papers": ("PaperAnalysisAgent", ["answer", "focus", "recommended_paper_ids"]),
    "write_review": ("ResearchWriterAgent", ["review_text"]),
}


class TaskDecompositionTool:
    """Produce one proposed research step; dispatch and execution belong to the runtime.

    This tool has no task state, worker registry, execution loop, or replanning policy.
    The optional LLM can suggest only read/analyze/write work supported by the context.
    """

    def __init__(self, *, llm_adapter: Any | None = None) -> None:
        self.llm_adapter = llm_adapter

    async def decompose(
        self, *, user_request: str, context: ResearchContext,
    ) -> DecomposeTaskFunctionOutput:
        paper_ids = list(dict.fromkeys([
            *context.selected_papers, *context.active_papers,
            *(paper.paper_id for paper in context.imported_papers),
        ]))
        available = ["search_literature"]
        if context.metadata.get("task_id"):
            if paper_ids:
                available.append("answer_question")
            if len(paper_ids) > 1:
                available.append("analyze_papers")
            if paper_ids or context.known_conclusions:
                available.append("write_review")
        request = user_request.casefold()
        action = "search_literature"
        if any(token in request for token in ("compare", "recommend", "对比", "比较", "推荐")):
            action = "analyze_papers"
        elif any(token in request for token in ("review", "survey", "综述", "报告")):
            action = "write_review"
        elif paper_ids:
            action = "answer_question"
        if action not in available:
            action = "search_literature"
        instruction = user_request
        source = "task_decomposition_tool"
        context_slice = {
            "research_topic": context.research_topic,
            "research_goals": list(context.research_goals),
            "selected_papers": paper_ids,
            "imported_papers": [paper.model_dump(mode="json") for paper in context.imported_papers],
            "known_conclusions": list(context.known_conclusions),
            "open_questions": list(context.open_questions),
        }
        if self.llm_adapter is not None:
            try:
                proposal = await self.llm_adapter.generate_structured(
                    prompt=(
                        "Propose one next research step for the request: {user_request}. "
                        "Use only an action from {available_actions}, grounded in {context}. "
                        "Return action_name and a concrete instruction. This is a plan proposal only."
                    ),
                    input_data={
                        "user_request": user_request,
                        "context": context_slice,
                        "available_actions": available,
                    },
                    response_model=_TaskProposal,
                )
                proposal = _TaskProposal.model_validate(proposal)
                if proposal.action_name in available:
                    action, instruction = proposal.action_name, proposal.instruction
                    source = "task_decomposition_llm"
            except Exception:  # noqa: BLE001 - planning remains usable without an LLM
                pass
        worker, required_fields = _ACTIONS[action]
        task_key = f"{context.metadata.get('task_id', '')}:{user_request}:{action}:{','.join(paper_ids)}"
        step = TaskStep(
            task_id=f"task_{uuid5(NAMESPACE_URL, task_key).hex[:12]}",
            assigned_to=worker,
            instruction=instruction,
            task_type=action,
            context_slice=context_slice,
            expected_output_schema={"required_fields": required_fields},
        )
        return DecomposeTaskFunctionOutput(
            task_plan=[step],
            assigned_sub_manager=worker,
            parallel_allowed=False,
            metadata={"decision_source": source, "action_name": action, "worker_agent": worker},
        )
