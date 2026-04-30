from __future__ import annotations

import logging
from typing import Any

from domain.schemas.research import ResearchAgentRunRequest
from services.research.research_knowledge_access import ResearchKnowledgeAccess
from services.research.research_skill_resolver import ResearchSkillResolver
from services.research.supervisor_tools import ResearchAgentToolContext
from services.research.unified_runtime import (
    build_phase1_unified_agent_registry,
    build_phase1_unified_blueprint,
    build_phase1_unified_runtime_context,
)

logger = logging.getLogger(__name__)


class ResearchAgentContextBuilder:
    """Build the supervisor tool context for one research-agent request."""

    def __init__(
        self,
        *,
        runtime: Any,
        skill_resolver: ResearchSkillResolver,
    ) -> None:
        self.runtime = runtime
        self.skill_resolver = skill_resolver

    def build(
        self,
        *,
        request: ResearchAgentRunRequest,
        graph_runtime: Any,
    ) -> ResearchAgentToolContext:
        hydrated_request, restored_task_response = self.runtime._hydrate_request_from_conversation(
            request=request
        )
        context = ResearchAgentToolContext(
            request=hydrated_request,
            research_service=self.runtime.research_service,
            graph_runtime=graph_runtime,
            warnings=[],
        )
        context.knowledge_access = ResearchKnowledgeAccess.from_runtime(graph_runtime)
        if restored_task_response is not None:
            context.task_response = restored_task_response
        elif hydrated_request.task_id:
            try:
                context.task_response = self.runtime.research_service.get_task(
                    hydrated_request.task_id
                )
            except KeyError:
                context.warnings.append(
                    f"research task not found: {hydrated_request.task_id}"
                )

        context.execution_context = self.runtime.research_service.build_execution_context(
            graph_runtime=graph_runtime,
            conversation_id=hydrated_request.conversation_id,
            task=context.task_response.task if context.task_response else None,
            report=context.task_response.report if context.task_response else None,
            papers=context.task_response.papers if context.task_response else None,
            document_ids=(
                context.task_response.task.imported_document_ids
                if context.task_response
                else []
            ),
            selected_paper_ids=hydrated_request.selected_paper_ids,
            skill_name=hydrated_request.skill_name,
            reasoning_style=hydrated_request.reasoning_style,
            metadata=hydrated_request.metadata,
        )
        context.unified_runtime_context = build_phase1_unified_runtime_context(
            graph_runtime=graph_runtime,
            research_service=self.runtime.research_service,
        )
        context.unified_agent_registry = build_phase1_unified_agent_registry(
            graph_runtime=graph_runtime,
            research_service=self.runtime.research_service,
            agent_delegates=getattr(self.runtime, "unified_agent_delegates", {}),
            execution_handlers=getattr(self.runtime, "unified_execution_handlers", {}),
        )
        context.unified_blueprint = build_phase1_unified_blueprint(
            graph_runtime=graph_runtime,
            research_service=self.runtime.research_service,
        ).model_dump(mode="json")
        self._resolve_skills(context=context, graph_runtime=graph_runtime)
        return context

    def _resolve_skills(
        self,
        *,
        context: ResearchAgentToolContext,
        graph_runtime: Any,
    ) -> None:
        try:
            selection = self.skill_resolver.resolve(
                message=context.request.message,
                explicit_skill_name=context.request.skill_name,
                available_tool_names=self._available_tool_names(graph_runtime),
            )
        except Exception:
            logger.warning("Skill matching failed", exc_info=True)
            return

        context.skill_selection = selection
        if selection.skill_context:
            context.skill_context = selection.skill_context
            logger.info(
                "Matched %d skill(s) for request: %s",
                len(selection.active_skill_names),
                ", ".join(selection.active_skill_names),
            )

    def _available_tool_names(self, graph_runtime: Any) -> list[str]:
        capability_registry = getattr(self.runtime, "capability_registry", None)
        if capability_registry is None:
            return sorted(name for name in getattr(self.runtime, "tools", {}).keys() if name)
        return capability_registry.local_capability_names(graph_runtime=graph_runtime)
