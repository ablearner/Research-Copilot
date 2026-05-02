"""Build user-facing ResearchMessage list from runtime results.

Extracted from supervisor_graph_runtime_core.py (P1) to separate
presentation logic from the core execution loop.
"""
from __future__ import annotations

from typing import Any

from domain.schemas.agent_message import AgentMessage, AgentResultMessage
from domain.schemas.research import (
    ResearchAgentRunRequest,
    ResearchAgentTraceStep,
    ResearchMessage,
    ResearchWorkspaceState,
)
from runtime.research.agent_protocol import (
    ResearchAgentToolContext,
    _message,
)
from runtime.research.unified_runtime import serialize_unified_delegation_plan


class ResearchResponseFormatter:
    """Build user-facing ResearchMessage list from runtime results."""

    def __init__(self, *, manager_display_name: str = "ResearchSupervisorAgent") -> None:
        self._manager_display_name = manager_display_name

    def build_messages(
        self,
        request: ResearchAgentRunRequest,
        context: ResearchAgentToolContext,
        trace: list[ResearchAgentTraceStep],
        workspace: ResearchWorkspaceState,
        *,
        agent_messages: list[AgentMessage],
        agent_results: list[AgentResultMessage],
        clarification_request: str | None,
        replan_count: int,
        serialize_task_plan_fn: Any = None,
    ) -> list[ResearchMessage]:
        executed_actions = {step.action_name for step in trace if step.status in {"succeeded", "skipped"}}
        messages = [
            _message(
                role="user",
                kind="topic" if request.mode != "qa" else "question",
                title="用户研究目标" if request.mode != "qa" else "研究集合提问",
                content=request.message,
            )
        ]
        if clarification_request:
            messages.append(
                _message(
                    role="assistant",
                    kind="warning",
                    title="需要澄清研究目标",
                    meta=f"{self._manager_display_name} 请求用户澄清范围",
                    content=clarification_request,
                    payload={"clarification_request": clarification_request},
                )
            )
        if agent_messages:
            plan_lines = [
                (
                    f"- {message.task_id} · {message.agent_to} · {message.task_type} "
                    f"· priority={message.priority}"
                    f"{' · depends_on=' + ','.join(message.depends_on) if message.depends_on else ''}"
                )
                for message in agent_messages[-8:]
            ]
            delegation_plan = serialize_task_plan_fn(agent_messages, agent_results) if serialize_task_plan_fn else []
            messages.append(
                _message(
                    role="assistant",
                    kind="notice",
                    title="Manager 决策轨迹",
                    meta=f"decisions={len(agent_messages)} · results={len(agent_results)} · recoveries={replan_count}",
                    content="\n".join(plan_lines),
                    payload={
                        "agent_messages": [message.model_dump(mode="json") for message in agent_messages],
                        "agent_results": [result.model_dump(mode="json") for result in agent_results],
                        "delegation_plan": delegation_plan,
                        "unified_delegation_plan": serialize_unified_delegation_plan(
                            agent_messages,
                            agent_results,
                            registry=context.unified_agent_registry,
                        ),
                    },
                )
            )
        report_is_fresh = executed_actions & {"search_literature", "write_review"}
        if context.report and report_is_fresh:
            messages.append(
                _message(
                    role="assistant",
                    kind="report",
                    title="自主文献综述",
                    meta=f"候选论文 {context.report.paper_count} 篇",
                    content=context.report.markdown,
                    payload={"report": context.report.model_dump(mode="json")},
                )
            )
        if context.papers and report_is_fresh:
            messages.append(
                _message(
                    role="assistant",
                    kind="candidates",
                    title="候选论文池",
                    meta=f"当前共 {len(context.papers)} 篇",
                    payload={"papers": [paper.model_dump(mode="json") for paper in context.papers]},
                )
            )
        if context.import_result:
            lines = [
                f"imported={context.import_result.imported_count} · skipped={context.import_result.skipped_count} · failed={context.import_result.failed_count}"
            ]
            lines.extend(
                f"- {result.title} · {result.status}{' · doc=' + result.document_id if result.document_id else ''}"
                for result in context.import_result.results[:5]
            )
            messages.append(
                _message(
                    role="assistant",
                    kind="import_result",
                    title="自主导入结果",
                    meta=f"{self._manager_display_name} 调用了论文导入工具",
                    content="\n".join(lines),
                    payload={"import_result": context.import_result.model_dump(mode="json")},
                )
            )
        if context.zotero_sync_results:
            synced_count = sum(1 for item in context.zotero_sync_results if str(item.get("status") or "") in {"imported", "reused"})
            lines = [f"synced={synced_count} · total={len(context.zotero_sync_results)}"]
            lines.extend(
                f"- {item.get('title', '')} · {item.get('status', 'unknown')}"
                for item in context.zotero_sync_results[:5]
            )
            messages.append(
                _message(
                    role="assistant",
                    kind="notice",
                    title="Zotero 同步结果",
                    meta=f"{self._manager_display_name} 调用了 Zotero 同步工具",
                    content="\n".join(lines),
                    payload={"zotero_sync_results": context.zotero_sync_results},
                )
            )
        if context.parsed_document:
            messages.append(
                _message(
                    role="assistant",
                    kind="notice",
                    title="文档理解结果",
                    meta=f"pages={len(context.parsed_document.pages)} · doc={context.parsed_document.id}",
                    content=(
                        f"已将文档 {context.parsed_document.filename} 解析为 "
                        f"{len(context.parsed_document.pages)} 页，并作为科研助手的证据工具输出。"
                    ),
                    payload={
                        "parsed_document": context.parsed_document.model_dump(mode="json"),
                        "document_index_result": context.document_index_result,
                    },
                )
            )
        if context.chart_result:
            chart = getattr(context.chart_result, "chart", None)
            chart_metadata = getattr(chart, "metadata", {}) or {} if chart is not None else {}
            chart_image_path = str(chart_metadata.get("image_path") or "").strip()
            chart_summary = getattr(chart, "summary", None) or "已完成图表结构化理解。"
            figure_answer = (getattr(context.chart_result, "answer", None) or "").strip()
            if figure_answer:
                display_content = figure_answer
                if chart_image_path:
                    display_content = f"{display_content}\n\n图片路径：{chart_image_path}"
                messages.append(
                    _message(
                        role="assistant",
                        kind="answer",
                        title="论文图表分析",
                        meta=f"chart_type={getattr(chart, 'chart_type', 'unknown')}",
                        content=display_content,
                        payload={
                            "chart": chart.model_dump(mode="json") if hasattr(chart, "model_dump") else None,
                            "graph_text": getattr(context.chart_result, "graph_text", None),
                            "image_path": chart_image_path or None,
                        },
                    )
                )
            else:
                if chart_image_path:
                    chart_summary = f"{chart_summary}\n图片路径：{chart_image_path}"
                messages.append(
                    _message(
                        role="assistant",
                        kind="notice",
                        title="图表理解结果",
                        meta=f"chart_type={getattr(chart, 'chart_type', 'unknown')}",
                        content=chart_summary,
                        payload={
                            "chart": chart.model_dump(mode="json") if hasattr(chart, "model_dump") else None,
                            "graph_text": getattr(context.chart_result, "graph_text", None),
                            "image_path": chart_image_path or None,
                        },
                    )
                )
        if context.qa_result:
            messages.append(
                _message(
                    role="assistant",
                    kind="answer",
                    title="研究集合回答",
                    meta=(
                        f"evidence={len(context.qa_result.qa.evidence_bundle.evidences)} · "
                        f"confidence={context.qa_result.qa.confidence if context.qa_result.qa.confidence is not None else 'empty'}"
                    ),
                    content=context.qa_result.qa.answer,
                    payload={"qa": context.qa_result.qa.model_dump(mode="json")},
                )
            )
        if context.general_answer:
            meta = context.general_answer_metadata or {}
            messages.append(
                _message(
                    role="assistant",
                    kind="answer",
                    title="通用回答",
                    meta=f"route=general_answer · confidence={meta.get('confidence', 'empty')}",
                    content=context.general_answer,
                    payload={"general_answer": meta},
                )
            )
        preference_recommendations_payload = None
        if context.preference_recommendation_result is not None:
            preference_recommendations_payload = context.preference_recommendation_result.model_dump(mode="json")
        if isinstance(preference_recommendations_payload, dict):
            recommendations = list(preference_recommendations_payload.get("recommendations") or [])
            topics_used = list((preference_recommendations_payload.get("metadata") or {}).get("topics_used") or [])
            resolved_sources = list((preference_recommendations_payload.get("metadata") or {}).get("resolved_sources") or [])
            topic_groups = list((preference_recommendations_payload.get("metadata") or {}).get("topic_groups") or [])
            content_lines: list[str] = []
            if topic_groups:
                for group in topic_groups[:4]:
                    topic_name = str(group.get("topic") or "其他").strip() or "其他"
                    papers = list(group.get("papers") or [])
                    content_lines.append(f"主题：{topic_name}")
                    for index, paper in enumerate(papers[:4], start=1):
                        paper_title = str(paper.get("title") or "").strip()
                        paper_source = str(paper.get("source") or "").strip()
                        paper_year = paper.get("year")
                        paper_url = str(paper.get("url") or "").strip()
                        paper_reason = str(paper.get("reason") or "").strip()
                        paper_explanation = str(paper.get("explanation") or "").strip()
                        meta_parts = [str(item) for item in (paper_year, paper_source) if item]
                        meta = f" ({', '.join(meta_parts)})" if meta_parts else ""
                        content_lines.append(f"{index}. {paper_title}{meta}")
                        if paper_url:
                            content_lines.append(f"链接：{paper_url}")
                        if paper_reason:
                            content_lines.append(f"推荐理由：{paper_reason}")
                        if paper_explanation:
                            content_lines.append(f"论文讲解：{paper_explanation}")
                    content_lines.append("")
                if content_lines and not content_lines[-1].strip():
                    content_lines.pop()
            else:
                for index, item in enumerate(recommendations[:5], start=1):
                    title = str(item.get("title") or "").strip()
                    source = str(item.get("source") or "").strip()
                    year = item.get("year")
                    url = str(item.get("url") or "").strip()
                    reason = str(item.get("reason") or "").strip()
                    meta_parts = [str(value) for value in (year, source) if value]
                    meta = f" ({', '.join(meta_parts)})" if meta_parts else ""
                    content_lines.append(f"{index}. {title}{meta}")
                    if url:
                        content_lines.append(f"链接：{url}")
                    if reason:
                        content_lines.append(f"推荐理由：{reason}")
            messages.append(
                _message(
                    role="assistant",
                    kind="notice",
                    title="长期兴趣论文推荐",
                    meta=(
                        f"recommended={len(recommendations)}"
                        f"{' · topics=' + ', '.join(topics_used[:3]) if topics_used else ''}"
                        f"{' · sources=' + ', '.join(resolved_sources[:3]) if resolved_sources else ''}"
                    ),
                    content="\n".join(content_lines) or "已生成基于长期兴趣的论文推荐。",
                    payload={"recommendations": preference_recommendations_payload},
                )
            )
        paper_analysis_payload = None
        if context.paper_analysis_result is not None:
            paper_analysis_payload = context.paper_analysis_result.model_dump(mode="json")
        if isinstance(paper_analysis_payload, dict):
            focus = str(paper_analysis_payload.get("focus") or "analysis")
            messages.append(
                _message(
                    role="assistant",
                    kind="notice",
                    title="论文分析结果",
                    meta=f"focus={focus}",
                    content=str(paper_analysis_payload.get("answer") or "").strip() or "已生成基于所选论文的分析结果。",
                    payload={"paper_analysis": paper_analysis_payload},
                )
            )
        compression_payload = context.compressed_context_summary
        if isinstance(compression_payload, dict):
            messages.append(
                _message(
                    role="assistant",
                    kind="notice",
                    title="上下文压缩摘要",
                    meta=(
                        f"papers={compression_payload.get('paper_count', 0)} · "
                        f"summaries={compression_payload.get('summary_count', 0)}"
                    ),
                    content="当前研究上下文已经压缩为更短的论文摘要视图，后续 QA、对比和推荐会复用它。",
                    payload={"context_compression": compression_payload},
                )
            )
        if workspace.status_summary or workspace.stop_reason:
            workspace_lines = []
            if workspace.stop_reason:
                workspace_lines.append(f"stop_reason: {workspace.stop_reason}")
            workspace_lines.extend(f"- {item}" for item in workspace.next_actions[:4])
            messages.append(
                _message(
                    role="assistant",
                    kind="notice",
                    title="Research Workspace",
                    meta=workspace.status_summary,
                    content="\n".join(workspace_lines),
                    payload={"workspace": workspace.model_dump(mode="json")},
                )
            )
        trace_lines = [
            f"{step.step_index}. {step.agent} · {step.phase}:{step.action_name} · {step.status} · {step.observation}"
            for step in trace
        ]
        messages.append(
            _message(
                role="assistant",
                kind="notice",
                title="Agent 决策轨迹",
                meta=f"{len(trace)} step(s)",
                content="\n".join(trace_lines),
                payload={"trace": [step.model_dump(mode="json") for step in trace]},
            )
        )
        return messages

    def build_next_actions(
        self,
        context: ResearchAgentToolContext,
        workspace: ResearchWorkspaceState,
        *,
        clarification_request: str | None = None,
    ) -> list[str]:
        actions: list[str] = list(workspace.next_actions)
        task = context.task
        if clarification_request:
            actions.insert(0, "补充更具体的研究子方向、评价维度、应用场景或时间范围后再继续。")
        if task and context.papers:
            actions.append(
                f"继续追问这个研究集合，{self._manager_display_name} 会基于已导入文献和候选论文池回答。"
            )
        if task and not task.imported_document_ids:
            actions.append("导入开放 PDF 后可以获得更强的 grounded QA 证据。")
        if task and task.todo_items:
            actions.append("执行或关闭自动 TODO，让研究空间持续补证据。")
        if context.parsed_document:
            actions.append("可以继续围绕刚解析的文档提问，或让助手补充相关领域文献。")
        if context.chart_result:
            actions.append("可以让助手把图表结论和相关论文证据合并分析。")
        if context.preference_recommendation_result is not None:
            actions.append("可以继续追问推荐列表里的某篇论文，或让助手按其中一个主题继续做深入调研。")
        if workspace.metadata.get("latest_paper_analysis"):
            actions.append("可以继续基于这组论文追问实验差异、适用场景、失败边界或下一步阅读建议。")
        if not actions:
            actions.append("换一个更具体的研究目标，或扩大时间窗口和数据源。")
        deduped: list[str] = []
        for action in actions:
            normalized = action.strip()
            if not normalized or normalized in deduped:
                continue
            deduped.append(normalized)
        return deduped[:5]
