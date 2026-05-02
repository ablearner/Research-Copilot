from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from domain.schemas.research import PaperCandidate, ResearchCluster, ResearchReport

logger = logging.getLogger(__name__)

_CLUSTER_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("路径规划与导航", ("path planning", "trajectory", "navigation", "route", "路径规划", "导航")),
    ("检测与感知", ("detection", "perception", "tracking", "识别", "检测", "感知")),
    ("群体协同与多智能体", ("swarm", "multi-agent", "cooperative", "群体", "协同")),
    ("控制与定位", ("control", "localization", "state estimation", "定位", "控制")),
    ("遥感与场景理解", ("remote sensing", "segmentation", "mapping", "遥感", "建图")),
)

_SURVEY_PROMPT = (
    "你是一个学术文献综述写作专家。请根据以下论文信息，生成一份结构化的中文文献调研报告。\n\n"
    "研究主题：{topic}\n"
    "写作风格：{style}\n"
    "论文数量：{paper_count}\n\n"
    "论文信息（JSON）：\n{papers_json}\n\n"
    "论文分组：\n{clusters_json}\n\n"
    "要求：\n"
    "- 使用中文撰写（论文标题、专有名词可保留英文）\n"
    "- 包含以下章节：研究背景、核心问题（按分组）、方法对比、关键发现、代表论文逐篇解读、研究空白与未来方向、证据边界与局限\n"
    "- 每篇论文用 [P1][P2] 等标记引用\n"
    "- 方法对比部分使用 markdown 表格\n"
    "- 对每篇论文的方法侧重点给出中文概括（不要直接复制英文摘要）\n"
    "- 保持学术严谨性，指出证据边界\n"
    "- 总字数不少于 {min_length} 字"
)

_SURVEY_PROMPT_EN = (
    "You are an expert academic survey writer. Based on the following papers, produce a structured literature review report in English.\n\n"
    "Research topic: {topic}\n"
    "Writing style: {style}\n"
    "Paper count: {paper_count}\n\n"
    "Papers (JSON-like list):\n{papers_json}\n\n"
    "Paper groups:\n{clusters_json}\n\n"
    "Requirements:\n"
    "- Write in English, while keeping paper titles and technical identifiers in their original form\n"
    "- Include these sections: Background, Core Problems (grouped), Method Comparison, Key Findings, Representative Papers, Research Gaps and Future Directions, Evidence Boundaries and Limitations\n"
    "- Cite each paper with markers like [P1][P2]\n"
    "- Use a markdown table for method comparison\n"
    "- Summarize each paper's method focus instead of copying raw abstract text\n"
    "- Preserve academic caution and uncertainty where needed\n"
    "- Total length should be at least {min_length} words"
)


class _LLMSurveyResponse(BaseModel):
    markdown: str = Field(description="完整的 markdown 格式文献调研报告")


def _short_summary(abstract: str) -> str:
    cleaned = " ".join(abstract.strip().split())
    if not cleaned:
        return "摘要缺失。"
    segments = [segment.strip() for segment in re.split(r"(?<=[。！？.!?])\s+", cleaned) if segment.strip()]
    if len(segments) >= 2:
        return " ".join(segments[:2])[:320]
    return cleaned[:320]


def _is_chinese_language(language: str | None) -> bool:
    return str(language or "").strip().lower().startswith("zh")


def _effective_min_length(llm_adapter: Any | None, requested_min_length: int) -> int:
    model_name = str(getattr(llm_adapter, "model", "") or "").lower()
    if "mini" in model_name:
        return min(requested_min_length, 600)
    return requested_min_length


class SurveyWriter:
    """Produce literature survey reports using LLM with deterministic fallback.
    
    When llm_adapter is provided, uses LLM for intelligent report generation.
    Falls back to template-based heuristic generation when LLM is unavailable.
    """

    def __init__(self, *, llm_adapter: Any | None = None) -> None:
        self.llm_adapter = llm_adapter

    def generate(
        self,
        *,
        topic: str,
        task_id: str | None,
        papers: list[PaperCandidate],
        warnings: list[str] | None = None,
        style: str = "academic",
        min_length: int = 800,
        include_citations: bool = True,
        language: str = "zh-CN",
    ) -> ResearchReport:
        """Synchronous generate — uses heuristic template logic."""
        return self._heuristic_generate(
            topic=topic, task_id=task_id, papers=papers, warnings=warnings,
            style=style, min_length=min_length, include_citations=include_citations, language=language,
        )

    async def generate_async(
        self,
        *,
        topic: str,
        task_id: str | None,
        papers: list[PaperCandidate],
        warnings: list[str] | None = None,
        style: str = "academic",
        min_length: int = 800,
        include_citations: bool = True,
        language: str = "zh-CN",
    ) -> ResearchReport:
        """Async generate — uses LLM if available, falls back to heuristic."""
        min_length = _effective_min_length(self.llm_adapter, min_length)
        if self.llm_adapter is not None and papers:
            try:
                return await self._llm_generate(
                    topic=topic, task_id=task_id, papers=papers, warnings=warnings,
                    style=style, min_length=min_length, include_citations=include_citations, language=language,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM survey generation failed, falling back to heuristic: %s", exc)
        return self._heuristic_generate(
            topic=topic, task_id=task_id, papers=papers, warnings=warnings,
            style=style, min_length=min_length, include_citations=include_citations, language=language,
        )

    async def _llm_generate(
        self,
        *,
        topic: str,
        task_id: str | None,
        papers: list[PaperCandidate],
        warnings: list[str] | None = None,
        style: str = "academic",
        min_length: int = 800,
        include_citations: bool = True,
        language: str = "zh-CN",
    ) -> ResearchReport:
        warnings = warnings or []
        papers_with_summary = [
            paper if paper.summary else paper.model_copy(update={"summary": _short_summary(paper.abstract)})
            for paper in papers
        ]
        clusters = self._build_clusters(papers_with_summary)
        source_counts = Counter(paper.source for paper in papers_with_summary)
        citation_map = {
            paper.paper_id: f"P{index}"
            for index, paper in enumerate(papers_with_summary, start=1)
        }

        # Build papers JSON for LLM
        papers_json = "\n".join(
            f'  [{citation_map[p.paper_id]}] title="{p.title}", year={p.year or "n/a"}, '
            f'source={p.source}, abstract="{(p.abstract or "")[:180]}"'
            for p in papers_with_summary
        )
        clusters_json = "\n".join(
            f'  - {c.name}: [{", ".join(citation_map.get(pid, "?") for pid in c.paper_ids)}]'
            for c in clusters
        )

        result = await self.llm_adapter.generate_structured(
            prompt=_SURVEY_PROMPT if _is_chinese_language(language) else _SURVEY_PROMPT_EN,
            input_data={
                "topic": topic,
                "style": style,
                "paper_count": str(len(papers_with_summary)),
                "papers_json": papers_json,
                "clusters_json": clusters_json,
                "min_length": str(min_length),
            },
            response_model=_LLMSurveyResponse,
        )

        # Prepend metadata header
        header_lines = [
            f"# 文献调研报告：{topic}",
            "",
            f"- 生成时间：{datetime.now(UTC).isoformat()}",
            f"- 命中文章数：{len(papers_with_summary)}",
            f"- 数据源分布：{', '.join(f'{source}={count}' for source, count in source_counts.items()) or 'empty'}",
            f"- 写作风格：{style}",
            "",
        ]

        # Check if LLM already included the header
        llm_markdown = result.markdown.strip()
        if llm_markdown.startswith("# 文献调研报告"):
            markdown = llm_markdown
        else:
            markdown = "\n".join(header_lines) + llm_markdown

        # Append citation mapping
        mapping_lines = ["", "## 论文映射"]
        for paper in papers_with_summary:
            label = citation_map.get(paper.paper_id, "P?")
            mapping_lines.append(
                f"- [{label}] {paper.title} | {paper.year or 'n/a'} | {paper.source} | {paper.url or paper.pdf_url or 'link-unavailable'}"
            )
        if warnings:
            mapping_lines.extend(["", "## 查询告警"])
            mapping_lines.extend([f"- {warning}" for warning in warnings])
        markdown += "\n".join(mapping_lines)

        highlights = [
            f"{paper.title}（{paper.year or 'n/a'}，{paper.source}）: {paper.summary or _short_summary(paper.abstract)}"
            for paper in papers_with_summary[:5]
        ]
        gaps: list[str] = []
        if len(papers_with_summary) < 5:
            gaps.append("当前主题命中的候选论文偏少，建议扩大时间窗口或补充更多关键词。")
        if warnings:
            gaps.append("部分数据源查询失败，当前综述可能不完整。")

        return ResearchReport(
            report_id=f"report_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
            task_id=task_id,
            topic=topic,
            generated_at=datetime.now(UTC).isoformat(),
            markdown=markdown,
            paper_count=len(papers_with_summary),
            source_counts=dict(source_counts),
            highlights=highlights,
            clusters=clusters,
            gaps=gaps,
            metadata={"writer": "SurveyWriter+LLM"},
        )

    def _build_clusters(self, papers: list[PaperCandidate]) -> list[ResearchCluster]:
        clusters: list[ResearchCluster] = []
        remaining_ids = {paper.paper_id for paper in papers}
        for cluster_name, keywords in _CLUSTER_RULES:
            matched_ids = [
                paper.paper_id
                for paper in papers
                if paper.paper_id in remaining_ids and any(keyword.lower() in f"{paper.title} {paper.abstract}".lower() for keyword in keywords)
            ]
            if matched_ids:
                remaining_ids.difference_update(matched_ids)
                clusters.append(
                    ResearchCluster(
                        name=cluster_name,
                        paper_ids=matched_ids,
                        description=f"该分组包含 {len(matched_ids)} 篇与 {cluster_name} 相关的候选论文。",
                    )
                )
        if remaining_ids:
            clusters.append(
                ResearchCluster(
                    name="其他相关工作",
                    paper_ids=[paper.paper_id for paper in papers if paper.paper_id in remaining_ids],
                    description="未明显落入预设分组的论文。",
                )
            )
        return clusters

    def _heuristic_generate(
        self,
        *,
        topic: str,
        task_id: str | None,
        papers: list[PaperCandidate],
        warnings: list[str] | None = None,
        style: str = "academic",
        min_length: int = 800,
        include_citations: bool = True,
        language: str = "zh-CN",
    ) -> ResearchReport:
        if not _is_chinese_language(language):
            return ResearchReport(
                report_id=f"report_{datetime.now(UTC).timestamp()}",
                task_id=task_id,
                topic=topic,
                generated_at=datetime.now(UTC).isoformat(),
                markdown=(
                    f"# Literature Review: {topic}\n\n"
                    f"- Generated at: {datetime.now(UTC).isoformat()}\n"
                    f"- Paper count: {len(papers)}\n\n"
                    "## Representative Papers\n"
                    + "\n".join(
                        f"- {paper.title} ({paper.year or 'n/a'}, {paper.source}): {paper.summary or _short_summary(paper.abstract)}"
                        for paper in papers[:8]
                    )
                ),
                paper_count=len(papers),
                source_counts=dict(Counter(paper.source for paper in papers)),
                highlights=[
                    f"{paper.title}: {paper.summary or _short_summary(paper.abstract)}"
                    for paper in papers[:5]
                ],
                clusters=self._build_clusters(papers),
                gaps=["Heuristic English fallback is concise; enable LLM generation for a fuller structured review."],
                metadata={"writer": "SurveyWriterHeuristicEN", "language": language},
            )
        warnings = warnings or []
        papers_with_summary = [
            paper if paper.summary else paper.model_copy(update={"summary": _short_summary(paper.abstract)})
            for paper in papers
        ]
        clusters = self._build_clusters(papers_with_summary)

        source_counts = Counter(paper.source for paper in papers_with_summary)
        highlights = [
            f"{paper.title}（{paper.year or 'n/a'}，{paper.source}）: {paper.summary or _short_summary(paper.abstract)}"
            for paper in papers_with_summary[:5]
        ]
        gaps: list[str] = []
        if len(papers_with_summary) < 5:
            gaps.append("当前主题命中的候选论文偏少，建议扩大时间窗口或补充更多关键词。")
        if not any(paper.pdf_url for paper in papers_with_summary):
            gaps.append("当前结果多为元数据级证据，缺少可直接入库的开放 PDF。")
        if warnings:
            gaps.append("部分数据源查询失败，当前综述可能不完整。")
        citation_map = {
            paper.paper_id: f"P{index}"
            for index, paper in enumerate(papers_with_summary, start=1)
        }
        cluster_names = [cluster.name for cluster in clusters] or ["相关研究"]
        style_hint = {
            "academic": "以下综述采用相对学术化的表述，强调问题定义、方法比较与证据边界。",
            "concise": "以下综述尽量压缩表述，但仍保留问题、方法、发现和研究空白四个层次。",
            "beginner": "以下综述尽量面向初学者解释术语，同时保留必要的论文出处。",
        }.get(style, "以下综述采用相对学术化的表述，强调问题定义、方法比较与证据边界。")

        markdown_lines = [
            f"# 文献调研报告：{topic}",
            "",
            f"- 生成时间：{datetime.now(UTC).isoformat()}",
            f"- 命中文章数：{len(papers_with_summary)}",
            f"- 数据源分布：{', '.join(f'{source}={count}' for source, count in source_counts.items()) or 'empty'}",
            f"- 写作风格：{style}",
            "",
            "## 研究背景",
            (
                f"围绕“{topic}”的当前候选文献显示，该方向并不是单一技术主题，而是由"
                f"{'、'.join(cluster_names[:4])}等子问题共同组成。{style_hint}"
            ),
        ]
        if papers_with_summary:
            top_labels = self._citation_suffix(papers_with_summary[: min(4, len(papers_with_summary))], citation_map, include_citations)
            markdown_lines.append(
                f"从时间和来源分布看，当前结果既包含偏快速发布的预印本，也包含具备一定引用积累的数据库条目，"
                f"说明该研究方向正在持续演化，且方法、实验设计与应用场景都在同步扩展{top_labels}。"
            )
        else:
            markdown_lines.append("当前尚未检索到足够的论文结果，因此无法形成可靠综述。")

        markdown_lines.extend(["", "## 核心问题"])
        if clusters:
            for cluster in clusters[:4]:
                cluster_papers = [
                    paper for paper in papers_with_summary if paper.paper_id in set(cluster.paper_ids)
                ]
                cluster_summary = self._cluster_problem_paragraph(
                    cluster_name=cluster.name,
                    papers=cluster_papers,
                    citation_map=citation_map,
                    include_citations=include_citations,
                )
                markdown_lines.append(f"### {cluster.name}")
                markdown_lines.append(cluster_summary)
        else:
            markdown_lines.append("当前候选集合不足以分解出稳定的核心问题结构。")

        markdown_lines.extend(["", "## 方法对比"])
        if papers_with_summary:
            markdown_lines.extend(
                self._comparison_table_lines(
                    papers=papers_with_summary,
                    citation_map=citation_map,
                    include_citations=include_citations,
                )
            )
            markdown_lines.append("")
            for paper in papers_with_summary[:8]:
                citation = self._paper_citation(paper, citation_map, include_citations)
                markdown_lines.append(
                    f"- **{paper.title}**：{self._method_comparison_sentence(paper)} {citation}".rstrip()
                )
        else:
            markdown_lines.append("- 当前没有可比较的方法。")

        markdown_lines.extend(["", "## 关键发现"])
        if papers_with_summary:
            for paper in papers_with_summary[:5]:
                citation = self._paper_citation(paper, citation_map, include_citations)
                markdown_lines.append(
                    f"- {paper.title} 表明 {paper.summary or _short_summary(paper.abstract)} {citation}".rstrip()
                )
        else:
            markdown_lines.append("- 暂无关键发现。")

        markdown_lines.extend(["", "## 代表论文逐篇解读"])
        if papers_with_summary:
            for paper in papers_with_summary[: min(6, len(papers_with_summary))]:
                citation = self._paper_citation(paper, citation_map, include_citations)
                markdown_lines.append(f"### {paper.title}")
                markdown_lines.append(
                    self._paper_deep_dive_paragraph(
                        paper=paper,
                        citation=citation,
                    )
                )
        else:
            markdown_lines.append("当前还没有足够的代表论文可供逐篇解读。")

        markdown_lines.extend(["", "## 研究空白与未来方向"])
        markdown_lines.extend(
            [
                f"- {gap}"
                for gap in (
                    gaps
                    or [
                        "现有候选论文在问题设置和评价方式上仍存在明显异质性，后续需要围绕统一 benchmark、可复现实现和更细粒度消融实验继续补足。"
                    ]
                )
            ]
        )
        if papers_with_summary:
            markdown_lines.append(
                f"- 从当前论文集合看，后续工作尤其值得关注方法泛化能力、实验设置可比性以及开放代码/开放 PDF 的覆盖度"
                f"{self._citation_suffix(papers_with_summary[: min(4, len(papers_with_summary))], citation_map, include_citations)}。"
            )

        markdown_lines.extend(["", "## 证据边界与局限"])
        if papers_with_summary:
            markdown_lines.extend(
                [
                    f"- 当前候选集合包含 {len(papers_with_summary)} 篇论文，但其中真正完成全文导入或开放获取的比例可能仍然有限，"
                    "这会影响对方法细节、消融实验和失败案例的深入比较。",
                    f"- 现有结果覆盖了 {'、'.join(cluster_names[:4]) or '若干'} 个子方向，但不同论文的任务定义、评价指标和数据集设置并不统一，"
                    "因此结论更适合作为研究路线图，而不是直接给出绝对优劣排序。",
                    f"- 从摘要级信息可以较好识别研究目标与方法定位，但对于实现复杂度、代码可复现性和统计显著性等问题，"
                    f"仍需要进一步阅读全文核实{self._citation_suffix(papers_with_summary[: min(3, len(papers_with_summary))], citation_map, include_citations)}。",
                ]
            )
        else:
            markdown_lines.append("- 当前缺少足够论文，无法判断证据边界。")

        markdown_lines.extend(["", "## 论文映射"])
        for paper in papers_with_summary:
            label = citation_map.get(paper.paper_id, "P?")
            markdown_lines.append(
                f"- [{label}] {paper.title} | {paper.year or 'n/a'} | {paper.source} | {paper.url or paper.pdf_url or 'link-unavailable'}"
            )
        if warnings:
            markdown_lines.extend(["", "## 查询告警"])
            markdown_lines.extend([f"- {warning}" for warning in warnings])

        markdown = "\n".join(markdown_lines)
        markdown = self._ensure_minimum_length(
            markdown=markdown,
            papers=papers_with_summary,
            citation_map=citation_map,
            include_citations=include_citations,
            min_length=min_length,
        )

        return ResearchReport(
            report_id=f"report_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
            task_id=task_id,
            topic=topic,
            generated_at=datetime.now(UTC).isoformat(),
            markdown=markdown,
            paper_count=len(papers_with_summary),
            source_counts=dict(source_counts),
            highlights=highlights,
            clusters=clusters,
            gaps=gaps,
            metadata={"writer": "heuristic"},
        )

    def _paper_citation(self, paper: PaperCandidate, citation_map: dict[str, str], include_citations: bool) -> str:
        if not include_citations:
            return ""
        return f"[{citation_map.get(paper.paper_id, 'P?')}]"

    def _citation_suffix(
        self,
        papers: list[PaperCandidate],
        citation_map: dict[str, str],
        include_citations: bool,
    ) -> str:
        if not include_citations or not papers:
            return ""
        return " " + "".join(self._paper_citation(paper, citation_map, include_citations) for paper in papers)

    def _cluster_problem_paragraph(
        self,
        *,
        cluster_name: str,
        papers: list[PaperCandidate],
        citation_map: dict[str, str],
        include_citations: bool,
    ) -> str:
        if not papers:
            return f"当前候选集合中与“{cluster_name}”直接相关的证据较少，暂时只能将其视为潜在问题方向。"
        exemplar = papers[: min(3, len(papers))]
        examples = "；".join(
            f"{paper.title} 关注 {paper.summary or _short_summary(paper.abstract)} {self._paper_citation(paper, citation_map, include_citations)}".rstrip()
            for paper in exemplar
        )
        return (
            f"在“{cluster_name}”这一分组中，论文主要围绕任务定义、方法路径与实验验证边界展开。"
            f"{examples}。这些工作共同反映出该子方向仍在持续寻找更稳定、更可复现的解决路径。"
        )

    def _method_comparison_sentence(self, paper: PaperCandidate) -> str:
        base = paper.summary or _short_summary(paper.abstract)
        venue_text = f"{paper.venue}，" if paper.venue else ""
        year_text = f"{paper.year} 年" if paper.year else "近期"
        return f"{year_text}{venue_text}该工作代表的方法侧重点可以概括为：{base}"

    def _comparison_table_lines(
        self,
        *,
        papers: list[PaperCandidate],
        citation_map: dict[str, str],
        include_citations: bool,
    ) -> list[str]:
        lines = [
            "| 论文 | 方法侧重 | 证据形态 | 当前局限 |",
            "| --- | --- | --- | --- |",
        ]
        for paper in papers[:6]:
            citation = self._paper_citation(paper, citation_map, include_citations)
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{paper.title} {citation}".strip(),
                        self._method_axis(paper),
                        self._evidence_axis(paper),
                        self._limitation_axis(paper),
                    ]
                )
                + " |"
            )
        return lines

    def _paper_deep_dive_paragraph(self, *, paper: PaperCandidate, citation: str) -> str:
        summary = paper.summary or _short_summary(paper.abstract)
        method_axis = self._method_axis(paper)
        evidence_axis = self._evidence_axis(paper)
        limitation_axis = self._limitation_axis(paper)
        return (
            f"这篇论文的核心价值在于 {method_axis}。从摘要可提炼出的主要内容是：{summary}。"
            f"如果把它放回整个主题脉络中看，它提供的证据更偏向于 {evidence_axis}，"
            f"因此适合作为综述中的代表性支撑点之一；但其当前可见局限也包括 {limitation_axis}。"
            f" {citation}".rstrip()
        )

    def _method_axis(self, paper: PaperCandidate) -> str:
        text = f"{paper.title} {paper.abstract}".lower()
        if any(keyword in text for keyword in ("survey", "review", "taxonomy", "综述")):
            return "对现有方法进行系统梳理和分类比较"
        if any(keyword in text for keyword in ("reinforcement", "learning", "policy", "agent")):
            return "利用学习式或 agent 驱动框架改进决策过程"
        if any(keyword in text for keyword in ("retrieval", "evidence", "citation", "检索", "证据")):
            return "强调检索、证据控制和引用链路"
        if any(keyword in text for keyword in ("benchmark", "evaluation", "experiment", "dataset", "实验")):
            return "围绕 benchmark、评价协议和实验验证进行设计"
        return "提出特定任务下的方法框架并尝试验证其有效性"

    def _evidence_axis(self, paper: PaperCandidate) -> str:
        text = f"{paper.title} {paper.abstract}".lower()
        if any(keyword in text for keyword in ("survey", "review", "taxonomy", "综述")):
            return "文献整理与方法归纳"
        if any(keyword in text for keyword in ("benchmark", "evaluation", "experiment", "dataset", "ablation", "实验")):
            return "实验或 benchmark 结果"
        if any(keyword in text for keyword in ("case study", "deployment", "real-world", "application")):
            return "案例验证或应用场景分析"
        return "摘要级方法描述"

    def _limitation_axis(self, paper: PaperCandidate) -> str:
        limitations: list[str] = []
        if not paper.pdf_url and not paper.is_open_access:
            limitations.append("开放全文可得性有限")
        if not paper.citations:
            limitations.append("引用积累仍然有限")
        if not any(keyword in f"{paper.title} {paper.abstract}".lower() for keyword in ("experiment", "benchmark", "dataset", "evaluation", "实验")):
            limitations.append("摘要中缺少明确实验细节")
        return "、".join(limitations) or "需要进一步阅读全文核对实验与实现细节"

    def _ensure_minimum_length(
        self,
        *,
        markdown: str,
        papers: list[PaperCandidate],
        citation_map: dict[str, str],
        include_citations: bool,
        min_length: int,
    ) -> str:
        if len(markdown) >= min_length:
            return markdown
        lines = [markdown, "", "## 结构化阅读摘记"]
        for paper in papers:
            if len("\n".join(lines)) >= min_length:
                break
            citation = self._paper_citation(paper, citation_map, include_citations)
            lines.append(
                f"- {paper.title}：作者为 {', '.join(paper.authors[:4]) or 'unknown'}，"
                f"来源于 {paper.source}，其摘要可提炼为“{paper.summary or _short_summary(paper.abstract)}” {citation}".rstrip()
            )
        return "\n".join(lines)


# Compatibility alias for the previous agent-like name.
SurveyWriterAgent = SurveyWriter
