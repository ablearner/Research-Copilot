from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from domain.schemas.research import PaperCandidate, ResearchTask


@dataclass(slots=True)
class PaperSelectionScope:
    scope_mode: str
    explicit_scope: bool = False
    paper_ids: list[str] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)
    papers: list[PaperCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def selected_titles(self) -> list[str]:
        return [paper.title for paper in self.papers]


class PaperSelectorService:
    """Resolve paper/document scope for research QA without changing task storage semantics."""

    def resolve_qa_scope(
        self,
        *,
        task: ResearchTask,
        papers: list[PaperCandidate],
        requested_paper_ids: list[str] | None = None,
        requested_document_ids: list[str] | None = None,
    ) -> PaperSelectionScope:
        requested_paper_ids = self._dedupe(requested_paper_ids or [])
        requested_document_ids = self._dedupe(requested_document_ids or [])
        explicit_scope = bool(requested_paper_ids or requested_document_ids)
        paper_by_id = {paper.paper_id: paper for paper in papers}
        imported_document_ids = {
            document_id.strip()
            for document_id in task.imported_document_ids
            if str(document_id).strip()
        }
        document_to_paper: dict[str, PaperCandidate] = {}
        for paper in papers:
            document_id = self._paper_document_id(paper)
            if document_id and document_id not in document_to_paper:
                document_to_paper[document_id] = paper

        selected_papers: list[PaperCandidate] = []
        selected_document_ids: list[str] = []
        warnings: list[str] = []
        missing_paper_ids: list[str] = []
        missing_document_ids: list[str] = []
        metadata_only_paper_ids: list[str] = []

        def add_paper(paper: PaperCandidate) -> None:
            if paper.paper_id not in {item.paper_id for item in selected_papers}:
                selected_papers.append(paper)

        def add_document(document_id: str) -> None:
            if document_id and document_id not in selected_document_ids:
                selected_document_ids.append(document_id)

        if requested_paper_ids:
            for paper_id in requested_paper_ids:
                paper = paper_by_id.get(paper_id)
                if paper is None:
                    warnings.append(f"未在当前研究任务中找到论文：{paper_id}")
                    missing_paper_ids.append(paper_id)
                    continue
                add_paper(paper)
                document_id = self._paper_document_id(paper)
                if document_id and document_id in imported_document_ids:
                    add_document(document_id)
                else:
                    warnings.append(f"论文尚未完成全文导入，当前只能基于元数据回答：{paper.title}")
                    metadata_only_paper_ids.append(paper.paper_id)

        if requested_document_ids:
            for document_id in requested_document_ids:
                if document_id not in imported_document_ids:
                    warnings.append(f"当前研究任务未登记该文档：{document_id}")
                    missing_document_ids.append(document_id)
                    continue
                add_document(document_id)
                paper = document_to_paper.get(document_id)
                if paper is not None:
                    add_paper(paper)

        if not explicit_scope:
            selected_document_ids = self._dedupe([doc_id for doc_id in imported_document_ids if doc_id])
            selected_papers = [
                paper
                for paper in papers
                if self._paper_document_id(paper) in set(selected_document_ids)
            ]
            scope_mode = "all_imported"
            if not selected_papers and papers:
                selected_papers = list(papers)
                scope_mode = "metadata_only"
        else:
            if requested_document_ids and selected_document_ids:
                scope_mode = "selected_documents"
            elif selected_document_ids:
                scope_mode = "selected_papers"
            else:
                scope_mode = "metadata_only"

        return PaperSelectionScope(
            scope_mode=scope_mode,
            explicit_scope=explicit_scope,
            paper_ids=[paper.paper_id for paper in selected_papers],
            document_ids=selected_document_ids,
            papers=selected_papers,
            warnings=warnings,
            metadata={
                "requested_paper_ids": requested_paper_ids,
                "requested_document_ids": requested_document_ids,
                "selected_titles": [paper.title for paper in selected_papers],
                "imported_paper_ids": [
                    paper.paper_id
                    for paper in selected_papers
                    if self._paper_document_id(paper) in imported_document_ids
                ],
                "metadata_only_paper_ids": metadata_only_paper_ids,
                "missing_paper_ids": missing_paper_ids,
                "missing_document_ids": missing_document_ids,
                "matched_document_ids": list(selected_document_ids),
                "paper_count": len(selected_papers),
                "document_count": len(selected_document_ids),
                "selection_summary": self._selection_summary(
                    scope_mode=scope_mode,
                    selected_papers=selected_papers,
                    selected_document_ids=selected_document_ids,
                    metadata_only_paper_ids=metadata_only_paper_ids,
                ),
            },
        )

    def _paper_document_id(self, paper: PaperCandidate) -> str | None:
        value = paper.metadata.get("document_id")
        normalized = str(value).strip() if value is not None else ""
        return normalized or None

    def _dedupe(self, items: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            normalized = str(item).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    def _selection_summary(
        self,
        *,
        scope_mode: str,
        selected_papers: list[PaperCandidate],
        selected_document_ids: list[str],
        metadata_only_paper_ids: list[str],
    ) -> str:
        titles = "；".join(paper.title for paper in selected_papers[:4]) or "未命中具体论文"
        summary = (
            f"scope={scope_mode}; papers={len(selected_papers)}; documents={len(selected_document_ids)}; "
            f"titles={titles}"
        )
        if metadata_only_paper_ids:
            summary += f"; metadata_only={len(metadata_only_paper_ids)}"
        return summary
