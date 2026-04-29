from __future__ import annotations

import re
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field

from domain.schemas.research import PaperSource


ResearchUserIntentName = Literal[
    "literature_search",
    "paper_import",
    "sync_to_zotero",
    "collection_qa",
    "single_paper_qa",
    "paper_comparison",
    "paper_recommendation",
    "figure_qa",
    "document_understanding",
    "general_answer",
    "general_follow_up",
]


class ResearchUserIntentResult(BaseModel):
    intent: ResearchUserIntentName = "general_follow_up"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    target_kind: Literal["none", "paper", "papers", "figure", "document", "collection"] = "none"
    resolved_paper_ids: list[str] = Field(default_factory=list)
    reference_type: Literal["none", "deictic", "ordinal", "alias", "title", "mixed"] = "none"
    needs_clarification: bool = False
    clarification_question: str | None = None
    rationale: str = ""
    markers: list[str] = Field(default_factory=list)
    source: Literal["heuristic", "llm"] = "heuristic"
    extracted_topic: str = Field(default="", description="Research topic with source/constraint phrases stripped")
    source_constraints: list[str] = Field(default_factory=list, description="Extracted source constraints, e.g. ['arxiv']")


_FIGURE_MARKERS = ("图", "图表", "figure", "fig.", "chart", "plot", "曲线", "横轴", "纵轴")
_COMPARE_MARKERS = ("对比", "比较", "区别", "差异", "compare", "comparison", " vs ", "versus")
_RECOMMEND_MARKERS = ("推荐", "先读", "值得读", "优先阅读", "recommend", "read first", "worth reading")
_IMPORT_MARKERS = ("导入", "导进", "导入到", "导入工作区", "入库", "ingest", "import")
_WORKSPACE_IMPORT_MARKERS = (
    "工作区",
    "本地",
    "问答",
    "qa",
    "grounded",
    "evidence",
    "证据",
    "入库",
    "索引",
    "workspace",
    "local",
)
_ZOTERO_SYNC_MARKERS = ("zotero", "citation manager", "文献库", "参考文献库", "书目库")
_SAVE_ADD_MARKERS = ("保存", "加入", "添加", "同步", "save", "add", "sync")
_SEARCH_MARKERS = (
    "检索",
    "搜索",
    "找",
    "调研",
    "论文",
    "文献",
    "paper",
    "papers",
    "arxiv",
    "search",
    "survey",
    "find papers",
)
# Auto-derive canonical source names from the PaperSource Literal,
# plus common aliases so heuristic matching covers user-facing names.
_PAPER_SOURCES: tuple[str, ...] = get_args(PaperSource)  # ("arxiv", "openalex", ...)
_SOURCE_ALIASES: dict[str, str] = {
    "semantic scholar": "semantic_scholar",
    "google scholar": "google_scholar",
}
_SOURCE_NAMES: dict[str, str] = {
    **{src: src for src in _PAPER_SOURCES},
    **{src.replace("_", " "): src for src in _PAPER_SOURCES if "_" in src},
    **_SOURCE_ALIASES,
}
_SOURCE_CONSTRAINT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:在|从|用|通过|去)\s*(?P<src>" + "|".join(re.escape(k) for k in _SOURCE_NAMES) + r")\s*(?:上|中|里|搜索|搜|找|检索|查)", re.IGNORECASE),
    re.compile(r"(?:on|from|via|using|through)\s+(?P<src>" + "|".join(re.escape(k) for k in _SOURCE_NAMES) + r")\b", re.IGNORECASE),
    re.compile(r"(?:search|find|look)\s+(?:on|in|from)\s+(?P<src>" + "|".join(re.escape(k) for k in _SOURCE_NAMES) + r")\b", re.IGNORECASE),
)
_DOCUMENT_MARKERS = ("文档", "pdf", "文件", "上传", "解析", "document")
_SINGLE_PAPER_MARKERS = ("这篇", "该论文", "这篇论文", "第一篇", "第二篇", "method", "experiment")
_AMBIGUOUS_REFERENCES = ("这篇", "这张", "这个", "该论文", "它", "上一个", "上一张", "this paper", "this figure", "it")
_GENERAL_QA_MARKERS = ("天气", "翻译", "解释一下", "是什么", "怎么做", "why", "what is", "how to")
_GREETING_MARKERS = ("你好", "您好", "嗨", "hello", "hi", "hey")
_GENERAL_OPT_OUT_MARKERS = (
    "不看当前论文",
    "不看这些论文",
    "先不看论文",
    "脱离当前论文",
    "退出当前调研",
    "退出调研语境",
    "单纯和我打个招呼",
    "单纯打个招呼",
    "普通聊天",
    "闲聊一下",
    "不要结合当前论文",
    "ignore current papers",
    "ignore the current papers",
    "ignore current research",
    "just say hello",
    "just greet me",
)

def _extract_source_constraints(message: str) -> tuple[list[str], str]:
    """Extract source constraints from message, return (constraints, cleaned_message)."""
    constraints: list[str] = []
    cleaned = message
    for pattern in _SOURCE_CONSTRAINT_PATTERNS:
        new_cleaned = cleaned
        for match in reversed(list(pattern.finditer(cleaned))):
            src_key = match.group("src").lower().strip()
            canonical = _SOURCE_NAMES.get(src_key)
            if canonical and canonical not in constraints:
                constraints.append(canonical)
            new_cleaned = new_cleaned[:match.start()] + " " + new_cleaned[match.end():]
        cleaned = new_cleaned
    for name, canonical in _SOURCE_NAMES.items():
        if name in cleaned.lower() and canonical not in constraints:
            constraints.append(canonical)
            cleaned = re.sub(re.escape(name), " ", cleaned, flags=re.IGNORECASE)
    cleaned = " ".join(cleaned.split()).strip()
    return constraints, cleaned


_ORDINAL_PATTERNS = (
    re.compile(r"\bp(?:aper)?\s*([0-9]{1,2})\b", re.IGNORECASE),
    re.compile(r"第\s*([0-9]{1,2})\s*篇"),
    re.compile(r"第\s*([一二两三四五六七八九十]{1,3})\s*篇"),
)
_CHINESE_ORDINAL_MAP = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


class ResearchIntentResolver:
    """Lightweight intent parsing helper used as context, not as a new orchestration node."""

    name = "ResearchIntentResolver"

    def __init__(self, *, llm_adapter: Any | None = None) -> None:
        self.llm_adapter = llm_adapter

    async def resolve_async(
        self,
        *,
        message: str,
        has_task: bool,
        candidate_paper_count: int,
        candidate_papers: list[dict[str, Any]] | None = None,
        active_paper_ids: list[str],
        selected_paper_ids: list[str],
        has_visual_anchor: bool = False,
        has_document_input: bool = False,
    ) -> ResearchUserIntentResult:
        heuristic_result = self.resolve(
            message=message,
            has_task=has_task,
            candidate_paper_count=candidate_paper_count,
            candidate_papers=candidate_papers,
            active_paper_ids=active_paper_ids,
            selected_paper_ids=selected_paper_ids,
            has_visual_anchor=has_visual_anchor,
            has_document_input=has_document_input,
        )
        if self.llm_adapter is None or heuristic_result.confidence >= 0.8:
            return heuristic_result
        try:
            llm_result = await self.llm_adapter.generate_structured(
                prompt=(
                    "你是科研助手的轻量意图解析器。请根据用户输入、当前任务上下文、已选论文、活动论文焦点、"
                    "图像/文档输入情况，判断最可能的用户意图。"
                    "如果用户用“这篇”“第一篇”“p1”“标题简称”等方式引用候选论文，请结合 candidate_papers、"
                    "active_paper_ids、selected_paper_ids 解析出 resolved_paper_ids。"
                    "主路径依赖你的语义理解，不要机械依赖关键词或固定映射。"
                    "关键词和 marker 只能作为弱信号，不能机械匹配；请优先理解用户真实语义。"
                    "重要：如果用户提到数据源（如 arxiv、ieee、semantic scholar 等），"
                    "将其放入 source_constraints 列表，并在 extracted_topic 中剥离这些来源名称，"
                    "只保留纯粹的研究主题。例如“在arxiv上找LLM agent论文”→ extracted_topic='LLM agent', source_constraints=['arxiv']。"
                    "只返回结构化字段，不要执行任务。"
                ),
                input_data={
                    "message": message,
                    "has_task": has_task,
                    "candidate_paper_count": candidate_paper_count,
                    "candidate_papers": list(candidate_papers or [])[:20],
                    "active_paper_ids": active_paper_ids,
                    "selected_paper_ids": selected_paper_ids,
                    "has_visual_anchor": has_visual_anchor,
                    "has_document_input": has_document_input,
                    "heuristic_hint": heuristic_result.model_dump(mode="json"),
                },
                response_model=ResearchUserIntentResult,
            )
            return llm_result.model_copy(update={"source": "llm"})
        except Exception:
            return heuristic_result

    def resolve(
        self,
        *,
        message: str,
        has_task: bool,
        candidate_paper_count: int,
        candidate_papers: list[dict[str, Any]] | None = None,
        active_paper_ids: list[str],
        selected_paper_ids: list[str],
        has_visual_anchor: bool = False,
        has_document_input: bool = False,
    ) -> ResearchUserIntentResult:
        normalized = f" {message.strip().lower()} "
        markers: list[str] = []
        source_constraints, cleaned_message = _extract_source_constraints(message)
        resolved_ordinal_ids = self._resolve_ordinal_references(
            normalized,
            candidate_papers=list(candidate_papers or []),
        )
        if resolved_ordinal_ids:
            markers.append("ordinal_reference")

        def has_any(candidates: tuple[str, ...]) -> bool:
            found = [marker for marker in candidates if marker in normalized]
            markers.extend(found)
            return bool(found)

        def reference_type_for(*, resolved_ids: list[str]) -> Literal["none", "deictic", "ordinal", "alias", "title", "mixed"]:
            if resolved_ordinal_ids and self._has_ambiguous_reference(normalized):
                return "mixed"
            if resolved_ordinal_ids:
                return "ordinal"
            if resolved_ids and self._has_ambiguous_reference(normalized):
                return "deictic"
            return "none"

        def resolved_target_ids() -> list[str]:
            return list(dict.fromkeys([*resolved_ordinal_ids, *(selected_paper_ids or active_paper_ids)]))[:4]

        general_opt_out = has_any(_GENERAL_OPT_OUT_MARKERS)
        greeting_only = has_any(_GREETING_MARKERS) and not (
            candidate_paper_count and any(
                marker in normalized
                for marker in (
                    *_SEARCH_MARKERS,
                    *_COMPARE_MARKERS,
                    *_RECOMMEND_MARKERS,
                    *_SINGLE_PAPER_MARKERS,
                    "方法",
                    "讲解",
                    "解释",
                    "论文",
                    "paper",
                )
            )
        )
        if general_opt_out or greeting_only:
            return self._result(
                "general_answer",
                "none",
                0.9 if general_opt_out else 0.82,
                markers,
                "The user explicitly requested a general conversation turn rather than a research follow-up.",
            )
        if has_visual_anchor or has_any(_FIGURE_MARKERS):
            resolved_ids = resolved_target_ids()
            ambiguous = self._has_ambiguous_reference(normalized) and not resolved_ids
            return self._result(
                "figure_qa",
                "figure",
                0.86 if not ambiguous else 0.58,
                markers,
                "Question contains visual/figure markers.",
                resolved_paper_ids=resolved_ids,
                reference_type=reference_type_for(resolved_ids=resolved_ids),
                needs_clarification=ambiguous,
                clarification_question=(
                    "你想看哪篇论文里的图或系统框图？请用序号、标题，或先选中论文后再问。"
                    if ambiguous
                    else None
                ),
            )
        if has_document_input or has_any(_DOCUMENT_MARKERS):
            return self._result("document_understanding", "document", 0.82, markers, "Question targets an uploaded document.")
        if has_any(_COMPARE_MARKERS):
            return self._result("paper_comparison", "papers", 0.84, markers, "Question asks for comparison.")
        if has_any(_RECOMMEND_MARKERS):
            return self._result("paper_recommendation", "papers", 0.82, markers, "Question asks for recommendation or reading priority.")
        import_requested = has_any(_IMPORT_MARKERS)
        zotero_requested = has_any(_ZOTERO_SYNC_MARKERS)
        save_or_add_requested = has_any(_SAVE_ADD_MARKERS)
        workspace_requested = has_any(_WORKSPACE_IMPORT_MARKERS)
        resolved_ids = resolved_target_ids()
        if zotero_requested and (import_requested or save_or_add_requested):
            ambiguous = self._has_ambiguous_reference(normalized) and not resolved_ids
            return self._result(
                "sync_to_zotero",
                "paper" if resolved_ids else "papers",
                0.88 if not ambiguous else 0.58,
                markers,
                "Question asks to sync candidate papers into Zotero rather than run research QA.",
                resolved_paper_ids=resolved_ids,
                reference_type=reference_type_for(resolved_ids=resolved_ids),
                needs_clarification=ambiguous,
                clarification_question=(
                    "你是想把哪篇论文同步到 Zotero？请用序号、标题，或先选中论文后再试。"
                    if ambiguous
                    else None
                ),
            )
        if import_requested and (workspace_requested or has_task or bool(resolved_ids)):
            ambiguous = self._has_ambiguous_reference(normalized) and not resolved_ids
            return self._result(
                "paper_import",
                "paper" if resolved_ids else "papers",
                0.84 if not ambiguous else 0.56,
                markers,
                "Question asks to import candidate papers into the local workspace for later grounded use.",
                resolved_paper_ids=resolved_ids,
                reference_type=reference_type_for(resolved_ids=resolved_ids),
                needs_clarification=ambiguous,
                clarification_question=(
                    "你是想导入哪篇论文到当前工作区？请用序号、标题，或先选中论文后再试。"
                    if ambiguous
                    else None
                ),
            )
        if has_any(_SEARCH_MARKERS):
            return self._result(
                "literature_search", "collection", 0.78, markers,
                "Question appears to need literature discovery.",
                extracted_topic=cleaned_message,
                source_constraints=source_constraints,
            )
        if resolved_ordinal_ids or has_any(_SINGLE_PAPER_MARKERS):
            ambiguous = self._has_ambiguous_reference(normalized) and not (active_paper_ids or selected_paper_ids)
            return self._result(
                "single_paper_qa",
                "paper",
                0.78 if not ambiguous else 0.52,
                markers,
                "Question appears to target one paper.",
                resolved_paper_ids=resolved_ids,
                reference_type="ordinal" if resolved_ordinal_ids else "deictic" if self._has_ambiguous_reference(normalized) else "none",
                needs_clarification=ambiguous,
                clarification_question=(
                    "你提到“这篇/该论文”，但我还不能确定具体是哪一篇。请用序号、标题或先选择论文后再问。"
                    if ambiguous
                    else None
                ),
            )
        if not has_task and (has_any(_GENERAL_QA_MARKERS) or candidate_paper_count == 0):
            return self._result("general_answer", "none", 0.72, markers, "Question looks like general Q&A rather than literature work.")
        ambiguous = self._has_ambiguous_reference(normalized) and candidate_paper_count > 1 and not active_paper_ids
        return self._result(
            "general_follow_up",
            "none",
            0.55,
            markers,
            "No strong task-specific marker was detected.",
            needs_clarification=ambiguous,
            clarification_question=(
                "这个追问可能依赖上一轮对象，但当前焦点不明确。你想问哪篇论文或哪张图？"
                if ambiguous
                else None
            ),
        )

    def _has_ambiguous_reference(self, normalized: str) -> bool:
        return any(marker in normalized for marker in _AMBIGUOUS_REFERENCES)

    def _result(
        self,
        intent: ResearchUserIntentName,
        target_kind: Literal["none", "paper", "papers", "figure", "document", "collection"],
        confidence: float,
        markers: list[str],
        rationale: str,
        *,
        resolved_paper_ids: list[str] | None = None,
        reference_type: Literal["none", "deictic", "ordinal", "alias", "title", "mixed"] = "none",
        needs_clarification: bool = False,
        clarification_question: str | None = None,
        extracted_topic: str = "",
        source_constraints: list[str] | None = None,
    ) -> ResearchUserIntentResult:
        return ResearchUserIntentResult(
            intent=intent,
            target_kind=target_kind,
            confidence=confidence,
            resolved_paper_ids=list(dict.fromkeys(resolved_paper_ids or [])),
            reference_type=reference_type,
            markers=list(dict.fromkeys(markers))[:8],
            rationale=rationale,
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
            extracted_topic=extracted_topic,
            source_constraints=list(source_constraints or []),
        )

    def _resolve_ordinal_references(
        self,
        normalized: str,
        *,
        candidate_papers: list[dict[str, Any]],
    ) -> list[str]:
        resolved: list[str] = []
        for pattern in _ORDINAL_PATTERNS:
            for match in pattern.finditer(normalized):
                index = self._parse_ordinal_token(match.group(1))
                if index is None:
                    continue
                if 1 <= index <= len(candidate_papers):
                    paper_id = str(candidate_papers[index - 1].get("paper_id") or "").strip()
                    if paper_id:
                        resolved.append(paper_id)
        return list(dict.fromkeys(resolved))

    def _parse_ordinal_token(self, raw_value: str | None) -> int | None:
        token = str(raw_value or "").strip()
        if not token:
            return None
        try:
            return int(token)
        except (TypeError, ValueError):
            pass
        if token == "十":
            return 10
        if token.startswith("十") and len(token) == 2:
            ones = _CHINESE_ORDINAL_MAP.get(token[1])
            return 10 + ones if ones is not None else None
        if token.endswith("十") and len(token) == 2:
            tens = _CHINESE_ORDINAL_MAP.get(token[0])
            return tens * 10 if tens is not None else None
        if len(token) == 3 and token[1] == "十":
            tens = _CHINESE_ORDINAL_MAP.get(token[0])
            ones = _CHINESE_ORDINAL_MAP.get(token[2])
            if tens is not None and ones is not None:
                return tens * 10 + ones
        return _CHINESE_ORDINAL_MAP.get(token)
