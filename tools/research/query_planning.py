from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

from pydantic import BaseModel, Field

from domain.schemas.research import PaperSource, ResearchTopicPlan

logger = logging.getLogger(__name__)

_SYNONYM_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(
            r"(?:\bvln\b|vision(?:\s|-and-|\s+and\s+)language navigation|vision language navigation)",
            re.IGNORECASE,
        ),
        (
            "vision-and-language navigation",
            "vision language navigation",
            "VLN",
            "embodied navigation",
            "instruction-guided navigation",
        ),
    ),
    (
        re.compile(
            r"(?:大模型|大语言模型|大型语言模型|基础模型|基座模型|\bllm(?:s)?\b|large language model|foundation model|generative ai)",
            re.IGNORECASE,
        ),
        (
            "large language model",
            "LLM",
            "foundation model",
            "generative AI",
            "reasoning language model",
            "multimodal large language model",
        ),
    ),
    (
        re.compile(r"(?:\buav\b|\bdrone\b|无人机)", re.IGNORECASE),
        ("UAV", "drone", "unmanned aerial vehicle"),
    ),
    (
        re.compile(r"(?:path planning|路径规划)", re.IGNORECASE),
        ("path planning", "trajectory planning", "navigation"),
    ),
    (
        re.compile(r"(?:swarm|群体|协同)", re.IGNORECASE),
        ("swarm", "multi-agent", "cooperative"),
    ),
    (
        re.compile(r"(?:detection|检测)", re.IGNORECASE),
        ("detection", "object detection", "perception"),
    ),
)
_SOURCE_NAMES_FOR_STRIP = tuple(dict.fromkeys(
    [*get_args(PaperSource), *(s.replace("_", " ") for s in get_args(PaperSource) if "_" in s), "google scholar"]
))
_STRIP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"最近\s*\d+\s*(?:个)?(?:月|周|年)"),
    re.compile(r"(?:近|过去)\s*\d+\s*(?:个)?(?:月|周|年)"),
    re.compile(r"(?:最近|近期|当前|最新)"),
    re.compile(r"(?:方向)?有(?:哪些|什么)?值得关注的论文"),
    re.compile(r"有哪些论文"),
    re.compile(r"哪些论文"),
    re.compile(r"值得关注"),
    re.compile(r"帮我整理"),
    re.compile(r"帮我找"),
    re.compile(r"请(?:帮我)?"),
    re.compile(r"(?:这一|这组)?论文"),
    # Strip source-constraint phrases: "在arxiv上", "从ieee搜索", "on semantic scholar" etc.
    re.compile(
        r"(?:在|从|用|通过|去)\s*(?:" + "|".join(re.escape(s) for s in _SOURCE_NAMES_FOR_STRIP) + r")\s*(?:上|中|里|搜索|搜|找|检索|查)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:on|from|via|using|through)\s+(?:" + "|".join(re.escape(s) for s in _SOURCE_NAMES_FOR_STRIP) + r")\b",
        re.IGNORECASE,
    ),
    # Bare source names that survived the above (e.g. standalone "arxiv")
    re.compile(
        r"\b(?:" + "|".join(re.escape(s) for s in _SOURCE_NAMES_FOR_STRIP) + r")\b",
        re.IGNORECASE,
    ),
)
_PUNCTUATION_PATTERN = re.compile(r"[？?！!,，。.;；:：]")
_CJK_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}")
_ENGLISH_SOURCE_LIMITS = {
    "arxiv": 2,
    "semantic_scholar": 1,
    "ieee": 2,
    "openalex": 3,
    "zotero": 5,
}

_REWRITE_PROMPT = (
    "You rewrite user research requests into academic search queries for scholarly search providers.\n\n"
    "Rules:\n"
    "- Preserve the user's research intent, time window, and field.\n"
    "- If the user writes in Chinese, produce English search queries for English-first scholarly providers "
    "such as arXiv, Semantic Scholar, OpenAlex, and IEEE Xplore.\n"
    "- Always include the user's original-language query in local_queries so local libraries (e.g. Zotero) "
    "can match titles in the user's language as well as English.\n"
    "- Remove conversational framing such as 'help me find', 'recent', 'which papers', and question punctuation.\n"
    "- Prefer canonical academic terms over literal translation.\n"
    "- Produce concise queries (2-6 words each). Do not include explanations.\n"
    "- Generate 3-5 diverse queries covering different aspects of the topic.\n\n"
    "User topic: {topic}"
)


class _LLMRewriteResponse(BaseModel):
    simplified_topic: str = Field(description="Short topic in the user's language")
    detected_language: str = Field(description="zh or en or mixed")
    english_queries: list[str] = Field(description="Provider-ready English queries", min_length=1)
    local_queries: list[str] = Field(default_factory=list, description="Optional local-language queries")
    rationale: str = Field(default="", description="Brief reason for the rewrite")


@dataclass(slots=True)
class ResearchQueryRewriteResult:
    original_topic: str
    simplified_topic: str
    detected_language: str
    local_queries: list[str] = field(default_factory=list)
    english_queries: list[str] = field(default_factory=list)
    expanded_queries: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_queries(self) -> list[str]:
        return _dedupe_queries([*self.local_queries, *self.english_queries, *self.expanded_queries])


class ResearchQueryRewriter:
    """Skill used by LiteratureScoutAgent to build provider-ready search queries.
    
    When llm_adapter is provided, uses LLM for intelligent query rewriting.
    Falls back to heuristic rewriting when LLM is unavailable.
    """

    name = "ResearchQueryRewriter"

    def __init__(
        self,
        *,
        prompt_path: str | Path = "prompts/research/rewrite_literature_query.txt",
        llm_adapter: Any | None = None,
    ) -> None:
        self.prompt_path = Path(prompt_path)
        self.llm_adapter = llm_adapter

    def rewrite(self, topic: str) -> ResearchQueryRewriteResult:
        """Synchronous rewrite — uses heuristic logic."""
        return self._heuristic_rewrite(topic)

    async def rewrite_async(self, topic: str, *, supervisor_instruction: str | None = None) -> ResearchQueryRewriteResult:
        """Async rewrite — uses LLM if available, falls back to heuristic."""
        if self.llm_adapter is not None:
            try:
                return await self._llm_rewrite(topic, supervisor_instruction=supervisor_instruction)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM query rewrite failed, falling back to heuristic: %s", exc)
        return self._heuristic_rewrite(topic)

    async def _llm_rewrite(self, topic: str, *, supervisor_instruction: str | None = None) -> ResearchQueryRewriteResult:
        input_data: dict[str, Any] = {"topic": topic}
        if supervisor_instruction:
            input_data["supervisor_instruction"] = supervisor_instruction
        result = await self.llm_adapter.generate_structured(
            prompt=_REWRITE_PROMPT,
            input_data=input_data,
            response_model=_LLMRewriteResponse,
        )
        return ResearchQueryRewriteResult(
            original_topic=topic,
            simplified_topic=result.simplified_topic,
            detected_language=result.detected_language,
            local_queries=result.local_queries,
            english_queries=result.english_queries,
            expanded_queries=[],
            metadata={
                "rewriter": f"{self.name}+LLM",
                "query_language_policy": "english_for_scholarly_sources",
                "prompt_path": str(self.prompt_path),
                "rationale": result.rationale,
            },
        )

    def _heuristic_rewrite(self, topic: str) -> ResearchQueryRewriteResult:
        normalized = _compact_spaces(topic)
        simplified = _simplify_topic(normalized)
        focused_queries = _focused_domain_queries(simplified)
        expanded_queries = _expanded_domain_queries(simplified)
        local_queries = [simplified] if simplified else []
        detected_language = "zh" if _contains_cjk(normalized) else "en"
        return ResearchQueryRewriteResult(
            original_topic=normalized,
            simplified_topic=simplified,
            detected_language=detected_language,
            local_queries=local_queries,
            english_queries=focused_queries or _english_queries_from_text(simplified),
            expanded_queries=expanded_queries,
            metadata={
                "rewriter": self.name,
                "query_language_policy": "english_for_scholarly_sources",
                "prompt_path": str(self.prompt_path),
            },
        )

    def queries_for_source(self, *, source: str, queries: list[str]) -> list[str]:
        cleaned = _dedupe_queries(_compact_spaces(query) for query in queries if _compact_spaces(query))
        if not cleaned:
            return []
        if source == "zotero":
            # Zotero local API uses AND-matching on multi-word queries,
            # so both CJK and English multi-word queries need to be split
            # into short individual terms for broader recall.
            cjk_terms: list[str] = []
            for q in cleaned:
                if not _contains_cjk(q) or _contains_ascii_letter(q):
                    continue
                words = q.split()
                if len(words) <= 1:
                    cjk_terms.append(q)
                else:
                    cjk_terms.extend(words)
            cjk_terms = _dedupe_queries(cjk_terms)
            short_eng_terms: list[str] = []
            split_eng_terms: list[str] = []
            for q in cleaned:
                if not _contains_ascii_letter(q) or _contains_cjk(q):
                    continue
                words = q.split()
                if len(words) <= 3:
                    short_eng_terms.append(q)
                else:
                    split_eng_terms.extend(words)
            eng_terms = _dedupe_queries([*short_eng_terms, *split_eng_terms])
            # Interleave CJK and English so both get fair slots.
            limit = _ENGLISH_SOURCE_LIMITS.get(source, 3)
            mixed: list[str] = []
            ci, ei = 0, 0
            while len(mixed) < limit and (ci < len(cjk_terms) or ei < len(eng_terms)):
                if ei < len(eng_terms):
                    if eng_terms[ei] not in mixed:
                        mixed.append(eng_terms[ei])
                    ei += 1
                if len(mixed) >= limit:
                    break
                if ci < len(cjk_terms):
                    if cjk_terms[ci] not in mixed:
                        mixed.append(cjk_terms[ci])
                    ci += 1
            if not mixed:
                mixed = cleaned[:limit]
            return mixed

        english_queries = [query for query in cleaned if _contains_ascii_letter(query)]
        selected = english_queries or cleaned
        limit = _ENGLISH_SOURCE_LIMITS.get(source, 3)
        return selected[:limit]


class TopicPlanner:
    """Skill for creating query plans; used by LiteratureScoutAgent/PaperSearchService.
    
    When llm_adapter is provided, uses LLM for intelligent query planning.
    Falls back to heuristic planning when LLM is unavailable.
    """

    def __init__(
        self,
        *,
        query_rewrite_skill: ResearchQueryRewriter | None = None,
        llm_adapter: Any | None = None,
    ) -> None:
        self.llm_adapter = llm_adapter
        self.query_rewrite_skill = query_rewrite_skill or ResearchQueryRewriter(llm_adapter=llm_adapter)

    def plan(
        self,
        *,
        topic: str,
        days_back: int,
        max_papers: int,
        sources: list[PaperSource],
    ) -> ResearchTopicPlan:
        """Synchronous plan — uses heuristic logic."""
        return self._heuristic_plan(topic=topic, days_back=days_back, max_papers=max_papers, sources=sources)

    async def plan_async(
        self,
        *,
        topic: str,
        days_back: int,
        max_papers: int,
        sources: list[PaperSource],
        supervisor_instruction: str | None = None,
    ) -> ResearchTopicPlan:
        """Async plan — uses LLM rewrite if available."""
        normalized = " ".join(topic.strip().split())
        rewrite = await self.query_rewrite_skill.rewrite_async(normalized, supervisor_instruction=supervisor_instruction)
        heuristic_rewrite = self.query_rewrite_skill.rewrite(normalized)
        # Preserve high-yield heuristic seeds alongside LLM rewrites so acronym-heavy
        # topics like "VLN" do not lose their most effective provider queries.
        queries = _merge_rewrite_queries(
            rewrite,
            heuristic_rewrite,
            prefer_local_library="zotero" in sources,
        )
        core_terms = extract_core_terms(rewrite.simplified_topic) or extract_core_terms(
            heuristic_rewrite.simplified_topic
        )
        if core_terms:
            core_query = " ".join(core_terms[:6])
            if core_query not in queries:
                queries.append(core_query)
        return ResearchTopicPlan(
            topic=topic,
            normalized_topic=normalized,
            queries=_limit_plan_queries(queries, limit=6),
            days_back=days_back,
            max_papers=max_papers,
            sources=sources,
            metadata={
                "planner": "LLM" if "LLM" in rewrite.metadata.get("rewriter", "") else "heuristic",
                "query_rewrite_skill": rewrite.metadata["rewriter"],
                "query_language_policy": rewrite.metadata["query_language_policy"],
                "simplified_topic": rewrite.simplified_topic,
                "detected_language": rewrite.detected_language,
                "english_queries": " | ".join(rewrite.english_queries),
                "heuristic_english_queries": " | ".join(heuristic_rewrite.english_queries),
                "rewrite_prompt_path": str(rewrite.metadata["prompt_path"]),
            },
        )

    def _heuristic_plan(
        self,
        *,
        topic: str,
        days_back: int,
        max_papers: int,
        sources: list[PaperSource],
    ) -> ResearchTopicPlan:
        normalized = " ".join(topic.strip().split())
        rewrite = self.query_rewrite_skill.rewrite(normalized)
        queries = list(rewrite.all_queries)
        core_terms = extract_core_terms(rewrite.simplified_topic)
        if core_terms:
            core_query = " ".join(core_terms[:6])
            if core_query not in queries:
                queries.append(core_query)
        return ResearchTopicPlan(
            topic=topic,
            normalized_topic=normalized,
            queries=_limit_plan_queries(queries, limit=4),
            days_back=days_back,
            max_papers=max_papers,
            sources=sources,
            metadata={
                "planner": "heuristic",
                "query_rewrite_skill": rewrite.metadata["rewriter"],
                "query_language_policy": rewrite.metadata["query_language_policy"],
                "simplified_topic": rewrite.simplified_topic,
                "detected_language": rewrite.detected_language,
                "english_queries": " | ".join(rewrite.english_queries),
                "rewrite_prompt_path": str(rewrite.metadata["prompt_path"]),
            },
        )

    def queries_for_source(self, *, source: str, queries: list[str]) -> list[str]:
        return self.query_rewrite_skill.queries_for_source(source=source, queries=queries)


def _compact_spaces(value: str) -> str:
    return " ".join(value.strip().split())


def _simplify_topic(topic: str) -> str:
    simplified = _compact_spaces(topic)
    for pattern in _STRIP_PATTERNS:
        simplified = pattern.sub(" ", simplified)
    simplified = _PUNCTUATION_PATTERN.sub(" ", simplified)
    simplified = _compact_spaces(simplified)
    return simplified or _compact_spaces(topic)


def _contains_cjk(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in value)


def _contains_ascii_letter(value: str) -> bool:
    return re.search(r"[A-Za-z]", value) is not None


def _dedupe_queries(values) -> list[str]:
    deduped: list[str] = []
    for value in values:
        normalized = _compact_spaces(str(value))
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _merge_rewrite_queries(
    rewrite: ResearchQueryRewriteResult,
    heuristic_rewrite: ResearchQueryRewriteResult,
    *,
    prefer_local_library: bool,
) -> list[str]:
    if not prefer_local_library:
        return _dedupe_queries([*rewrite.all_queries, *heuristic_rewrite.all_queries])

    return _dedupe_queries(
        [
            *rewrite.local_queries[:1],
            *heuristic_rewrite.expanded_queries,
            *heuristic_rewrite.english_queries,
            *rewrite.english_queries,
            *rewrite.local_queries[1:],
            *heuristic_rewrite.local_queries,
        ]
    )


def _limit_plan_queries(queries: list[str], *, limit: int) -> list[str]:
    cleaned = _dedupe_queries(queries)
    if len(cleaned) <= limit:
        return cleaned

    selected: list[str] = []
    cjk_only = [query for query in cleaned if _contains_cjk(query) and not _contains_ascii_letter(query)]
    english_only = [query for query in cleaned if _contains_ascii_letter(query) and not _contains_cjk(query)]

    if cjk_only:
        selected.append(cjk_only[0])

    # Keep English provider/library terms from being pushed out by several
    # local-language variants in LLM rewrites.
    english_slots = max(0, limit - len(selected) - 1)
    for query in english_only[:english_slots]:
        if query not in selected:
            selected.append(query)

    for query in cleaned:
        if len(selected) >= limit:
            break
        if query not in selected:
            selected.append(query)

    return selected[:limit]


def _focused_domain_queries(topic: str) -> list[str]:
    spaced = _insert_cjk_ascii_spaces(topic)
    focused: list[str] = []
    if re.search(
        r"(?:\bvln\b|vision(?:\s|-and-|\s+and\s+)language navigation|vision language navigation)",
        spaced,
        re.IGNORECASE,
    ):
        focused.extend(
            [
                "vision-and-language navigation",
                "vision language navigation",
                "VLN",
                "embodied navigation",
            ]
        )
    if re.search(
        r"(?:大模型|大语言模型|大型语言模型|基础模型|基座模型|\bllm(?:s)?\b|large language model|foundation model|generative ai)",
        spaced,
        re.IGNORECASE,
    ):
        focused.extend(["large language model", "foundation model", "LLM"])
    if re.search(r"(?:\buav\b|\bdrone\b|无人机)", spaced, re.IGNORECASE) and re.search(
        r"(?:path planning|路径规划)",
        spaced,
        re.IGNORECASE,
    ):
        focused.extend(["UAV path planning", "drone trajectory planning", "unmanned aerial vehicle navigation"])
    return _dedupe_queries(focused)


_CJK_ASCII_BOUNDARY = re.compile(r"([\u4e00-\u9fff])([A-Za-z0-9])")
_ASCII_CJK_BOUNDARY = re.compile(r"([A-Za-z0-9])([\u4e00-\u9fff])")


def _insert_cjk_ascii_spaces(text: str) -> str:
    """Insert spaces at CJK/ASCII boundaries so \\b works for synonym matching."""
    text = _CJK_ASCII_BOUNDARY.sub(r"\1 \2", text)
    text = _ASCII_CJK_BOUNDARY.sub(r"\1 \2", text)
    return text


def _expanded_domain_queries(topic: str) -> list[str]:
    spaced_topic = _insert_cjk_ascii_spaces(topic)
    expansions: list[str] = []
    for pattern, synonyms in _SYNONYM_RULES:
        if pattern.search(spaced_topic):
            expansions.extend(synonyms)
    unique_expansions = _dedupe_queries(expansions)
    if not unique_expansions:
        return []
    expanded = [" ".join(unique_expansions[:8])]
    if _contains_cjk(topic):
        expanded.append(f"{topic} {' '.join(unique_expansions[:6])}")
    # Also include individual short synonyms so local-library sources
    # (e.g. Zotero) can pick them as standalone queries.
    for term in unique_expansions:
        if len(term.split()) <= 3 and term not in expanded:
            expanded.append(term)
    return expanded


def _english_queries_from_text(topic: str) -> list[str]:
    english_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]+", topic)
    if len(english_tokens) >= 2:
        return [" ".join(english_tokens[:6])]
    if len(english_tokens) == 1 and len(english_tokens[0]) >= 3:
        return [english_tokens[0]]
    return []


def extract_core_terms(topic: str) -> list[str]:
    terms: list[str] = []
    for token in _CJK_TOKEN_PATTERN.findall(topic):
        normalized = token.strip()
        if normalized and normalized not in terms:
            terms.append(normalized)
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", topic):
        normalized = token.strip()
        if normalized and normalized not in terms:
            terms.append(normalized)
    return terms


# Compatibility aliases for older imports or tests that still use the old names.
ResearchQueryRewriteAgent = ResearchQueryRewriter
TopicPlannerAgent = TopicPlanner
