from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from core.utils import now_iso as _now_iso
from pydantic import BaseModel, Field

from domain.schemas.research import PaperCandidate, PaperSource
from domain.schemas.research_functions import RecommendPapersFunctionOutput, RecommendedPaper
from domain.schemas.research_memory import InterestTopic, UserResearchProfile
from memory.memory_manager import MemoryManager
from tools.research.paper_search import PaperSearchService
from tooling.research_runtime_schemas import NotificationItem

if TYPE_CHECKING:
    from runtime.research.agent_protocol.base import (
        ResearchAgentToolContext,
        ResearchToolResult,
    )


_EN_STOPWORDS = {
    "a",
    "an",
    "and",
    "about",
    "based",
    "best",
    "compare",
    "for",
    "find",
    "from",
    "give",
    "help",
    "in",
    "interesting",
    "latest",
    "me",
    "most",
    "new",
    "of",
    "on",
    "paper",
    "papers",
    "read",
    "recent",
    "related",
    "show",
    "suggest",
    "survey",
    "the",
    "to",
    "topic",
    "what",
    "which",
    "worth",
}
_CN_STOPWORDS = {
    "一个",
    "一下",
    "什么",
    "关于",
    "值得",
    "作者",
    "哪些",
    "哪个",
    "告诉",
    "工作",
    "帮我",
    "怎么",
    "推荐",
    "最近",
    "最新",
    "文章",
    "方向",
    "有没有",
    "相关",
    "研究",
    "给我",
    "论文",
    "阅读",
}
_SOURCE_MARKERS: dict[str, tuple[str, ...]] = {
    "arxiv": ("arxiv", "arxiv.org"),
    "openalex": ("openalex",),
    "semantic_scholar": ("semantic scholar", "semanticscholar"),
    "ieee": ("ieee", "ieee xplore", "ieeexplore"),
    "zotero": ("zotero",),
}
_ALLOWED_SOURCES: tuple[PaperSource, ...] = ("arxiv", "openalex", "semantic_scholar", "ieee", "zotero")
_RECENCY_PATTERNS: list[tuple[str, int]] = [
    (r"最近一周|近一周|过去一周|last week", 7),
    (r"最近一个月|近一个月|过去一个月|last month", 30),
    (r"最近两个月|近两个月|过去两个月", 60),
    (r"最近三个月|近三个月|过去三个月|last 3 months", 90),
    (r"最近半年|近半年|过去半年|last 6 months", 180),
    (r"最近一年|近一年|过去一年|last year", 365),
    (r"最近|最新|近期|recent|latest|new", 30),
]
_GENERIC_TOPIC_BLACKLIST = {
    "给我推荐最近",
    "最近值得看",
    "值得看",
    "最近",
    "最新",
    "未命名研究会话",
}


class RecommendationIntentResult(BaseModel):
    """Structured output from LLM intent understanding."""
    search_topics: list[str] = Field(default_factory=list, description="Precise search topics extracted from user intent, max 4")
    recency_days: int | None = Field(default=None, description="How many days back to search, null means use default")
    constraints: dict[str, Any] = Field(default_factory=dict, description="Extra constraints: must_include_keywords, exclude_keywords, preferred_venues")
    rationale: str = Field(default="", description="Brief explanation of the interpreted intent")
    is_generic: bool = Field(default=False, description="True if user wants a broad recommendation without specific topics")


class PersonalizedRankItem(BaseModel):
    """Single ranked paper from LLM personalized ranking."""
    paper_id: str = Field(description="ID of the paper")
    score: float = Field(description="Relevance score 0-1")
    reason: str = Field(description="Personalized explanation why this paper is recommended for this user")


class PersonalizedRankResult(BaseModel):
    """Structured output from LLM personalized ranking."""
    ranked_papers: list[PersonalizedRankItem] = Field(default_factory=list)
    ranking_rationale: str = Field(default="", description="Overall rationale for the ranking")


class RecommendationSourceDecision(BaseModel):
    sources: list[PaperSource] = Field(default_factory=list, min_length=1, max_length=3)
    rationale: str = ""


def _normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(value or "").lower())).strip()


def _dedupe_strings(values: list[str], *, limit: int) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if not normalized:
            continue
        marker = normalized.casefold()
        if marker in seen:
            continue
        deduped.append(normalized)
        seen.add(marker)
        if len(deduped) >= limit:
            break
    return deduped


class PreferenceMemoryAgent:
    """Peer specialist that learns user interests and recommends new papers from long-term memory."""

    name = "PreferenceMemoryAgent"

    def __init__(
        self,
        *,
        memory_manager: MemoryManager,
        paper_search_service: PaperSearchService,
        storage_root: str | Path | None = None,
        memory_gateway: Any | None = None,
        llm_adapter: Any | None = None,
    ) -> None:
        self._memory_backend = memory_manager
        self.memory_gateway = memory_gateway
        self.paper_search_service = paper_search_service
        self.storage_root = Path(storage_root) if storage_root is not None else None
        self.llm_adapter = llm_adapter

    # ------------------------------------------------------------------
    # New unified entry point (SpecialistAgent protocol)
    # ------------------------------------------------------------------

    async def run_action(
        self,
        context: ResearchAgentToolContext,
        decision: Any,
    ) -> ResearchToolResult:
        from runtime.research.agent_protocol.base import MemoryOp, ResearchStateDelta, ResearchToolResult
        from runtime.research.unified_action_adapters import resolve_active_message

        active_message = resolve_active_message(decision)
        payload = dict(active_message.payload or {}) if active_message is not None else {}
        question = str(payload.get("goal") or context.request.message or "").strip()
        top_k = max(
            1,
            min(
                10,
                int(
                    payload.get("top_k")
                    or context.request.recommendation_top_k
                    or 6
                ),
            ),
        )
        days_back = max(
            1,
            int(
                payload.get("days_back")
                or context.request.days_back
                or 30
            ),
        )
        raw_sources = payload.get("sources")
        sources = (
            [str(item).strip().lower() for item in raw_sources if str(item).strip()]
            if isinstance(raw_sources, list)
            else list(context.request.sources)
        )
        recommendation_output = await self.recommend_recent_papers(
            question=question,
            days_back=days_back,
            top_k=top_k,
            sources=sources,
            include_notification=True,
            persist=False,
            supervisor_instruction=context.supervisor_instruction,
        )
        recommendations = list(recommendation_output.recommendations)
        memory_ops = [
            MemoryOp(
                op_type="record_user_recommendations",
                params={
                    "user_id": "local-user",
                    "topics_used": list(recommendation_output.metadata.get("topics_used", [])),
                    "recommendation_ids": [item.paper_id for item in recommendations],
                    "query": question,
                },
            ),
        ] if recommendations else []
        delta = ResearchStateDelta(
            preference_recommendation_result=recommendation_output,
            memory_ops=memory_ops,
        )
        if not recommendations:
            return ResearchToolResult(
                status="skipped",
                observation="no personalized paper recommendations could be generated from long-term preferences",
                metadata={
                    "reason": "no_preference_recommendations",
                    **recommendation_output.model_dump(mode="json"),
                },
                state_delta=delta,
            )
        return ResearchToolResult(
            status="succeeded",
            observation=(
                f"generated {len(recommendations)} personalized recommendations from long-term preferences"
            ),
            metadata=recommendation_output.model_dump(mode="json"),
            state_delta=delta,
        )

    # ------------------------------------------------------------------
    # Legacy unified runtime entry point (will be removed in Step 6)
    # ------------------------------------------------------------------

    def observe_user_message(
        self,
        *,
        message: str,
        user_id: str = "local-user",
        sources: list[str] | None = None,
        answer_language: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UserResearchProfile:
        signal = self.extract_preference_signal(message)
        profile = self._load_user_profile(user_id=user_id)
        if signal["topics"] or signal["keywords"] or signal["sources"]:
            profile = self._observe_user_query(
                user_id=user_id,
                topics=list(signal["topics"]),
                sources=list(signal["sources"]),
                keywords=list(signal["keywords"]),
                preferred_recency_days=signal["preferred_recency_days"],
                signal_strength=float(signal["signal_strength"]),
                metadata={
                    "last_observed_query": message,
                    "session_sources": list(sources or []),
                    **dict(metadata or {}),
                },
            )
        if answer_language:
            profile = self._update_user_profile(
                user_id=user_id,
                answer_language=answer_language,
            )
        return profile

    def extract_preference_signal(
        self,
        message: str,
        *,
        sources: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized_message = _normalize_text(message)
        del sources
        extracted_sources = _dedupe_strings(self._extract_sources(message), limit=6)
        topics = self._extract_topics(message)
        keywords = self._extract_keywords(message, topics=topics)
        preferred_recency_days = self._extract_recency_days(message)
        signal_strength = 1.0
        if any(marker in normalized_message for marker in ("推荐", "recommend", "suggest", "worth", "值得看")):
            signal_strength += 0.2
        if any(marker in normalized_message for marker in ("最近", "最新", "recent", "latest")):
            signal_strength += 0.1
        if len(topics) >= 2:
            signal_strength += 0.1
        return {
            "topics": topics,
            "keywords": keywords,
            "sources": extracted_sources,
            "preferred_recency_days": preferred_recency_days,
            "signal_strength": signal_strength,
        }

    async def recommend_recent_papers(
        self,
        *,
        question: str,
        user_id: str = "local-user",
        days_back: int = 30,
        top_k: int = 6,
        sources: list[PaperSource] | None = None,
        include_notification: bool = True,
        persist: bool = True,
        supervisor_instruction: str | None = None,
    ) -> RecommendPapersFunctionOutput:
        profile = self._load_user_profile(user_id=user_id)

        # --- Phase 1: Intent understanding (LLM with heuristic fallback) ---
        intent = await self._understand_recommendation_intent(
            question=question,
            profile=profile,
            supervisor_instruction=supervisor_instruction,
        )
        intent_mode = "llm" if intent is not None else "heuristic"

        if intent is not None:
            # LLM-driven topic & recency extraction
            llm_topics = [
                InterestTopic(
                    topic_name=t,
                    normalized_topic=_normalize_text(t),
                    weight=0.9,
                    confidence=0.8,
                    mention_count=1,
                    recent_mention_count=1,
                )
                for t in intent.search_topics[:4]
                if t.strip()
            ]
            # If LLM says generic, supplement with profile top topics
            if intent.is_generic or not llm_topics:
                profile_topics = self._select_recommendation_topics(
                    question=question,
                    profile=profile,
                    explicit_topics=[],
                )
                llm_topics = self._dedupe_topics([*llm_topics, *profile_topics])[:4]
            topics = llm_topics
            effective_days_back = max(1, intent.recency_days or days_back)
            intent_rationale = intent.rationale
        else:
            # Heuristic fallback (original logic)
            explicit_topics = self._extract_topics(question)
            preferred_days_back = self._extract_recency_days(question)
            effective_days_back = max(
                1,
                preferred_days_back
                or min(
                    days_back,
                    min(
                        (
                            item.preferred_recency_days
                            for item in profile.interest_topics
                            if item.preferred_recency_days is not None
                        ),
                        default=days_back,
                    ),
                ),
            )
            topics = self._select_recommendation_topics(
                question=question,
                profile=profile,
                explicit_topics=explicit_topics,
            )
            intent_rationale = ""

        if not topics:
            return RecommendPapersFunctionOutput(
                recommendations=[],
                metadata={
                    "topics_used": [],
                    "days_back": effective_days_back,
                    "sources": list(sources or []),
                    "reason": "no_preference_topics",
                    "intent_mode": intent_mode,
                },
            )

        # --- Phase 2: Source resolution (unchanged) ---
        source_pool, source_selection_metadata = await self._resolve_recommendation_sources(
            question=question,
            profile=profile,
            topics=topics,
            requested_sources=list(sources or []),
        )

        # --- Phase 3: Paper search ---
        paper_matches: dict[str, dict[str, Any]] = {}
        per_topic_limit = max(top_k, 4)
        for topic in topics:
            bundle = await self.paper_search_service.search(
                topic=topic.topic_name,
                days_back=topic.preferred_recency_days or effective_days_back,
                max_papers=per_topic_limit,
                sources=source_pool,
            )
            for paper in bundle.papers:
                marker = self._paper_key(paper)
                if marker not in paper_matches:
                    paper_matches[marker] = {"paper": paper, "topics": [], "search_topics": []}
                paper_matches[marker]["topics"].append(topic)
                paper_matches[marker]["search_topics"].append(topic.topic_name)

        # --- Phase 4: Ranking (LLM with heuristic fallback) ---
        ranking_mode = "heuristic"
        llm_rank = await self._personalized_rank(
            question=question,
            candidates=list(paper_matches.values()),
            profile=profile,
            top_k=top_k,
            intent_rationale=intent_rationale,
            supervisor_instruction=supervisor_instruction,
        )

        if llm_rank is not None and llm_rank.ranked_papers:
            ranking_mode = "llm"
            # Build lookup from paper_matches by paper_id
            paper_by_id: dict[str, dict[str, Any]] = {}
            for payload in paper_matches.values():
                paper_by_id[payload["paper"].paper_id] = payload
            ranked_candidates = []
            for rank_item in llm_rank.ranked_papers[:top_k]:
                match = paper_by_id.get(rank_item.paper_id)
                if match is not None:
                    ranked_candidates.append({
                        "paper": match["paper"],
                        "score": rank_item.score,
                        "reason": rank_item.reason,
                        "matched_topics": [t.topic_name for t in match.get("topics", [])[:2]],
                        "primary_topic": match["search_topics"][0] if match.get("search_topics") else "其他",
                    })
            # If LLM returned fewer than requested, supplement with heuristic
            if len(ranked_candidates) < top_k:
                used_ids = {item["paper"].paper_id for item in ranked_candidates}
                heuristic_fill = sorted(
                    (
                        self._build_ranked_recommendation_candidate(
                            paper=payload["paper"],
                            matched_topics=payload["topics"],
                            profile=profile,
                            days_back=effective_days_back,
                        )
                        for payload in paper_matches.values()
                        if payload["paper"].paper_id not in used_ids
                    ),
                    key=lambda item: float(item["score"]),
                    reverse=True,
                )
                ranked_candidates.extend(heuristic_fill[: top_k - len(ranked_candidates)])
        else:
            # Full heuristic fallback
            all_ranked_candidates = sorted(
                (
                    self._build_ranked_recommendation_candidate(
                        paper=payload["paper"],
                        matched_topics=payload["topics"],
                        profile=profile,
                        days_back=effective_days_back,
                    )
                    for payload in paper_matches.values()
                ),
                key=lambda item: float(item["score"]),
                reverse=True,
            )
            ranked_candidates = all_ranked_candidates[:top_k]

        recommendations = [
            RecommendedPaper(
                paper_id=item["paper"].paper_id,
                title=item["paper"].title,
                reason=item["reason"],
                source=item["paper"].source,
                year=item["paper"].year,
                url=item["paper"].url or item["paper"].pdf_url,
            )
            for item in ranked_candidates
        ]
        topics_used = _dedupe_strings([topic.topic_name for topic in topics], limit=4)
        topic_groups = self._build_topic_groups(ranked_candidates, limit_per_topic=2, limit_topics=3)
        output = RecommendPapersFunctionOutput(
            recommendations=recommendations,
            metadata={
                "topics_used": topics_used,
                "days_back": effective_days_back,
                "sources": source_pool,
                "resolved_sources": source_pool,
                "source_selection": source_selection_metadata,
                "topic_groups": topic_groups,
                "profile_interest_count": len(profile.interest_topics),
                "generated_at": _now_iso(),
                "intent_mode": intent_mode,
                "ranking_mode": ranking_mode,
                "intent_rationale": intent_rationale,
            },
        )
        if persist:
            self._record_user_recommendations(
                user_id=user_id,
                topics_used=topics_used,
                recommendation_ids=[item.paper_id for item in recommendations],
                query=question,
            )
        if persist and include_notification and recommendations:
            self._enqueue_recommendation_notification(
                user_id=user_id,
                output=output,
            )
        return output

    def _load_user_profile(self, *, user_id: str = "local-user") -> UserResearchProfile:
        gateway = self.memory_gateway
        if gateway is not None and hasattr(gateway, "load_user_profile"):
            return gateway.load_user_profile(user_id=user_id)
        return self._memory_backend.load_user_profile(user_id=user_id)

    def _observe_user_query(
        self,
        *,
        user_id: str,
        topics: list[str],
        sources: list[str],
        keywords: list[str],
        preferred_recency_days: int | None,
        signal_strength: float,
        metadata: dict[str, Any],
    ) -> UserResearchProfile:
        gateway = self.memory_gateway
        if gateway is not None and hasattr(gateway, "observe_user_query"):
            return gateway.observe_user_query(
                user_id=user_id,
                topics=topics,
                sources=sources,
                keywords=keywords,
                preferred_recency_days=preferred_recency_days,
                signal_strength=signal_strength,
                metadata=metadata,
            )
        return self._memory_backend.observe_user_query(
            user_id=user_id,
            topics=topics,
            sources=sources,
            keywords=keywords,
            preferred_recency_days=preferred_recency_days,
            signal_strength=signal_strength,
            metadata=metadata,
        )

    def _update_user_profile(
        self,
        *,
        user_id: str,
        answer_language: str | None = None,
    ) -> UserResearchProfile:
        gateway = self.memory_gateway
        if gateway is not None and hasattr(gateway, "update_user_profile"):
            return gateway.update_user_profile(
                user_id=user_id,
                answer_language=answer_language,
            )
        return self._memory_backend.update_user_profile(
            user_id=user_id,
            answer_language=answer_language,
        )

    def _record_user_recommendations(
        self,
        *,
        user_id: str,
        topics_used: list[str],
        recommendation_ids: list[str],
        query: str,
    ) -> None:
        gateway = self.memory_gateway
        if gateway is not None and hasattr(gateway, "record_user_recommendations"):
            gateway.record_user_recommendations(
                user_id=user_id,
                topics_used=topics_used,
                recommendation_ids=recommendation_ids,
                query=query,
            )
            return
        self._memory_backend.record_user_recommendations(
            user_id=user_id,
            topics_used=topics_used,
            recommendation_ids=recommendation_ids,
            query=query,
        )

    def _extract_sources(self, message: str) -> list[str]:
        normalized = message.lower()
        matches: list[str] = []
        for source, markers in _SOURCE_MARKERS.items():
            if any(marker in normalized for marker in markers):
                matches.append(source)
        return matches

    def _extract_recency_days(self, message: str) -> int | None:
        normalized = message.lower()
        for pattern, days in _RECENCY_PATTERNS:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                return days
        explicit_day_match = re.search(r"(近|最近|过去|last)\s*(\d{1,4})\s*(天|day|days)", normalized, re.IGNORECASE)
        if explicit_day_match:
            return max(1, int(explicit_day_match.group(2)))
        explicit_month_match = re.search(r"(近|最近|过去|last)\s*(\d{1,2})\s*(个月|month|months)", normalized, re.IGNORECASE)
        if explicit_month_match:
            return max(1, int(explicit_month_match.group(2)) * 30)
        return None

    def _extract_topics(self, message: str) -> list[str]:
        candidates: list[str] = []
        patterns = [
            r"(?:关于|关注|研究|想找|想看|推荐|比较|看看)\s*([A-Za-z0-9\u4e00-\u9fff\-\+/ ]{2,50}?)(?:的论文|相关论文|方向|工作|研究)",
            r"(?:papers?|work|works|research|methods?)\s+(?:on|about|for)\s+([A-Za-z0-9\-\+/ ]{2,60})",
            r"([A-Za-z0-9\u4e00-\u9fff\-\+/ ]{2,50}?)(?:有哪些|有什么|值得看|值得读|相关论文|papers?)",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, message, flags=re.IGNORECASE):
                cleaned = self._clean_topic_candidate(match)
                if cleaned:
                    candidates.append(cleaned)
        for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,24}", message):
            if self._looks_like_topic_token(token):
                candidates.append(token)
        return _dedupe_strings(candidates, limit=6)

    def _clean_topic_candidate(self, value: str) -> str | None:
        cleaned = str(value or "").strip(" .,:;!?，。；：、")
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"^(关于|关注|研究|推荐|帮我找|给我推荐)", "", cleaned, flags=re.IGNORECASE)
        cleaned = self._strip_topic_prefixes(cleaned)
        cleaned = re.sub(r"\s+(的论文|论文|papers?|work|works|research)$", "", cleaned, flags=re.IGNORECASE)
        normalized = _normalize_text(cleaned)
        if not normalized:
            return None
        if normalized in _EN_STOPWORDS or normalized in _CN_STOPWORDS:
            return None
        if normalized in _GENERIC_TOPIC_BLACKLIST:
            return None
        if self._looks_like_request_wrapped_topic(cleaned):
            return None
        if self._normalize_source_name(cleaned) is not None:
            return None
        if len(normalized.replace(" ", "")) <= 2:
            return None
        return cleaned

    def _looks_like_request_wrapped_topic(self, value: str) -> bool:
        normalized = _normalize_text(value)
        if not normalized:
            return False
        session_markers = ("未命名研究会话", "研究会话")
        if any(marker in str(value) for marker in session_markers):
            return True
        request_markers = (
            "有没有",
            "什么",
            "哪些",
            "给我",
            "帮我",
            "值得看",
            "值得读",
            "推荐一下",
            "what",
            "which",
            "give me",
            "can you",
        )
        if any(marker in normalized for marker in request_markers):
            return True
        if any(marker in str(value) for marker in ("，", ",", "？", "?")) and len(normalized.replace(" ", "")) >= 8:
            return True
        return False

    def _looks_like_topic_token(self, token: str) -> bool:
        normalized = token.strip()
        if len(normalized) <= 2:
            return False
        lowered = normalized.lower()
        if lowered in _EN_STOPWORDS:
            return False
        if self._normalize_source_name(normalized) is not None:
            return False
        if normalized.isupper() and 2 <= len(normalized) <= 12:
            return True
        if any(char.isupper() for char in normalized[1:]):
            return True
        return "-" in normalized and len(normalized) >= 4

    def _extract_keywords(self, message: str, *, topics: list[str]) -> list[str]:
        candidates = list(topics)
        candidates.extend(re.findall(r"[\u4e00-\u9fff]{2,8}", message))
        candidates.extend(re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,24}", message))
        keywords: list[str] = []
        for candidate in candidates:
            cleaned = str(candidate or "").strip()
            normalized = _normalize_text(cleaned)
            if not normalized:
                continue
            if re.fullmatch(r"[\u4e00-\u9fff]{5,}", cleaned) and any(
                marker in cleaned for marker in ("推荐", "最近", "最新", "论文", "给我", "值得")
            ):
                continue
            if normalized in _EN_STOPWORDS or normalized in _CN_STOPWORDS:
                continue
            if len(normalized.replace(" ", "")) <= 2:
                continue
            keywords.append(cleaned)
        return _dedupe_strings(keywords, limit=12)

    def _select_recommendation_topics(
        self,
        *,
        question: str,
        profile: UserResearchProfile,
        explicit_topics: list[str],
    ) -> list[InterestTopic]:
        selected: list[InterestTopic] = []
        by_normalized = {
            item.normalized_topic: item
            for item in profile.interest_topics
        }
        for topic_name in explicit_topics:
            normalized = _normalize_text(topic_name)
            if normalized in by_normalized:
                selected.append(by_normalized[normalized])
            else:
                selected.append(
                    InterestTopic(
                        topic_name=topic_name,
                        normalized_topic=normalized,
                        weight=0.85,
                        confidence=0.7,
                        mention_count=1,
                        recent_mention_count=1,
                    )
                )
        generic_request = self._looks_like_generic_recommendation(question)
        if generic_request or not selected:
            top_topics = sorted(
                [
                    item
                    for item in profile.interest_topics
                    if self._is_valid_interest_topic(item)
                ],
                key=lambda item: (float(item.weight), int(item.recent_mention_count), item.last_seen_at),
                reverse=True,
            )[:3]
            selected.extend(top_topics)
        return self._dedupe_topics(selected)[:3]

    def _dedupe_topics(self, topics: list[InterestTopic]) -> list[InterestTopic]:
        deduped: list[InterestTopic] = []
        seen: set[str] = set()
        for topic in topics:
            marker = topic.normalized_topic
            if not marker or marker in seen:
                continue
            deduped.append(topic)
            seen.add(marker)
        return deduped

    def _is_valid_interest_topic(self, topic: InterestTopic) -> bool:
        normalized = topic.normalized_topic
        if not normalized:
            return False
        if normalized in _GENERIC_TOPIC_BLACKLIST:
            return False
        if len(normalized.replace(" ", "")) <= 2:
            return False
        stripped = self._strip_topic_prefixes(topic.topic_name).strip()
        if stripped and self._normalize_source_name(stripped) is not None:
            return False
        return True

    def _looks_like_generic_recommendation(self, message: str) -> bool:
        normalized = _normalize_text(message)
        recommend_markers = ("推荐", "suggest", "recommend", "worth", "reading list", "值得看")
        scope_markers = ("论文", "paper", "papers", "work", "works")
        recent_markers = ("最近", "最新", "近期", "recent", "latest", "new")
        preference_markers = ("按我的兴趣", "根据我的兴趣", "我的偏好", "长期记忆", "根据我的历史")
        return (
            any(marker in normalized for marker in recommend_markers)
            and any(marker in normalized for marker in scope_markers)
            and (
                any(marker in normalized for marker in recent_markers)
                or any(marker in normalized for marker in preference_markers)
            )
        )

    def _normalize_sources(
        self,
        sources: list[str],
        *,
        default: list[PaperSource] | None = None,
    ) -> list[PaperSource]:
        normalized = [
            str(source).strip().lower().replace(" ", "_")
            for source in sources
            if str(source).strip().lower().replace(" ", "_") in set(_ALLOWED_SOURCES)
        ]
        if normalized:
            return list(dict.fromkeys(normalized))  # type: ignore[return-value]
        return list(default or [])

    def _normalize_source_name(self, value: str | None) -> PaperSource | None:
        normalized = _normalize_text(value)
        if not normalized:
            return None
        for source, markers in _SOURCE_MARKERS.items():
            if normalized == _normalize_text(source):
                return source  # type: ignore[return-value]
            for marker in markers:
                if normalized == _normalize_text(marker):
                    return source  # type: ignore[return-value]
        return None

    def _strip_topic_prefixes(self, value: str) -> str:
        cleaned = str(value or "").strip()
        patterns = [
            r"^(最近|近|过去)\s*(一周|一个月|两个月|三个月|半年|一年|\d+\s*(天|周|个月|年))\s*",
            r"^(最近|最新|近期)\s*",
            r"^last\s+(week|month|year)\s+",
            r"^last\s+\d+\s*(days?|weeks?|months?|years?)\s+",
            r"^(优先|优先看|优先用|只看|只搜|只用)\s*",
        ]
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    async def _resolve_recommendation_sources(
        self,
        *,
        question: str,
        profile: UserResearchProfile,
        topics: list[InterestTopic],
        requested_sources: list[str],
    ) -> tuple[list[PaperSource], dict[str, Any]]:
        explicit_question_sources = self._normalize_sources(self._extract_sources(question))
        if explicit_question_sources:
            return explicit_question_sources, {
                "mode": "explicit_question",
                "rationale": "The user explicitly named the academic sources in the question.",
            }

        llm_decision = await self._select_sources_with_llm(
            question=question,
            profile=profile,
            topics=topics,
            requested_sources=requested_sources,
        )
        if llm_decision is not None:
            return llm_decision.sources, {
                "mode": "llm",
                "rationale": llm_decision.rationale,
            }

        topic_hint_sources = self._normalize_sources(
            [
                source
                for topic in topics
                for source in getattr(topic, "preferred_sources", []) or []
            ]
        )
        if topic_hint_sources:
            return topic_hint_sources, {
                "mode": "topic_hints",
                "rationale": "Used topic-level source hints learned from past explicit source mentions.",
            }

        generic_request = self._looks_like_generic_recommendation(question)
        if generic_request:
            broad_default = self._normalize_sources(["arxiv", "openalex", "semantic_scholar"])
            return broad_default, {
                "mode": "broad_default",
                "rationale": "Generic recommendation requests should search across broad academic sources when the user did not constrain the source.",
            }

        request_scoped_sources = self._normalize_sources(requested_sources)
        if request_scoped_sources:
            return request_scoped_sources, {
                "mode": "request_scope",
                "rationale": "Fell back to the current request scope because no explicit or learned source preference was available.",
            }

        return self._normalize_sources(["arxiv", "openalex"]), {
            "mode": "fallback",
            "rationale": "Used the default multi-source academic search scope.",
        }

    async def _select_sources_with_llm(
        self,
        *,
        question: str,
        profile: UserResearchProfile,
        topics: list[InterestTopic],
        requested_sources: list[str],
    ) -> RecommendationSourceDecision | None:
        llm_adapter = getattr(self.paper_search_service, "llm_adapter", None)
        if llm_adapter is None:
            return None
        try:
            decision = await llm_adapter.generate_structured(
                prompt=(
                    "Choose the most appropriate academic literature sources for this paper recommendation request. "
                    "Infer the source scope from the user's question, topic hints, and long-term interests. "
                    "Do not blindly copy the current session sources unless the request clearly depends on them."
                ),
                input_data={
                    "question": question,
                    "allowed_sources": list(_ALLOWED_SOURCES),
                    "requested_sources": self._normalize_sources(requested_sources),
                    "profile_preferred_sources": self._normalize_sources(list(profile.preferred_sources)),
                    "topic_hints": [
                        {
                            "topic": topic.topic_name,
                            "weight": float(topic.weight),
                            "preferred_sources": self._normalize_sources(list(topic.preferred_sources)),
                        }
                        for topic in topics[:4]
                    ],
                },
                response_model=RecommendationSourceDecision,
            )
        except Exception:
            return None
        normalized_sources = self._normalize_sources(list(decision.sources))
        if not normalized_sources:
            return None
        return RecommendationSourceDecision(sources=normalized_sources, rationale=decision.rationale)

    def _paper_key(self, paper: PaperCandidate) -> str:
        if paper.paper_id:
            return paper.paper_id
        if paper.doi:
            return f"doi:{paper.doi.casefold()}"
        if paper.url:
            return f"url:{paper.url}"
        return _normalize_text(paper.title)

    def _build_ranked_recommendation_candidate(
        self,
        *,
        paper: PaperCandidate,
        matched_topics: list[InterestTopic],
        profile: UserResearchProfile,
        days_back: int,
    ) -> dict[str, Any]:
        matched_topics = sorted(
            self._dedupe_topics(list(matched_topics)),
            key=lambda item: (float(item.weight), int(item.recent_mention_count), item.last_seen_at),
            reverse=True,
        )
        max_weight = max((float(item.weight) for item in profile.interest_topics), default=1.0)
        weighted_interest = sum(float(item.weight) for item in matched_topics) / max(1.0, max_weight)
        source_bonus = 0.2 if str(paper.source) in set(profile.preferred_sources) else 0.0
        recency_bonus = self._paper_recency_score(paper, days_back=days_back)
        relevance_bonus = float(paper.relevance_score or 0.0)
        score = round(weighted_interest + source_bonus + recency_bonus + relevance_bonus, 4)
        topic_names = [item.topic_name for item in matched_topics[:2]]
        topic_phrase = "、".join(topic_names) if topic_names else "你的长期兴趣"
        if recency_bonus >= 0.8:
            reason = f"你长期高频关注 {topic_phrase}，这篇属于近 {days_back} 天内值得优先跟进的新工作。"
        else:
            reason = f"你长期高频关注 {topic_phrase}，这篇和你的稳定兴趣主题匹配度较高。"
        return {
            "paper": paper,
            "score": score,
            "reason": reason,
            "matched_topics": topic_names,
            "primary_topic": topic_names[0] if topic_names else "其他",
        }

    def _paper_explanation(self, paper: PaperCandidate) -> str:
        summary = re.sub(r"\s+", " ", str(paper.summary or "").strip())
        if summary:
            return summary[:180] + ("..." if len(summary) > 180 else "")
        abstract = re.sub(r"\s+", " ", str(paper.abstract or "").strip())
        if abstract:
            return abstract[:180] + ("..." if len(abstract) > 180 else "")
        return "暂无摘要信息，建议打开原文重点看问题设定、方法设计和实验结论。"

    def _build_topic_groups(
        self,
        ranked_candidates: list[dict[str, Any]],
        *,
        limit_per_topic: int = 2,
        limit_topics: int = 3,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in ranked_candidates:
            topic_name = str(item.get("primary_topic") or "其他").strip() or "其他"
            grouped.setdefault(topic_name, []).append(
                {
                    "paper_id": item["paper"].paper_id,
                    "title": item["paper"].title,
                    "reason": item["reason"],
                    "source": item["paper"].source,
                    "year": item["paper"].year,
                    "url": item["paper"].url or item["paper"].pdf_url,
                    "explanation": self._paper_explanation(item["paper"]),
                    "matched_topics": list(item.get("matched_topics") or []),
                }
            )
        return [
            {"topic": topic_name, "papers": papers[:limit_per_topic]}
            for topic_name, papers in list(grouped.items())[:limit_topics]
        ]

    def _paper_recency_score(self, paper: PaperCandidate, *, days_back: int) -> float:
        published_at = str(paper.published_at or "").strip()
        if published_at:
            try:
                published = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                age_days = max(0.0, (datetime.now(UTC) - published.astimezone(UTC)).total_seconds() / 86400.0)
                if age_days <= days_back:
                    return 1.0 - min(0.7, age_days / max(days_back, 1) * 0.7)
            except ValueError:
                pass
        if paper.year is not None:
            current_year = datetime.now(UTC).year
            if paper.year >= current_year:
                return 0.75
            if paper.year == current_year - 1:
                return 0.45
        return 0.15

    def _enqueue_recommendation_notification(
        self,
        *,
        user_id: str,
        output: RecommendPapersFunctionOutput,
    ) -> None:
        if self.storage_root is None:
            return
        queue_path = self.storage_root / "notifications" / "queue.json"
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        items: list[NotificationItem] = []
        if queue_path.exists():
            try:
                payload = json.loads(queue_path.read_text(encoding="utf-8"))
                items = [NotificationItem.model_validate(item) for item in payload]
            except Exception:
                items = []
        topics_used = list(output.metadata.get("topics_used") or [])
        items.append(
            NotificationItem(
                notification_id=f"notify_{uuid4().hex}",
                message=(
                    f"已生成 {len(output.recommendations)} 篇个性化论文推荐"
                    f"{'：' + ' / '.join(topics_used[:3]) if topics_used else ''}"
                ),
                channel="queue",
                metadata={
                    "type": "preference_recommendations",
                    "user_id": user_id,
                    "recommendations": output.model_dump(mode="json"),
                },
            )
        )
        queue_path.write_text(
            json.dumps([item.model_dump(mode="json") for item in items[-100:]], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # LLM-driven intent understanding
    # ------------------------------------------------------------------

    _INTENT_PROMPT = (
        "You are a research interest analyst for a personalized paper recommendation system.\n"
        "Given the user's request, their long-term interest profile, and an optional supervisor instruction,\n"
        "extract the precise search intent.\n\n"
        "Rules:\n"
        "- Output search_topics: concise academic search queries (English preferred for better coverage), max 4.\n"
        "- If the user references past interests implicitly (e.g. '跟上次类似'), infer from the profile.\n"
        "- If the request is generic ('推荐最近的论文'), set is_generic=true and pick top topics from the profile.\n"
        "- If a supervisor_instruction is given, incorporate its constraints (e.g. venue preferences, time range).\n"
        "- Output recency_days if the user specifies a time range, otherwise null.\n"
        "- Output constraints.must_include_keywords for important filter terms.\n"
        "- Output constraints.preferred_venues if venues/conferences are mentioned or implied.\n"
        "- Follow the user's language in the rationale field."
    )

    async def _understand_recommendation_intent(
        self,
        *,
        question: str,
        profile: UserResearchProfile,
        supervisor_instruction: str | None = None,
    ) -> RecommendationIntentResult | None:
        if self.llm_adapter is None:
            return None
        profile_summary = [
            {
                "topic": item.topic_name,
                "weight": round(float(item.weight), 2),
                "mention_count": int(item.mention_count),
                "preferred_sources": list(item.preferred_sources)[:2],
            }
            for item in sorted(
                profile.interest_topics,
                key=lambda t: (float(t.weight), int(t.recent_mention_count)),
                reverse=True,
            )[:5]
        ]
        input_data: dict[str, Any] = {
            "question": question,
            "user_interest_profile": profile_summary,
            "profile_preferred_sources": list(profile.preferred_sources)[:3],
        }
        if supervisor_instruction:
            input_data["supervisor_instruction"] = supervisor_instruction
        try:
            result = await self.llm_adapter.generate_structured(
                prompt=self._INTENT_PROMPT,
                input_data=input_data,
                response_model=RecommendationIntentResult,
            )
            if result.search_topics or result.is_generic:
                return result
            return None
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "LLM recommendation intent understanding failed, falling back to heuristic",
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # LLM-driven personalized ranking
    # ------------------------------------------------------------------

    _RANK_PROMPT = (
        "You are a personalized paper ranking assistant.\n"
        "Given a user's research interest profile, their original question, and a list of candidate papers,\n"
        "rank the papers by personal relevance and generate a unique, specific recommendation reason for each.\n\n"
        "Rules:\n"
        "- Score each paper 0.0 to 1.0 based on relevance to the user's interests AND the current question.\n"
        "- The reason should be personalized: explain WHY this paper matters to THIS user, not a generic summary.\n"
        "- Reference the user's known interests when explaining relevance.\n"
        "- If a supervisor_instruction is given, factor it into scoring (e.g. prefer certain venues or recency).\n"
        "- Return only paper_ids that exist in the candidate list.\n"
        "- Follow the user's language in the reason field.\n"
        "- Rank at most top_k papers."
    )

    async def _personalized_rank(
        self,
        *,
        question: str,
        candidates: list[dict[str, Any]],
        profile: UserResearchProfile,
        top_k: int,
        intent_rationale: str = "",
        supervisor_instruction: str | None = None,
    ) -> PersonalizedRankResult | None:
        if self.llm_adapter is None or not candidates:
            return None
        profile_summary = [
            {
                "topic": item.topic_name,
                "weight": round(float(item.weight), 2),
            }
            for item in sorted(
                profile.interest_topics,
                key=lambda t: float(t.weight),
                reverse=True,
            )[:5]
        ]
        papers_for_llm = [
            {
                "paper_id": item["paper"].paper_id,
                "title": item["paper"].title,
                "year": item["paper"].year,
                "source": item["paper"].source,
                "abstract": (item["paper"].abstract or item["paper"].summary or "")[:300],
                "matched_topics": [
                    (t.topic_name if hasattr(t, "topic_name") else str(t))
                    for t in item.get("topics", [])[:2]
                ],
            }
            for item in candidates[:15]
        ]
        if not papers_for_llm:
            return None
        input_data: dict[str, Any] = {
            "question": question,
            "user_interest_profile": profile_summary,
            "candidate_papers": papers_for_llm,
            "top_k": top_k,
            "intent_rationale": intent_rationale,
        }
        if supervisor_instruction:
            input_data["supervisor_instruction"] = supervisor_instruction
        try:
            return await self.llm_adapter.generate_structured(
                prompt=self._RANK_PROMPT,
                input_data=input_data,
                response_model=PersonalizedRankResult,
            )
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "LLM personalized ranking failed, falling back to heuristic",
                exc_info=True,
            )
            return None
