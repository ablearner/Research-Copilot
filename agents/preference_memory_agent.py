from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from domain.schemas.research import PaperCandidate, PaperSource
from domain.schemas.research_functions import RecommendPapersFunctionOutput, RecommendedPaper
from domain.schemas.research_memory import InterestTopic, UserResearchProfile
from memory.memory_manager import MemoryManager
from services.research.paper_search_service import PaperSearchService
from tooling.research_runtime_schemas import NotificationItem


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


class RecommendationSourceDecision(BaseModel):
    sources: list[PaperSource] = Field(default_factory=list, min_length=1, max_length=3)
    rationale: str = ""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


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
    ) -> None:
        self.memory_manager = memory_manager
        self.paper_search_service = paper_search_service
        self.storage_root = Path(storage_root) if storage_root is not None else None

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
        profile = self.memory_manager.load_user_profile(user_id=user_id)
        if signal["topics"] or signal["keywords"] or signal["sources"]:
            profile = self.memory_manager.observe_user_query(
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
            profile = self.memory_manager.update_user_profile(
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
    ) -> RecommendPapersFunctionOutput:
        profile = self.memory_manager.load_user_profile(user_id=user_id)
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
        if not topics:
            return RecommendPapersFunctionOutput(
                recommendations=[],
                metadata={
                    "topics_used": [],
                    "days_back": effective_days_back,
                    "sources": list(sources or []),
                    "reason": "no_preference_topics",
                },
            )
        source_pool, source_selection_metadata = await self._resolve_recommendation_sources(
            question=question,
            profile=profile,
            topics=topics,
            requested_sources=list(sources or []),
        )
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
        topic_groups = self._build_topic_groups(all_ranked_candidates, limit_per_topic=2, limit_topics=3)
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
            },
        )
        self.memory_manager.record_user_recommendations(
            user_id=user_id,
            topics_used=topics_used,
            recommendation_ids=[item.paper_id for item in recommendations],
            query=question,
        )
        if include_notification and recommendations:
            self._enqueue_recommendation_notification(
                user_id=user_id,
                output=output,
            )
        return output

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
