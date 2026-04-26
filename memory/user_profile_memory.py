from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re

from domain.schemas.research_memory import (
    InterestTopic,
    LongTermMemoryQuery,
    LongTermMemoryRecord,
    UserResearchProfile,
)
from memory.long_term_memory import LongTermMemory


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_topic(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value.lower())).strip()


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


class UserProfileMemory:
    def __init__(self, long_term_memory: LongTermMemory) -> None:
        self.long_term_memory = long_term_memory

    def load_profile(self, *, user_id: str = "local-user") -> UserResearchProfile:
        result = self.long_term_memory.search(
            LongTermMemoryQuery(
                query=user_id,
                topic=user_id,
                keywords=[user_id, "user_profile"],
                top_k=5,
            )
        )
        for record in result.records:
            if record.memory_type != "user_profile":
                continue
            profile_payload = record.metadata.get("profile")
            if isinstance(profile_payload, dict):
                profile = UserResearchProfile.model_validate(profile_payload)
                return self._normalize_profile(profile)
        return UserResearchProfile(user_id=user_id)

    def save_profile(self, profile: UserResearchProfile) -> UserResearchProfile:
        normalized = self._normalize_profile(profile)
        record = LongTermMemoryRecord(
            memory_id=f"user_profile:{normalized.user_id}",
            memory_type="user_profile",
            topic=normalized.user_id,
            content=" | ".join(
                [
                    *(normalized.research_interests[:5]),
                    *(normalized.preferred_keywords[:5]),
                    *(normalized.notes[:3]),
                ]
            )
            or normalized.display_name
            or normalized.user_id,
            keywords=[
                *normalized.research_interests[:8],
                *normalized.preferred_keywords[:8],
                "user_profile",
            ],
            metadata={"profile": normalized.model_dump(mode="json")},
        )
        self.long_term_memory.upsert(record)
        return normalized

    def clear_profile(self, *, user_id: str = "local-user") -> UserResearchProfile:
        return self.save_profile(UserResearchProfile(user_id=user_id))

    def remove_topics(
        self,
        *,
        user_id: str = "local-user",
        topics: list[str],
    ) -> UserResearchProfile:
        targets = {_normalize_topic(item) for item in topics if _normalize_topic(item)}
        profile = self.load_profile(user_id=user_id)
        if not targets:
            return self.save_profile(profile)
        profile.interest_topics = [
            item
            for item in profile.interest_topics
            if item.normalized_topic not in targets
        ]
        profile.research_interests = [
            item
            for item in profile.research_interests
            if _normalize_topic(item) not in targets
        ]
        profile.recommendation_history = [
            {
                **entry,
                "topics_used": [
                    topic_name
                    for topic_name in list(entry.get("topics_used") or [])
                    if _normalize_topic(topic_name) not in targets
                ],
            }
            for entry in profile.recommendation_history
        ]
        profile.recommendation_history = [
            entry
            for entry in profile.recommendation_history
            if entry.get("topics_used") or not entry.get("recommendation_ids")
        ]
        if _normalize_topic(profile.last_active_topic or "") in targets:
            profile.last_active_topic = profile.interest_topics[0].topic_name if profile.interest_topics else None
        profile.preferred_sources = [
            source
            for item in profile.interest_topics[:6]
            for source in item.preferred_sources
        ]
        profile.preferred_keywords = [
            keyword
            for item in profile.interest_topics[:6]
            for keyword in item.preferred_keywords
        ]
        return self.save_profile(profile)

    def update_profile(
        self,
        *,
        user_id: str = "local-user",
        topic: str | None = None,
        sources: list[str] | None = None,
        keywords: list[str] | None = None,
        reasoning_style: str | None = None,
        answer_language: str | None = None,
        note: str | None = None,
        topics: list[str] | None = None,
        preferred_recency_days: int | None = None,
    ) -> UserResearchProfile:
        profile = self.load_profile(user_id=user_id)
        topic_candidates = [
            *([topic] if topic and topic.strip() else []),
            *[item for item in (topics or []) if str(item).strip()],
        ]
        if topic_candidates:
            profile = self.observe_query(
                user_id=user_id,
                topics=topic_candidates,
                sources=sources,
                keywords=keywords,
                preferred_recency_days=preferred_recency_days,
                signal_strength=1.1 if topic and topic.strip() else 0.9,
                profile=profile,
            )
        else:
            if sources:
                profile.preferred_sources = _dedupe_strings(
                    [*profile.preferred_sources, *sources],
                    limit=10,
                )
            if keywords:
                profile.preferred_keywords = _dedupe_strings(
                    [*profile.preferred_keywords, *keywords],
                    limit=20,
                )
        if topic and topic.strip():
            profile.last_active_topic = topic.strip()
        if reasoning_style and reasoning_style.strip():
            profile.preferred_reasoning_style = reasoning_style.strip()
        if answer_language and answer_language.strip():
            profile.preferred_answer_language = answer_language.strip()
        if note and note.strip():
            profile.notes = _dedupe_strings([note.strip(), *profile.notes], limit=10)
        return self.save_profile(profile)

    def observe_query(
        self,
        *,
        user_id: str = "local-user",
        topics: list[str],
        sources: list[str] | None = None,
        keywords: list[str] | None = None,
        preferred_recency_days: int | None = None,
        signal_strength: float = 1.0,
        profile: UserResearchProfile | None = None,
        metadata: dict | None = None,
    ) -> UserResearchProfile:
        profile = self.load_profile(user_id=user_id) if profile is None else profile.model_copy(deep=True)
        if not topics and not keywords and not sources:
            return self.save_profile(profile)
        now = utc_now()
        topic_by_key = {
            item.normalized_topic: item.model_copy(deep=True)
            for item in profile.interest_topics
            if item.normalized_topic
        }
        recent_cutoff = now - timedelta(days=30)
        for item in topic_by_key.values():
            if item.last_seen_at < recent_cutoff:
                item.recent_mention_count = 0
            item.weight = round(item.weight * 0.985, 4)
        normalized_keywords = _dedupe_strings(list(keywords or []), limit=12)
        normalized_sources = _dedupe_strings(list(sources or []), limit=6)
        for raw_topic in topics:
            topic_name = str(raw_topic or "").strip()
            normalized_topic = _normalize_topic(topic_name)
            if not topic_name or not normalized_topic:
                continue
            topic_entry = topic_by_key.get(normalized_topic)
            if topic_entry is None:
                topic_entry = InterestTopic(
                    topic_name=topic_name,
                    normalized_topic=normalized_topic,
                    weight=0.0,
                    confidence=min(1.0, max(0.35, 0.45 + 0.15 * min(len(normalized_keywords), 3))),
                    mention_count=0,
                    recent_mention_count=0,
                    first_seen_at=now,
                    last_seen_at=now,
                )
            topic_entry.topic_name = topic_name
            topic_entry.last_seen_at = now
            topic_entry.mention_count += 1
            topic_entry.recent_mention_count += 1
            topic_entry.weight = round(
                topic_entry.weight + max(0.25, signal_strength) + 0.08 * len(normalized_keywords),
                4,
            )
            topic_entry.preferred_sources = _dedupe_strings(
                [*topic_entry.preferred_sources, *normalized_sources],
                limit=6,
            )
            topic_entry.preferred_keywords = _dedupe_strings(
                [*topic_entry.preferred_keywords, *normalized_keywords],
                limit=12,
            )
            if preferred_recency_days is not None:
                topic_entry.preferred_recency_days = preferred_recency_days
            if metadata:
                topic_entry.metadata = {**topic_entry.metadata, **metadata}
            topic_by_key[normalized_topic] = topic_entry
            profile.last_active_topic = topic_name

        profile.interest_topics = sorted(
            topic_by_key.values(),
            key=lambda item: (
                float(item.weight),
                int(item.recent_mention_count),
                item.last_seen_at,
            ),
            reverse=True,
        )[:12]
        profile.preferred_sources = _dedupe_strings(
            [
                *(sources or []),
                *profile.preferred_sources,
                *[
                    source
                    for item in profile.interest_topics[:6]
                    for source in item.preferred_sources
                ],
            ],
            limit=10,
        )
        profile.preferred_keywords = _dedupe_strings(
            [
                *(keywords or []),
                *profile.preferred_keywords,
                *[
                    keyword
                    for item in profile.interest_topics[:6]
                    for keyword in item.preferred_keywords
                ],
            ],
            limit=20,
        )
        return self.save_profile(profile)

    def record_recommendations(
        self,
        *,
        user_id: str = "local-user",
        topics_used: list[str],
        recommendation_ids: list[str],
        query: str,
    ) -> UserResearchProfile:
        profile = self.load_profile(user_id=user_id)
        history_entry = {
            "query": query,
            "topics_used": _dedupe_strings(list(topics_used), limit=6),
            "recommendation_ids": _dedupe_strings(list(recommendation_ids), limit=20),
            "created_at": utc_now().isoformat(),
        }
        profile.recommendation_history = [history_entry, *profile.recommendation_history][:20]
        return self.save_profile(profile)

    def _normalize_profile(self, profile: UserResearchProfile) -> UserResearchProfile:
        normalized = profile.model_copy(deep=True)
        normalized.interest_topics = sorted(
            [
                item
                for item in normalized.interest_topics
                if item.topic_name.strip() and item.normalized_topic.strip()
            ],
            key=lambda item: (
                float(item.weight),
                int(item.recent_mention_count),
                item.last_seen_at,
            ),
            reverse=True,
        )[:12]
        normalized.research_interests = _dedupe_strings(
            [item.topic_name for item in normalized.interest_topics] + list(normalized.research_interests),
            limit=12,
        )
        normalized.preferred_sources = _dedupe_strings(
            [
                *normalized.preferred_sources,
                *[
                    source
                    for item in normalized.interest_topics[:6]
                    for source in item.preferred_sources
                ],
            ],
            limit=10,
        )
        normalized.preferred_keywords = _dedupe_strings(
            [
                *normalized.preferred_keywords,
                *[
                    keyword
                    for item in normalized.interest_topics[:6]
                    for keyword in item.preferred_keywords
                ],
            ],
            limit=20,
        )
        normalized.notes = _dedupe_strings(list(normalized.notes), limit=10)
        normalized.recommendation_history = list(normalized.recommendation_history[:20])
        normalized.updated_at = utc_now()
        return normalized
