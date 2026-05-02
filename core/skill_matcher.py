"""Skill matching — decides which skills to activate for a given user query.

Three-level matching pipeline:
1. **L1 – Rule-based** — trigger regex, tag overlap, description overlap (sync, fast)
2. **L2 – Embedding similarity** — cosine distance between query and skill description (async)
3. **L3 – Cross-encoder rerank** — local reranker scores query vs skill description (async)

When embedding_adapter or reranker are unavailable, the matcher
gracefully falls back to L1-only scoring.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from core.skill_registry import SkillMeta, SkillRegistry, Skill

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ACTIVE = 3
_MIN_SCORE_THRESHOLD = 0.35


class SkillMatchResult:
    """A matched skill with its relevance score."""

    __slots__ = ("meta", "score", "match_reason")

    def __init__(self, meta: SkillMeta, score: float, match_reason: str) -> None:
        self.meta = meta
        self.score = score
        self.match_reason = match_reason

    def __repr__(self) -> str:
        return f"SkillMatchResult(name={self.meta.name!r}, score={self.score:.2f}, reason={self.match_reason!r})"


class SkillMatcher:
    """Match user queries against the skill registry.

    Usage::

        matcher = SkillMatcher(registry)
        results = matcher.match("对比这三篇论文的方法差异")
        # → [SkillMatchResult(name='paper-comparison', score=1.0, ...)]

    Enhanced (async) usage with embedding + rerank::

        matcher = SkillMatcher(registry, embedding_adapter=emb, reranker=reranker)
        results = await matcher.amatch("帮我找几篇关于 diffusion 的论文")
    """

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        max_active: int = _DEFAULT_MAX_ACTIVE,
        embedding_adapter: Any | None = None,
        reranker: Any | None = None,
    ) -> None:
        self.registry = registry
        self.max_active = max_active
        self.embedding_adapter = embedding_adapter
        self.reranker = reranker
        # Cached skill description embeddings: {"skill_name": [float, ...]}
        self._desc_embedding_cache: dict[str, list[float]] = {}
        self._desc_cache_key: str = ""

    def match(
        self,
        query: str,
        *,
        available_tool_names: list[str] | None = None,
        max_results: int | None = None,
    ) -> list[SkillMatchResult]:
        """Synchronous L1-only matching (backward-compatible)."""
        limit = max_results or self.max_active
        skills = self.registry.list_skills(include_disabled=False)
        available_tools = set(available_tool_names or [])
        query_lower = query.lower()

        candidates: list[SkillMatchResult] = []
        for meta in skills:
            score, reason = self._l1_score(meta, query_lower, available_tools)
            if score >= _MIN_SCORE_THRESHOLD:
                candidates.append(SkillMatchResult(meta=meta, score=score, match_reason=reason))

        candidates.sort(key=lambda r: -r.score)
        return candidates[:limit]

    async def amatch(
        self,
        query: str,
        *,
        available_tool_names: list[str] | None = None,
        max_results: int | None = None,
    ) -> list[SkillMatchResult]:
        """Async three-level matching: L1 rules → L2 embedding → L3 rerank."""
        limit = max_results or self.max_active
        skills = self.registry.list_skills(include_disabled=False)
        available_tools = set(available_tool_names or [])
        query_lower = query.lower()

        # --- L1: Rule-based scoring (trigger + tag + description) ---
        l1_candidates: list[SkillMatchResult] = []
        for meta in skills:
            score, reason = self._l1_score(meta, query_lower, available_tools)
            l1_candidates.append(SkillMatchResult(meta=meta, score=score, match_reason=reason))

        l1_max = max((c.score for c in l1_candidates), default=0.0)
        if l1_max == 0.0:
            # No L1 signal at all — skip expensive L2/L3 (e.g. greetings, off-topic)
            return []

        # --- L2: Embedding similarity ---
        l2_applied = False
        if self.embedding_adapter is not None:
            try:
                l2_candidates = await self._l2_embedding_score(query, l1_candidates)
                l1_candidates = l2_candidates
                l2_applied = True
            except Exception:
                logger.warning("L2 embedding scoring failed, using L1 only", exc_info=True)

        # --- L3: Cross-encoder rerank (only top candidates) ---
        l3_applied = False
        above_threshold = [c for c in l1_candidates if c.score >= _MIN_SCORE_THRESHOLD]
        if self.reranker is not None and above_threshold:
            try:
                l3_candidates = await self._l3_rerank(query, above_threshold)
                above_threshold = l3_candidates
                l3_applied = True
            except Exception:
                logger.warning("L3 rerank scoring failed, using L2/L1 scores", exc_info=True)

        above_threshold.sort(key=lambda r: -r.score)
        results = above_threshold[:limit]

        if results:
            stages = ["L1"]
            if l2_applied:
                stages.append("L2-embed")
            if l3_applied:
                stages.append("L3-rerank")
            logger.info(
                "Skill matching [%s]: %s",
                "+".join(stages),
                ", ".join(f"{r.meta.name}({r.score:.2f})" for r in results),
            )
        return results

    # -- L1: Rule-based scoring -----------------------------------------------

    def _l1_score(
        self,
        meta: SkillMeta,
        query_lower: str,
        available_tools: set[str],
    ) -> tuple[float, str]:
        """Return (score, reason) using trigger/tag/description rules."""
        score = 0.0
        reason = ""

        # 1. Trigger regex matching (highest priority)
        for trigger in meta.triggers:
            try:
                if re.search(trigger, query_lower):
                    score = max(score, 1.0)
                    reason = f"trigger:{trigger}"
                    break
            except re.error:
                continue

        # 2. Tag / keyword matching
        if score < 1.0:
            tag_score = self._tag_score(meta, query_lower)
            if tag_score > score:
                score = tag_score
                reason = "tags"

        # 3. Description keyword matching
        if score < 0.5:
            desc_score = self._description_score(meta, query_lower)
            if desc_score > score:
                score = desc_score
                reason = "description"

        # Demote if required tools are missing
        if score > 0 and meta.requires_tools and available_tools:
            missing = set(meta.requires_tools) - available_tools
            if missing:
                score *= 0.5
                reason += f" (missing_tools:{','.join(missing)})"

        return score, reason

    @staticmethod
    def _tag_score(meta: SkillMeta, query_lower: str) -> float:
        if not meta.tags:
            return 0.0
        hits = sum(1 for tag in meta.tags if tag.lower() in query_lower)
        if hits == 0:
            return 0.0
        return min(0.3 + 0.2 * hits, 0.8)

    @staticmethod
    def _description_score(meta: SkillMeta, query_lower: str) -> float:
        if not meta.description:
            return 0.0
        desc_words = set(meta.description.lower().split())
        query_words = set(query_lower.split())
        overlap = desc_words & query_words
        # Filter out very short words
        meaningful = {w for w in overlap if len(w) > 1}
        if not meaningful:
            return 0.0
        return min(0.2 + 0.1 * len(meaningful), 0.6)

    # -- L2: Embedding similarity ---------------------------------------------

    async def _l2_embedding_score(
        self,
        query: str,
        candidates: list[SkillMatchResult],
    ) -> list[SkillMatchResult]:
        """Compute cosine similarity between query and skill descriptions."""
        # Build cache key from sorted skill names to detect changes
        cache_key = ",".join(sorted(c.meta.name for c in candidates))
        uncached = []
        if cache_key != self._desc_cache_key:
            # Skill set changed — rebuild full cache
            uncached = candidates
            self._desc_embedding_cache.clear()
            self._desc_cache_key = cache_key
        else:
            uncached = [c for c in candidates if c.meta.name not in self._desc_embedding_cache]

        # Embed uncached skill descriptions (skip if all cached)
        if uncached:
            texts_to_embed = [
                f"{c.meta.name}: {c.meta.description or ''}" for c in uncached
            ]
            vectors = await self.embedding_adapter.embed_texts(texts_to_embed)
            for c, vec in zip(uncached, vectors):
                self._desc_embedding_cache[c.meta.name] = vec.values

        # Embed only the query
        query_vectors = await self.embedding_adapter.embed_texts([query])
        query_vec = query_vectors[0].values

        for candidate in candidates:
            desc_vec = self._desc_embedding_cache[candidate.meta.name]
            similarity = self._cosine_similarity(query_vec, desc_vec)
            l1_score = candidate.score
            blended = l1_score * 0.4 + similarity * 0.6
            candidate.score = round(blended, 4)
            if similarity > 0.5:
                candidate.match_reason += f"+embed({similarity:.2f})"
        return candidates

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))

    # -- L3: Cross-encoder rerank ---------------------------------------------

    async def _l3_rerank(
        self,
        query: str,
        candidates: list[SkillMatchResult],
    ) -> list[SkillMatchResult]:
        """Use cross-encoder to compute precise query-skill relevance."""
        documents = [
            f"{c.meta.name}: {c.meta.description or ''}"
            for c in candidates
        ]
        rerank_scores = await self.reranker.score(query, documents)
        for candidate, rerank_score in zip(candidates, rerank_scores):
            # Normalize cross-encoder score to [0, 1] via sigmoid
            normalized = 1.0 / (1.0 + math.exp(-rerank_score))
            # Blend: previous score * 0.3 + rerank * 0.7 (reranker is most precise)
            prev_score = candidate.score
            blended = prev_score * 0.3 + normalized * 0.7
            candidate.score = round(blended, 4)
            candidate.match_reason += f"+rerank({normalized:.2f})"
        return candidates


def build_skill_context(skills: list[Skill]) -> str:
    """Build a context string from matched skills for injection into the agent.

    Returns a Markdown-formatted string with skill instructions.
    """
    if not skills:
        return ""
    parts = ["## Active Skills\n"]
    for skill in skills:
        header = f"### Skill: {skill.meta.name}"
        if skill.meta.description:
            header += f"\n> {skill.meta.description}"
        parts.append(f"{header}\n\n{skill.body}\n")
    return "\n".join(parts)
