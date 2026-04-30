"""Skill matching — decides which skills to activate for a given user query.

Two matching strategies:
1. **Trigger matching** — regex patterns defined in skill frontmatter
2. **Keyword matching** — tag/description overlap with the query

The matcher returns a ranked list of skills.  The supervisor can then
inject the top-N skill instructions into its context.
"""

from __future__ import annotations

import logging
import re
from core.skill_registry import SkillMeta, SkillRegistry, Skill

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ACTIVE = 3


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
    """

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        max_active: int = _DEFAULT_MAX_ACTIVE,
    ) -> None:
        self.registry = registry
        self.max_active = max_active

    def match(
        self,
        query: str,
        *,
        available_tool_names: list[str] | None = None,
        max_results: int | None = None,
    ) -> list[SkillMatchResult]:
        """Return skills matching *query*, ranked by relevance score.

        Parameters
        ----------
        query:
            The user's natural-language request.
        available_tool_names:
            Tool names currently available in the runtime.  Skills whose
            ``requires.tools`` are not all present will be demoted.
        max_results:
            Maximum number of results to return (default: ``max_active``).
        """
        limit = max_results or self.max_active
        skills = self.registry.list_skills(include_disabled=False)
        available_tools = set(available_tool_names or [])
        query_lower = query.lower()

        candidates: list[SkillMatchResult] = []
        for meta in skills:
            score, reason = self._score(meta, query_lower, available_tools)
            if score > 0:
                candidates.append(SkillMatchResult(meta=meta, score=score, match_reason=reason))

        candidates.sort(key=lambda r: -r.score)
        return candidates[:limit]

    # -- Scoring helpers -----------------------------------------------------

    def _score(
        self,
        meta: SkillMeta,
        query_lower: str,
        available_tools: set[str],
    ) -> tuple[float, str]:
        """Return (score, reason) for a skill against a query."""
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
