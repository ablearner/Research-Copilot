from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PromptResolverError(RuntimeError):
    """Raised when prompt mapping cannot be resolved."""


class PromptResolver:
    def __init__(
        self,
        mapping_path: str | Path = "prompts/skill_prompt_mapping.yaml",
        default_prompt_set: dict[str, str] | None = None,
    ) -> None:
        self.mapping_path = Path(mapping_path)
        self._mapping_cache: dict[str, Any] | None = None
        self.default_prompt_set = default_prompt_set or {
            "answer_prompt_path": "prompts/document/answer_question_with_hybrid_rag.txt",
            "rewrite_prompt_path": "prompts/retrieval/rewrite_query.txt",
            "chart_prompt_path": "prompts/chart/parse_chart.txt",
            "graph_prompt_path": "prompts/graph/extract_triples.txt",
        }

    def load_mapping(self, force_reload: bool = False) -> dict[str, Any]:
        if self._mapping_cache is not None and not force_reload:
            return self._mapping_cache

        if not self.mapping_path.exists():
            logger.warning("Prompt mapping file not found, using built-in defaults")
            self._mapping_cache = {
                "version": 1,
                "defaults": dict(self.default_prompt_set),
                "skills": {},
                "aliases": {},
            }
            return self._mapping_cache

        content = self.mapping_path.read_text(encoding="utf-8")
        parsed = self._parse_mapping_content(content)
        if not isinstance(parsed, dict):
            raise PromptResolverError(f"Invalid prompt mapping content: {self.mapping_path}")
        parsed.setdefault("defaults", {})
        parsed.setdefault("skills", {})
        parsed.setdefault("aliases", {})
        self._mapping_cache = parsed
        return parsed

    def resolve_prompt_set(
        self,
        skill_name: str | None,
        explicit_prompt_set: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        mapping = self.load_mapping()
        defaults = mapping.get("defaults") if isinstance(mapping.get("defaults"), dict) else {}
        skills = mapping.get("skills") if isinstance(mapping.get("skills"), dict) else {}
        canonical_skill = self._canonical_skill_name(skill_name, mapping)
        skill_mapping = skills.get(canonical_skill, {}) if canonical_skill else {}
        if not isinstance(skill_mapping, dict):
            skill_mapping = {}

        resolved: dict[str, str] = {}
        resolved.update(self._coerce_prompt_set(defaults))
        resolved.update(self._coerce_prompt_set(skill_mapping))

        for key, value in self._coerce_prompt_set(explicit_prompt_set or {}).items():
            if key not in resolved:
                resolved[key] = value

        for key, value in self.default_prompt_set.items():
            resolved.setdefault(key, value)
        return resolved

    def resolve_prompt_path(
        self,
        prompt_key: str,
        skill_name: str | None = None,
        explicit_prompt_path: str | None = None,
    ) -> Path:
        candidates: list[str] = []
        if explicit_prompt_path:
            candidates.append(explicit_prompt_path)

        prompt_set = self.resolve_prompt_set(skill_name=skill_name)
        mapped = prompt_set.get(prompt_key)
        if mapped:
            candidates.append(mapped)

        fallback = self.default_prompt_set.get(prompt_key)
        if fallback:
            candidates.append(fallback)

        for candidate in candidates:
            path = Path(candidate)
            if path.exists():
                return path

        if candidates:
            return Path(candidates[-1])
        raise PromptResolverError(f"Prompt key not found: {prompt_key}")

    def _canonical_skill_name(self, skill_name: str | None, mapping: dict[str, Any]) -> str | None:
        if not skill_name:
            return None
        aliases = mapping.get("aliases") if isinstance(mapping.get("aliases"), dict) else {}
        return str(aliases.get(skill_name, skill_name))

    def _coerce_prompt_set(self, value: dict[str, Any] | Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        result: dict[str, str] = {}
        for key, prompt_path in value.items():
            if isinstance(prompt_path, str) and prompt_path.strip():
                result[str(key)] = prompt_path
        return result

    def _parse_mapping_content(self, content: str) -> dict[str, Any] | list[Any] | None:
        stripped = content.strip()
        if not stripped:
            return None
        try:
            import yaml  # type: ignore[import-not-found]

            return yaml.safe_load(stripped)
        except ModuleNotFoundError:
            return json.loads(stripped)
