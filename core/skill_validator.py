"""Security validation for skills — prompt injection detection and path safety.

Checks applied to every skill before it enters the active pool:
1. Prompt injection patterns in body text
2. Path traversal in reference paths
3. Size limits
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from core.skill_registry import Skill, SkillMeta

logger = logging.getLogger(__name__)

# Max total bytes for a single skill directory
_MAX_SKILL_DIR_SIZE = 1_048_576  # 1 MiB

# Suspicious patterns that may indicate prompt injection
_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+(all\s+)?previous\s+instructions", "ignore-previous"),
    (r"you\s+are\s+now", "role-hijack"),
    (r"disregard\s+(your|all)", "disregard"),
    (r"forget\s+your\s+instructions", "forget-instructions"),
    (r"new\s+instructions\s*:", "new-instructions"),
    (r"system\s+prompt\s*:", "system-prompt"),
    (r"<\s*system\s*>", "xml-system-tag"),
    (r"<\s*/?\s*tool_call\s*>", "xml-tool-tag"),
    (r"\]\]>", "cdata-escape"),
    (r"<\|im_start\|>", "chat-template-inject"),
]

_COMPILED_PATTERNS = [(re.compile(pat, re.IGNORECASE), label) for pat, label in _INJECTION_PATTERNS]


class SkillValidationIssue:
    """A single validation issue found in a skill."""

    __slots__ = ("severity", "code", "message")

    def __init__(self, severity: str, code: str, message: str) -> None:
        self.severity = severity   # "error" | "warning"
        self.code = code
        self.message = message

    def __repr__(self) -> str:
        return f"SkillValidationIssue({self.severity}: {self.code} — {self.message})"


class SkillValidationResult:
    """Result of validating a skill."""

    __slots__ = ("skill_name", "issues")

    def __init__(self, skill_name: str, issues: list[SkillValidationIssue] | None = None) -> None:
        self.skill_name = skill_name
        self.issues = issues or []

    @property
    def passed(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == "warning" for i in self.issues)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"SkillValidationResult({self.skill_name}: {status}, {len(self.issues)} issue(s))"


class SkillValidator:
    """Validate skills for security and correctness."""

    def validate(self, skill: Skill) -> SkillValidationResult:
        """Run all checks on a skill."""
        issues: list[SkillValidationIssue] = []
        issues.extend(self._check_injection(skill))
        issues.extend(self._check_meta(skill.meta))
        issues.extend(self._check_size(skill))
        return SkillValidationResult(skill_name=skill.meta.name, issues=issues)

    def validate_directory(self, skill_dir: Path) -> SkillValidationResult:
        """Validate a skill directory on disk (including size check)."""
        name = skill_dir.name
        issues: list[SkillValidationIssue] = []
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.is_file():
            issues.append(SkillValidationIssue("error", "missing_skill_md", "SKILL.md not found"))
            return SkillValidationResult(name, issues)
        total_size = sum(f.stat().st_size for f in skill_dir.rglob("*") if f.is_file())
        if total_size > _MAX_SKILL_DIR_SIZE:
            issues.append(SkillValidationIssue(
                "error", "dir_too_large",
                f"Skill directory is {total_size} bytes (max {_MAX_SKILL_DIR_SIZE})",
            ))
        # Check for path traversal in file names
        for f in skill_dir.rglob("*"):
            try:
                rel = f.relative_to(skill_dir)
            except ValueError:
                issues.append(SkillValidationIssue("error", "path_escape", f"File outside skill dir: {f}"))
                continue
            if ".." in rel.parts:
                issues.append(SkillValidationIssue("error", "path_traversal", f"Path traversal: {rel}"))

        return SkillValidationResult(name, issues)

    # -- Internal checks -----------------------------------------------------

    def _check_injection(self, skill: Skill) -> list[SkillValidationIssue]:
        issues: list[SkillValidationIssue] = []
        text = skill.body
        for pattern, label in _COMPILED_PATTERNS:
            if pattern.search(text):
                issues.append(SkillValidationIssue(
                    "warning", f"injection_{label}",
                    f"Suspicious pattern detected: {label}",
                ))
        return issues

    def _check_meta(self, meta: SkillMeta) -> list[SkillValidationIssue]:
        issues: list[SkillValidationIssue] = []
        if not meta.name or len(meta.name) > 64:
            issues.append(SkillValidationIssue(
                "error", "invalid_name",
                f"Skill name must be 1-64 chars, got {len(meta.name or '')}",
            ))
        if len(meta.description) > 1024:
            issues.append(SkillValidationIssue(
                "warning", "description_too_long",
                f"Description is {len(meta.description)} chars (recommended max 1024)",
            ))
        if meta.trust_level not in ("builtin", "trusted", "community"):
            issues.append(SkillValidationIssue(
                "warning", "unknown_trust_level",
                f"Unknown trust_level: {meta.trust_level}",
            ))
        return issues

    def _check_size(self, skill: Skill) -> list[SkillValidationIssue]:
        issues: list[SkillValidationIssue] = []
        if len(skill.body) > 30_000:
            issues.append(SkillValidationIssue(
                "warning", "body_too_large",
                f"Skill body is {len(skill.body)} chars (recommended max 30000)",
            ))
        return issues
