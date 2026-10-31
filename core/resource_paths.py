"""Resolve bundled resources while preserving caller-supplied paths."""

from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_bundled_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    bundled = _PROJECT_ROOT / candidate
    return bundled if bundled.exists() else candidate
