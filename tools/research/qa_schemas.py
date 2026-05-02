from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ResearchQARouteDecision:
    route: str
    confidence: float
    rationale: str
    visual_anchor: dict[str, Any] | None = None
    recovery_count: int = 0
