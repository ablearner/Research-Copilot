from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_POLISH_PROMPT = (
    "你是一个学术写作润色助手。请对以下文献调研报告进行学术风格润色。\n\n"
    "要求：\n"
    "- 保持原有结构和引用标记（如 [P1][P2]）不变\n"
    "- 提升表述的学术性和精确性\n"
    "- 消除口语化表达\n"
    "- 保持简洁，不要增加无关内容\n"
    "- 不要改变事实内容\n"
    "- 写作风格：{tone}\n"
    "{journal_hint}\n\n"
    "原文：\n{text}"
)


class _LLMPolishResponse(BaseModel):
    polished_text: str = Field(description="润色后的文本")


class WritingPolisher:
    """Apply academic-style cleanup using LLM with regex-based fallback.
    
    When llm_adapter is provided, uses LLM for intelligent polishing.
    Falls back to simple regex replacements when LLM is unavailable.
    """

    def __init__(self, *, llm_adapter: Any | None = None) -> None:
        self.llm_adapter = llm_adapter

    def polish(
        self,
        *,
        text: str,
        tone: str = "academic",
        target_journal: str | None = None,
    ) -> str:
        """Synchronous polish — uses heuristic logic."""
        return self._heuristic_polish(text=text, tone=tone, target_journal=target_journal)

    async def polish_async(
        self,
        *,
        text: str,
        tone: str = "academic",
        target_journal: str | None = None,
    ) -> str:
        """Async polish — uses LLM if available, falls back to heuristic."""
        if self.llm_adapter is not None and text.strip():
            try:
                return await self._llm_polish(text=text, tone=tone, target_journal=target_journal)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM polish failed, falling back to heuristic: %s", exc)
        return self._heuristic_polish(text=text, tone=tone, target_journal=target_journal)

    async def _llm_polish(
        self,
        *,
        text: str,
        tone: str = "academic",
        target_journal: str | None = None,
    ) -> str:
        journal_hint = f"目标期刊风格：{target_journal}" if target_journal else ""
        # Limit text length to avoid token overflow
        truncated = text[:8000] if len(text) > 8000 else text
        result = await self.llm_adapter.generate_structured(
            prompt=_POLISH_PROMPT,
            input_data={"text": truncated, "tone": tone, "journal_hint": journal_hint},
            response_model=_LLMPolishResponse,
        )
        # If original text was truncated, append the unpolished remainder
        if len(text) > 8000:
            return result.polished_text + text[8000:]
        return result.polished_text

    def _heuristic_polish(
        self,
        *,
        text: str,
        tone: str = "academic",
        target_journal: str | None = None,
    ) -> str:
        polished = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
        replacements = {
            "当前结果显示": "当前证据表明",
            "可以看到": "可以据此观察到",
            "值得关注": "尤其值得关注",
        }
        for source, target in replacements.items():
            polished = polished.replace(source, target)
        if tone == "beginner":
            return polished
        if target_journal:
            polished = polished.replace(
                "# 文献调研报告：",
                f"# 面向 {target_journal} 风格的文献调研报告：",
                1,
            )
        return polished
