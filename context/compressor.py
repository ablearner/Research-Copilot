"""3-layer context compression engine for Kepler conversations.

Layer 1: Prune old tool results (replace with one-line summary).
Layer 2: Split messages into head / compressible-middle / tail.
Layer 3: LLM-summarize the compressible middle region.

Includes anti-thrashing guards (改进 14) to avoid ineffective
compression loops and protect the most recent user message.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from context.token_counter import estimate_tokens_rough

logger = logging.getLogger(__name__)

# ── Summary prompt template ────────────────────────────────────────

_SUMMARY_SYSTEM_PROMPT = """\
You are a context compressor for a research assistant.
Summarize the conversation turns below into a structured handoff note.
The summary will replace these turns in the assistant's context window.

Use the following template:

## Active Task
(What the user is currently asking or working on)

## Completed Actions
(Bullet list of actions already taken and their outcomes)

## Key Findings
(Important facts, evidence, or conclusions discovered)

## Pending / Remaining
(What still needs to be done)

## Resolved Questions
(Questions that have been answered)

Be concise. Prefer bullet points. Preserve paper IDs, URLs, and numeric results exactly.
Do NOT invent information not present in the turns.
"""

_SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — This is a structured summary of earlier conversation "
    "turns. Treat it as authoritative background context.]\n\n"
)

# ── Tool output patterns that can be safely pruned ─────────────────

_PRUNABLE_TOOL_PREFIXES = (
    "hybrid_retrieve",
    "query_graph",
    "parse_document",
    "arxiv_search",
    "semantic_scholar",
    "openalex_search",
    "ieee_search",
)


class ContextCompressor:
    """3-layer context compression with anti-thrashing guards."""

    def __init__(
        self,
        llm_adapter: Any | None = None,
        *,
        target_budget_ratio: float = 0.75,
        protect_first_n: int = 2,
        protect_last_n: int = 4,
        tail_token_budget: int = 8_000,
        summary_prompt: str = _SUMMARY_SYSTEM_PROMPT,
    ) -> None:
        self.llm_adapter = llm_adapter
        self.target_budget_ratio = target_budget_ratio
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n
        self.tail_token_budget = tail_token_budget
        self.summary_prompt = summary_prompt

        # Anti-thrashing state (改进 14)
        self._ineffective_count: int = 0
        self._summary_cooldown_until: float = 0.0

    # ── Public API ─────────────────────────────────────────────────

    def should_compress(self, current_tokens: int, budget_available: int) -> bool:
        """Return True if compression should be attempted."""
        threshold = int(budget_available * self.target_budget_ratio)
        if current_tokens < threshold:
            return False
        if self._ineffective_count >= 2:
            logger.warning(
                "Compression skipped — last %d compressions saved <10%% each",
                self._ineffective_count,
            )
            return False
        if time.monotonic() < self._summary_cooldown_until:
            return False
        return True

    async def compress_messages(
        self,
        messages: list[dict[str, Any]],
        model_context_length: int,
        *,
        protected_tool_names: set[str] | None = None,
        focus_topic: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run 3-layer compression and return a shorter message list."""
        budget = int(model_context_length * self.target_budget_ratio)
        before_tokens = estimate_tokens_rough(messages)
        if before_tokens <= budget:
            return messages

        # Layer 1: prune old tool outputs
        messages = self._prune_tool_outputs(
            messages,
            protected_tool_names or set(),
        )
        if estimate_tokens_rough(messages) <= budget:
            self._record_savings(before_tokens, messages)
            return messages

        # Layer 2: identify compressible region
        head, middle, tail = self._split_protected(messages, tail_budget=budget // 3)
        if not middle:
            self._record_savings(before_tokens, messages)
            return messages

        # Layer 3: LLM summarize
        summary_msg = await self._summarize_region(middle, focus_topic=focus_topic)
        compressed = [*head, summary_msg, *tail]
        compressed = self._sanitize_tool_pairs(compressed)
        self._record_savings(before_tokens, compressed)
        return compressed

    # ── Layer 1: Tool output pruning ───────────────────────────────

    def _prune_tool_outputs(
        self,
        messages: list[dict[str, Any]],
        protected_names: set[str],
    ) -> list[dict[str, Any]]:
        """Replace old large tool results with a one-line stub."""
        n = len(messages)
        tail_start = max(0, n - self.protect_last_n)
        result: list[dict[str, Any]] = []

        for i, msg in enumerate(messages):
            if i >= tail_start or msg.get("role") != "tool":
                result.append(msg)
                continue
            content = str(msg.get("content", ""))
            if len(content) < 500:
                result.append(msg)
                continue
            tool_name = msg.get("name", "")
            if tool_name in protected_names:
                result.append(msg)
                continue
            if any(tool_name.startswith(prefix) for prefix in _PRUNABLE_TOOL_PREFIXES):
                stub = f'{{"ok": true, "pruned": true, "summary": "[tool output pruned — {len(content)} chars]"}}'
                result.append({**msg, "content": stub})
            else:
                result.append(msg)
        return result

    # ── Layer 2: Split into head / middle / tail ───────────────────

    def _split_protected(
        self,
        messages: list[dict[str, Any]],
        tail_budget: int,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        head_end = min(self.protect_first_n, len(messages))
        tail_start = self._find_tail_cut(messages, head_end, tail_budget)
        tail_start = self._ensure_last_user_in_tail(messages, tail_start, head_end)
        head = messages[:head_end]
        middle = messages[head_end:tail_start]
        tail = messages[tail_start:]
        return head, middle, tail

    def _find_tail_cut(
        self,
        messages: list[dict[str, Any]],
        head_end: int,
        tail_budget: int,
    ) -> int:
        """Walk backward from end to find where tail begins by token budget."""
        accumulated = 0
        for i in range(len(messages) - 1, head_end - 1, -1):
            msg_tokens = estimate_tokens_rough([messages[i]])
            if accumulated + msg_tokens > tail_budget:
                return i + 1
            accumulated += msg_tokens
        return head_end

    def _ensure_last_user_in_tail(
        self,
        messages: list[dict[str, Any]],
        cut_idx: int,
        head_end: int,
    ) -> int:
        """Guarantee the most recent user message is in the protected tail."""
        for i in range(len(messages) - 1, head_end - 1, -1):
            if messages[i].get("role") == "user":
                if i < cut_idx:
                    return max(i, head_end + 1)
                return cut_idx
        return cut_idx

    # ── Layer 3: LLM summarization ─────────────────────────────────

    async def _summarize_region(
        self,
        turns: list[dict[str, Any]],
        *,
        focus_topic: str | None = None,
    ) -> dict[str, Any]:
        """Generate a structured summary for the middle turns."""
        serialized = self._serialize_turns(turns)

        if self.llm_adapter is not None:
            try:
                from pydantic import BaseModel, Field

                class _SummaryResult(BaseModel):
                    summary: str = Field(description="Structured summary of the conversation turns")

                user_content = f"Summarize these conversation turns:\n\n{serialized}"
                if focus_topic:
                    user_content += f"\n\nFocus topic: {focus_topic}"

                result = await self.llm_adapter.generate_structured(
                    prompt=self.summary_prompt,
                    input_data={"turns": serialized, "focus_topic": focus_topic or ""},
                    response_model=_SummaryResult,
                )
                summary_text = result.summary
            except Exception:
                logger.warning("LLM summarization failed, using fallback", exc_info=True)
                self._summary_cooldown_until = time.monotonic() + 60.0
                summary_text = self._fallback_summary(turns)
        else:
            summary_text = self._fallback_summary(turns)

        return {
            "role": "assistant",
            "content": _SUMMARY_PREFIX + summary_text,
        }

    def _fallback_summary(self, turns: list[dict[str, Any]]) -> str:
        """Cheap text-only summary when no LLM is available."""
        lines = [f"[{len(turns)} conversation turns were compressed]"]
        for turn in turns:
            role = turn.get("role", "?")
            content = str(turn.get("content", ""))[:120]
            lines.append(f"- {role}: {content}")
            if len(lines) > 20:
                lines.append(f"  ... and {len(turns) - 20} more turns")
                break
        return "\n".join(lines)

    def _serialize_turns(self, turns: list[dict[str, Any]]) -> str:
        from security.redact import redact_sensitive_text

        parts: list[str] = []
        for turn in turns:
            role = turn.get("role", "?")
            content = str(turn.get("content", ""))
            if len(content) > 2000:
                content = content[:1997] + "..."
            content = redact_sensitive_text(content)
            parts.append(f"[{role}] {content}")
        return "\n\n".join(parts)

    # ── Anti-thrashing (改进 14) ───────────────────────────────────

    def _record_savings(
        self,
        before_tokens: int,
        compressed: list[dict[str, Any]],
    ) -> None:
        after_tokens = estimate_tokens_rough(compressed)
        if before_tokens <= 0:
            return
        savings_pct = (before_tokens - after_tokens) / before_tokens * 100
        logger.info(
            "Context compression: %d → %d tokens (%.1f%% saved)",
            before_tokens,
            after_tokens,
            savings_pct,
        )
        if savings_pct < 10:
            self._ineffective_count += 1
        else:
            self._ineffective_count = 0

    # ── Tool pair sanitization ─────────────────────────────────────

    def _sanitize_tool_pairs(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Ensure every tool_call has a matching tool result message."""
        sanitized: list[dict[str, Any]] = []
        pending_ids: list[str] = []

        def _flush() -> None:
            for tid in pending_ids:
                sanitized.append({
                    "role": "tool",
                    "tool_call_id": tid,
                    "content": '{"ok": false, "error": "tool call was compressed away"}',
                })
            pending_ids.clear()

        for msg in messages:
            role = msg.get("role", "")
            if role == "assistant":
                _flush()
                sanitized.append(msg)
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        pending_ids.append(tc["id"])
            elif role == "tool":
                tid = str(msg.get("tool_call_id", ""))
                if tid in pending_ids:
                    sanitized.append(msg)
                    pending_ids.remove(tid)
                else:
                    sanitized.append(msg)
            else:
                _flush()
                sanitized.append(msg)

        _flush()
        return sanitized
