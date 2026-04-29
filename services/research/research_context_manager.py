from __future__ import annotations

import json
import logging
from typing import Any

from domain.schemas.research import PaperCandidate, ResearchReport, ResearchTask
from domain.schemas.research_context import (
    CompressedPaperSummary,
    PaperSummaryLevel,
    QAPair,
    ResearchContext,
    ResearchContextPaperMeta,
    ResearchContextSlice,
    ResearchUserPreferences,
    utc_now,
)
from domain.schemas.sub_manager import SubManagerState, TaskStep

logger = logging.getLogger(__name__)


class ResearchContextManager:
    def build_from_artifacts(
        self,
        *,
        task: ResearchTask | None = None,
        report: ResearchReport | None = None,
        papers: list[PaperCandidate] | None = None,
        selected_paper_ids: list[str] | None = None,
        history_entries: list[dict[str, Any]] | None = None,
        paper_summaries: list[CompressedPaperSummary] | None = None,
        current_task_plan: list[TaskStep] | None = None,
        sub_manager_states: dict[str, SubManagerState] | None = None,
        user_preferences: ResearchUserPreferences | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ResearchContext:
        selected_ids = list(dict.fromkeys(selected_paper_ids or []))
        resolved_papers = list(papers or [])
        imported_papers = [
            self.paper_meta_from_candidate(paper)
            for paper in resolved_papers
            if paper.ingest_status == "ingested"
        ]
        workspace = task.workspace if task else None
        history = self._history_pairs(history_entries or [])
        return ResearchContext(
            research_topic=(task.topic if task else None) or (report.topic if report else "") or "",
            research_goals=self._dedupe(
                [
                    *(workspace.research_questions if workspace else []),
                    *(([workspace.objective] if workspace and workspace.objective else [])),
                ]
            ),
            selected_papers=selected_ids,
            imported_papers=imported_papers,
            known_conclusions=self._dedupe(
                [
                    *(report.highlights if report else []),
                    *(workspace.key_findings if workspace else []),
                ]
            ),
            open_questions=self._dedupe(
                [
                    *(report.gaps if report else []),
                    *(workspace.evidence_gaps if workspace else []),
                ]
            ),
            session_history=history,
            user_preferences=user_preferences or ResearchUserPreferences(),
            paper_summaries=list(paper_summaries or []),
            current_task_plan=list(current_task_plan or []),
            sub_manager_states=self._merge_sub_manager_states(sub_manager_states),
            metadata=metadata or {},
        )

    def compress_papers(
        self,
        *,
        papers: list[PaperCandidate],
        selected_paper_ids: list[str] | None = None,
        paper_reading_skill: Any | None = None,
        max_papers: int = 6,
    ) -> list[CompressedPaperSummary]:
        if not papers:
            return []
        if paper_reading_skill is None:
            from services.research.capabilities import PaperReader

            paper_reading_skill = PaperReader()
        selected = set(selected_paper_ids or [])
        ranked_papers = list(papers)
        ranked_papers.sort(
            key=lambda paper: (
                1 if selected and paper.paper_id in selected else 0,
                float(paper.relevance_score or 0.0),
                int(paper.citations or 0),
                int(paper.year or 0),
            ),
            reverse=True,
        )
        summaries: list[CompressedPaperSummary] = []
        for paper in ranked_papers[: max(1, max_papers)]:
            card = paper_reading_skill.extract(paper=paper)
            paragraph_summary = self._compact_text(
                card.summary or paper.summary or paper.abstract or paper.title,
                limit=180,
            )
            section_summary = self._compact_text(
                "；".join(
                    item
                    for item in (
                        f"贡献: {card.contribution}",
                        f"方法: {card.method}",
                        f"实验: {card.experiment}",
                    )
                    if item
                ),
                limit=320,
            )
            document_summary = self._compact_text(
                "；".join(
                    item
                    for item in (
                        f"贡献: {card.contribution}",
                        f"方法: {card.method}",
                        f"实验: {card.experiment}",
                        f"局限: {card.limitation}",
                    )
                    if item
                ),
                limit=480,
            )
            for level, summary in (
                ("paragraph", paragraph_summary),
                ("section", section_summary),
                ("document", document_summary),
            ):
                summaries.append(
                    CompressedPaperSummary(
                        paper_id=paper.paper_id,
                        level=level,  # type: ignore[arg-type]
                        summary=summary,
                        source_section_ids=[f"{paper.paper_id}:{level}"],
                        relevance_score=paper.relevance_score,
                        metadata={
                            "title": paper.title,
                            "year": paper.year,
                            "source": paper.source,
                            "selected": paper.paper_id in selected,
                            "has_pdf": bool(paper.pdf_url),
                        },
                    )
                )
        return summaries

    def update_context(
        self,
        *,
        current_context: ResearchContext | None = None,
        topic: str = "",
        keywords: list[str] | None = None,
        goals: list[str] | None = None,
        known_conclusions: list[str] | None = None,
        selected_papers: list[str] | None = None,
        imported_papers: list[ResearchContextPaperMeta] | None = None,
        open_questions: list[str] | None = None,
        session_history: list[QAPair] | None = None,
        paper_summaries: list[CompressedPaperSummary] | None = None,
        current_task_plan: list[TaskStep] | None = None,
        sub_manager_states: dict[str, SubManagerState] | None = None,
        user_preferences: ResearchUserPreferences | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ResearchContext:
        base = (current_context or ResearchContext()).model_copy(deep=True)
        imported_by_id = {paper.paper_id: paper for paper in base.imported_papers}
        for paper in imported_papers or []:
            imported_by_id[paper.paper_id] = paper
        keyword_list = self._dedupe([*(goals or []), *(keywords or [])])
        return base.model_copy(
            update={
                "research_topic": topic or base.research_topic,
                "research_goals": self._dedupe([*base.research_goals, *keyword_list]),
                "selected_papers": self._dedupe([*base.selected_papers, *(selected_papers or [])]),
                "imported_papers": list(imported_by_id.values()),
                "known_conclusions": self._dedupe(
                    [*base.known_conclusions, *(known_conclusions or [])]
                ),
                "open_questions": self._dedupe([*base.open_questions, *(open_questions or [])]),
                "paper_summaries": self._merge_paper_summaries(
                    base.paper_summaries,
                    paper_summaries or [],
                ),
                "current_task_plan": list(current_task_plan or base.current_task_plan),
                "sub_manager_states": self._merge_sub_manager_states(
                    {
                        **base.sub_manager_states,
                        **(sub_manager_states or {}),
                    }
                ),
                "session_history": [*base.session_history, *(session_history or [])][
                    -(
                        user_preferences.max_history_turns
                        if user_preferences is not None
                        else base.user_preferences.max_history_turns
                    ) :
                ],
                "user_preferences": user_preferences or base.user_preferences,
                "metadata": {
                    **base.metadata,
                    **(metadata or {}),
                },
                "updated_at": utc_now(),
            }
        )

    def slice_for_agent(
        self,
        context: ResearchContext,
        *,
        paper_ids: list[str] | None = None,
        max_papers: int | None = None,
        max_history_turns: int | None = None,
        max_task_steps: int | None = None,
        include_preferences: bool = True,
        agent_scope: str = "worker",
        sub_manager_key: str | None = None,
        summary_level: PaperSummaryLevel | None = None,
    ) -> ResearchContextSlice:
        target_paper_ids = set(paper_ids or context.selected_papers)
        paper_limit = max_papers or context.user_preferences.max_selected_papers
        history_limit = max_history_turns or context.user_preferences.max_history_turns
        task_step_limit = max_task_steps or (12 if agent_scope == "manager" else 6)
        summary_target = summary_level or ("document" if agent_scope == "worker" else "section")
        imported_papers = [
            paper
            for paper in context.imported_papers
            if not target_paper_ids or paper.paper_id in target_paper_ids
        ][:paper_limit]
        relevant_summaries = [
            summary
            for summary in context.paper_summaries
            if (
                (not target_paper_ids or summary.paper_id in target_paper_ids)
                and summary.level == summary_target
            )
        ][: max(paper_limit, 1)]
        if not relevant_summaries:
            relevant_summaries = [
                summary
                for summary in context.paper_summaries
                if not target_paper_ids or summary.paper_id in target_paper_ids
            ][: max(paper_limit, 1)]
        current_task_plan = [
            step
            for step in context.current_task_plan
            if sub_manager_key is None
            or step.assigned_to == sub_manager_key
            or step.metadata.get("sub_manager") == sub_manager_key
        ][:task_step_limit]
        return ResearchContextSlice(
            research_topic=context.research_topic,
            research_goals=context.research_goals[:8],
            selected_papers=[
                paper_id
                for paper_id in context.selected_papers
                if not target_paper_ids or paper_id in target_paper_ids
            ][:paper_limit],
            imported_papers=imported_papers,
            known_conclusions=context.known_conclusions[:8],
            open_questions=context.open_questions[:8],
            session_history=context.session_history[-history_limit:],
            relevant_summaries=relevant_summaries,
            current_task_plan=current_task_plan,
            sub_manager_state=(
                context.sub_manager_states.get(sub_manager_key)
                if sub_manager_key is not None
                else None
            ),
            context_scope=agent_scope,  # type: ignore[arg-type]
            summary_level=summary_target,
            user_preferences=context.user_preferences if include_preferences else None,
            memory_context={
                "recalled_memories": list(context.metadata.get("recalled_memories") or []),
                "recalled_memory_ids": list(context.metadata.get("recalled_memory_ids") or []),
            },
            metadata={
                **dict(context.metadata),
                "context_scope": agent_scope,
                "summary_level": summary_target,
            },
        )

    def paper_meta_from_candidate(self, paper: PaperCandidate) -> ResearchContextPaperMeta:
        return ResearchContextPaperMeta(
            paper_id=paper.paper_id,
            title=paper.title,
            authors=list(paper.authors),
            year=paper.year,
            source=paper.source,
            document_id=str(paper.metadata.get("document_id") or "") or None,
            summary=paper.summary,
            metadata=dict(paper.metadata),
        )

    def _history_pairs(self, history_entries: list[dict[str, Any]]) -> list[QAPair]:
        history: list[QAPair] = []
        for entry in history_entries:
            question = str(entry.get("question") or "").strip()
            answer = str(entry.get("answer") or "").strip()
            if not question or not answer:
                continue
            history.append(
                QAPair(
                    question=question,
                    answer=answer,
                    citations=list(entry.get("citations") or []),
                    metadata={
                        "task_id": entry.get("task_id"),
                        "conversation_id": entry.get("conversation_id"),
                        "document_ids": list(entry.get("document_ids") or []),
                    },
                )
            )
        return history

    def _dedupe(self, items: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    def _compact_text(self, text: str, *, limit: int) -> str:
        normalized = " ".join((text or "").strip().split())
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[: max(limit - 1, 1)].rstrip()}…"

    def _merge_paper_summaries(
        self,
        existing: list[CompressedPaperSummary],
        incoming: list[CompressedPaperSummary],
    ) -> list[CompressedPaperSummary]:
        merged: dict[tuple[str, str], CompressedPaperSummary] = {
            (summary.paper_id, summary.level): summary for summary in existing
        }
        for summary in incoming:
            merged[(summary.paper_id, summary.level)] = summary
        return list(merged.values())

    def _merge_sub_manager_states(
        self,
        overrides: dict[str, SubManagerState] | None = None,
    ) -> dict[str, SubManagerState]:
        base = ResearchContext().sub_manager_states
        merged = {name: state.model_copy(deep=True) for name, state in base.items()}
        for name, state in (overrides or {}).items():
            merged[name] = state
        return merged

    # ------------------------------------------------------------------
    # Hermes-inspired context compression
    # ------------------------------------------------------------------
    #
    # Design reference: hermes-agent/agent/context_compressor.py
    #
    # Hermes compresses *messages* in 3 phases:
    #   1. Prune old tool results → cheap 1-line summaries (no LLM)
    #   2. Protect head + tail by token budget; identify middle to compress
    #   3. LLM-summarize middle turns with structured template
    #
    # Research-Copilot compresses a *ResearchContextSlice* (structured
    # Pydantic model, not a flat message list).  We adapt the same
    # philosophy:
    #
    #   Phase 1 — Metadata pruning (cheap, no LLM):
    #     Strip raw page content from metadata, paper metadata, QA
    #     metadata, memory_context.  Like Hermes's _prune_old_tool_results
    #     replacing 50 KB tool outputs with 1-line summaries.
    #
    #   Phase 2 — Session history compaction:
    #     Protect the most recent N QA pairs (tail).  Fold older pairs
    #     into a structured rolling summary QA entry — preserving what
    #     was asked, what was concluded, what's still pending.  Like
    #     Hermes's "protect tail by token budget + summarize middle".
    #
    #   Phase 3 — Size audit:
    #     If the slice is still over budget after phases 1-2, apply
    #     progressive field stripping (summaries, conclusions, papers).
    #     Like Hermes's anti-thrashing fallback.

    # Defaults
    _PRUNE_VALUE_LIMIT = 500          # chars; metadata values above this get pruned
    _HISTORY_TAIL_PROTECT = 3         # QA pairs to keep verbatim (most recent)
    _HISTORY_QUESTION_LIMIT = 120     # chars kept per question in the summary
    _HISTORY_ANSWER_LIMIT = 300       # chars kept per answer in the summary
    _MEMORY_ITEM_LIMIT = 200          # chars per recalled memory item
    _PAPER_SUMMARY_LIMIT = 200        # chars per imported paper summary
    _DEFAULT_BUDGET_CHARS = 100_000   # target budget for serialized slice

    def compress_context_slice(
        self,
        context_slice: ResearchContextSlice,
        *,
        budget_chars: int | None = None,
        protect_tail: int | None = None,
    ) -> ResearchContextSlice:
        """Compress a *ResearchContextSlice* so it fits within *budget_chars*.

        Inspired by Hermes's 3-phase context compressor:
          Phase 1 — prune heavy metadata values (cheap, no LLM)
          Phase 2 — fold old session history into a rolling summary
          Phase 3 — progressive field stripping if still over budget

        The method is **idempotent** — calling it on an already-small slice
        is a no-op.  It never calls an LLM; all compression is rule-based
        and deterministic.

        Args:
            context_slice: The slice to compress (deep-copied internally).
            budget_chars: Target serialized JSON size.  Defaults to 100 000.
            protect_tail: Number of most-recent QA pairs to keep verbatim.
                Defaults to 3.

        Returns:
            A new ResearchContextSlice that fits within the budget.
        """
        budget = budget_chars or self._DEFAULT_BUDGET_CHARS
        tail_n = protect_tail if protect_tail is not None else self._HISTORY_TAIL_PROTECT

        if self._slice_chars(context_slice) <= budget:
            return context_slice

        compressed = context_slice.model_copy(deep=True)
        phase_stats: dict[str, Any] = {}

        # ---- Phase 1: Metadata pruning (like Hermes _prune_old_tool_results) ----
        before_phase1 = self._slice_chars(compressed)
        compressed.metadata = self._prune_dict(
            compressed.metadata,
            value_limit=self._PRUNE_VALUE_LIMIT,
            preserve_keys={"context_scope", "summary_level", "context_compression"},
        )
        for paper in compressed.imported_papers:
            paper.metadata = self._prune_dict(paper.metadata, value_limit=self._PRUNE_VALUE_LIMIT)
            paper.summary = self._compact_text(paper.summary or "", limit=self._PAPER_SUMMARY_LIMIT)
        for qa in compressed.session_history:
            qa.metadata = self._prune_dict(qa.metadata, value_limit=self._PRUNE_VALUE_LIMIT)
        for summary in compressed.relevant_summaries:
            summary.metadata = self._prune_dict(summary.metadata, value_limit=self._PRUNE_VALUE_LIMIT)
        compressed.memory_context = self._prune_memory_context(compressed.memory_context)
        after_phase1 = self._slice_chars(compressed)
        phase_stats["phase1_pruned_chars"] = before_phase1 - after_phase1

        if after_phase1 <= budget:
            compressed.metadata["_compression"] = {
                "phases_applied": ["prune_metadata"],
                **phase_stats,
            }
            logger.info(
                "Context compression Phase 1 sufficient: %d -> %d chars (budget=%d)",
                before_phase1, after_phase1, budget,
            )
            return compressed

        # ---- Phase 2: Session history compaction (like Hermes protect tail + summarize middle) ----
        before_phase2 = self._slice_chars(compressed)
        compressed.session_history = self._compress_session_history(
            compressed.session_history,
            protect_tail=tail_n,
        )
        after_phase2 = self._slice_chars(compressed)
        phase_stats["phase2_history_chars"] = before_phase2 - after_phase2

        if after_phase2 <= budget:
            compressed.metadata["_compression"] = {
                "phases_applied": ["prune_metadata", "compress_history"],
                **phase_stats,
            }
            logger.info(
                "Context compression Phase 2 sufficient: %d -> %d chars (budget=%d)",
                before_phase1, after_phase2, budget,
            )
            return compressed

        # ---- Phase 3: Progressive field stripping (like Hermes anti-thrashing fallback) ----
        before_phase3 = self._slice_chars(compressed)
        strips_applied: list[str] = []

        # Step 3a: cap summaries to top 3
        if len(compressed.relevant_summaries) > 3:
            compressed.relevant_summaries = compressed.relevant_summaries[:3]
            strips_applied.append("cap_summaries")
            if self._slice_chars(compressed) <= budget:
                phase_stats["phase3_strips"] = strips_applied
                compressed.metadata["_compression"] = {
                    "phases_applied": ["prune_metadata", "compress_history", "strip_fields"],
                    **phase_stats,
                }
                return compressed

        # Step 3b: strip conclusions + open questions
        compressed.known_conclusions = compressed.known_conclusions[:3]
        compressed.open_questions = compressed.open_questions[:3]
        strips_applied.append("cap_conclusions_questions")
        if self._slice_chars(compressed) <= budget:
            phase_stats["phase3_strips"] = strips_applied
            compressed.metadata["_compression"] = {
                "phases_applied": ["prune_metadata", "compress_history", "strip_fields"],
                **phase_stats,
            }
            return compressed

        # Step 3c: cap imported papers to 3, strip remaining metadata
        compressed.imported_papers = compressed.imported_papers[:3]
        for paper in compressed.imported_papers:
            paper.metadata = {}
        strips_applied.append("cap_papers")
        if self._slice_chars(compressed) <= budget:
            phase_stats["phase3_strips"] = strips_applied
            compressed.metadata["_compression"] = {
                "phases_applied": ["prune_metadata", "compress_history", "strip_fields"],
                **phase_stats,
            }
            return compressed

        # Step 3d: reduce history to last 1 entry
        compressed.session_history = compressed.session_history[-1:]
        compressed.current_task_plan = compressed.current_task_plan[:2]
        strips_applied.append("minimal_history")
        if self._slice_chars(compressed) <= budget:
            phase_stats["phase3_strips"] = strips_applied
            compressed.metadata["_compression"] = {
                "phases_applied": ["prune_metadata", "compress_history", "strip_fields"],
                **phase_stats,
            }
            return compressed

        # Step 3e: nuclear — keep only topic, goals, selected_papers
        #   Truncate any surviving history answers to fit the budget
        #   (like Hermes dropping all middle turns entirely)
        after_phase3 = self._slice_chars(compressed)
        phase_stats["phase3_strips"] = strips_applied
        phase_stats["phase3_chars"] = before_phase3 - after_phase3
        logger.warning(
            "Context compression Phase 3 exhausted all strips: %d -> %d chars (budget=%d). "
            "Falling back to minimal slice.",
            before_phase1, after_phase3, budget,
        )
        truncated_history: list[QAPair] = []
        for qa in compressed.session_history[-1:]:
            truncated_history.append(QAPair(
                question=self._compact_text(qa.question, limit=200),
                answer=self._compact_text(qa.answer, limit=500),
                citations=qa.citations[:3],
                metadata={"_compacted": True},
            ))
        return ResearchContextSlice(
            research_topic=compressed.research_topic,
            research_goals=compressed.research_goals[:3],
            selected_papers=compressed.selected_papers[:8],
            session_history=truncated_history,
            metadata={"_compression": {
                "phases_applied": ["prune_metadata", "compress_history", "strip_fields", "nuclear_fallback"],
                **phase_stats,
            }},
        )

    # ---- Phase 1 helpers ----

    def _prune_dict(
        self,
        data: dict[str, Any],
        *,
        value_limit: int = 500,
        preserve_keys: set[str] | None = None,
    ) -> dict[str, Any]:
        """Replace large values in a dict with informative 1-line placeholders.

        Like Hermes's ``_summarize_tool_result`` — instead of a generic
        "[pruned]" we keep the type and size so the LLM has some signal.
        """
        preserved = preserve_keys or set()
        pruned: dict[str, Any] = {}
        for key, value in data.items():
            if key in preserved:
                pruned[key] = value
                continue
            serialized = self._safe_serialize(value)
            if len(serialized) <= value_limit:
                pruned[key] = value
            elif isinstance(value, str):
                pruned[key] = f"{value[:80]}… [pruned: {len(value):,} chars]"
            elif isinstance(value, list):
                pruned[key] = f"[pruned list: {len(value)} items, {len(serialized):,} chars]"
            elif isinstance(value, dict):
                pruned[key] = f"[pruned dict: {len(value)} keys, {len(serialized):,} chars]"
            else:
                pruned[key] = f"[pruned: {len(serialized):,} chars]"
        return pruned

    def _prune_memory_context(self, memory_context: dict[str, Any]) -> dict[str, Any]:
        """Truncate recalled memories to bounded 1-line items."""
        if not memory_context:
            return {}
        pruned = dict(memory_context)
        memories = pruned.get("recalled_memories")
        if isinstance(memories, list):
            pruned["recalled_memories"] = [
                self._compact_text(str(m), limit=self._MEMORY_ITEM_LIMIT)
                for m in memories[:10]
            ]
        return pruned

    # ---- Phase 2 helpers ----

    def _compress_session_history(
        self,
        history: list[QAPair],
        *,
        protect_tail: int = 3,
    ) -> list[QAPair]:
        """Fold old QA pairs into a rolling summary entry, keep recent ones.

        Like Hermes's "protect tail by token budget + summarize middle":
        - Recent ``protect_tail`` pairs stay verbatim (Hermes tail)
        - Older pairs get folded into a single structured summary QA
          (Hermes LLM summary, but rule-based here for speed)

        The summary preserves: what was asked, key conclusions, unresolved
        questions — structured for the LLM to parse as context.
        """
        if len(history) <= protect_tail:
            return history

        old = history[:-protect_tail] if protect_tail > 0 else history
        tail = history[-protect_tail:] if protect_tail > 0 else []

        # Build structured rolling summary (Hermes-style template)
        summary_parts: list[str] = []
        resolved: list[str] = []
        for i, qa in enumerate(old, 1):
            q_short = self._compact_text(qa.question, limit=self._HISTORY_QUESTION_LIMIT)
            a_short = self._compact_text(qa.answer, limit=self._HISTORY_ANSWER_LIMIT)
            summary_parts.append(f"{i}. Q: {q_short}\n   A: {a_short}")
            # Extract key conclusions from answers (heuristic)
            if len(qa.answer) > 50:
                first_sentence = qa.answer.split("。")[0].split(". ")[0]
                resolved.append(self._compact_text(first_sentence, limit=100))

        summary_question = (
            f"[CONTEXT COMPACTION] {len(old)} earlier QA turns were compacted. "
            f"This is background reference — do NOT re-answer these questions."
        )
        summary_answer_lines = [
            "## Compacted Conversation History",
            f"Turns compacted: {len(old)}",
            "",
            "## Resolved Q&A",
            *summary_parts,
        ]
        if resolved:
            summary_answer_lines += [
                "",
                "## Key Conclusions",
                *[f"- {c}" for c in resolved[:8]],
            ]

        summary_qa = QAPair(
            question=summary_question,
            answer="\n".join(summary_answer_lines),
            citations=[],
            metadata={"_compacted_turn_count": len(old), "_is_summary": True},
        )

        return [summary_qa, *tail]

    # ---- Measurement helpers ----

    @staticmethod
    def _slice_chars(context_slice: ResearchContextSlice) -> int:
        """Return serialized JSON size in characters."""
        try:
            return len(json.dumps(
                context_slice.model_dump(mode="json"),
                ensure_ascii=False,
                default=str,
            ))
        except Exception:
            return 0

    @staticmethod
    def _safe_serialize(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)
