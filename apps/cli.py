from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import time
import unicodedata
from typing import Any
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import Settings

PLUGIN_NAMES = (
    "academic_search",
    "zotero_local_mcp",
    "local_code_execution",
    "trajectory_inspector",
    "terminal_agent",
)

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_BLUE = "\033[38;5;39m"
_BLUE_SOFT = "\033[38;5;75m"
_BLUE_DEEP = "\033[38;5;33m"
_CYAN = "\033[38;5;45m"
_ICE = "\033[38;5;159m"
_WHITE = "\033[38;5;255m"
_SLATE = "\033[38;5;110m"

_KEPLER_LETTER_GRADIENT = [_ICE, _CYAN, _BLUE_SOFT, _BLUE, _BLUE_DEEP, _BLUE]

_KEPLER_ART = [
    " ██╗  ██╗███████╗██████╗ ██╗     ███████╗██████╗                                          ",
    " ██║ ██╔╝██╔════╝██╔══██╗██║     ██╔════╝██╔══██╗                                         ",
    " █████╔╝ █████╗  ██████╔╝██║     █████╗  ██████╔╝                                         ",
    " ██╔═██╗ ██╔══╝  ██╔═══╝ ██║     ██╔══╝  ██╔══██╗                                         ",
    " ██║  ██╗███████╗██║     ███████╗███████╗██║  ██║                                         ",
    " ╚═╝  ╚═╝╚══════╝╚═╝     ╚══════╝╚══════╝╚═╝  ╚═╝                                         ",
    "                                 // science & progress //                                  ",
    "                                                                                            ",
    "        ·        ✦            ·          ✦          ·            ✦        ·               ",
    "                              /\\                                                            ",
    "                             /  \\                          _/\\_                              ",
    "                            /_/\\_\\                        <_[]_>                             ",
    "                              ||                           /__\\                              ",
    "               ·· ─ ─ ─ ─ ─ ─ || ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ || ─ ─ ··                      ",
    "                        .-''''''''''''''''''''''''''''''''''''-.                           ",
    "                    .-'                                            '-.                     ",
    "                  .'         ╭──────────────────────────────╮          '.                   ",
    "                .'         ╭─╯   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓     ╰─╮          '.                ",
    "               /          ╭╯    ▓▓▓▓░░░░░░░░░░░░░░░░▓▓▓▓      ╰╮           \\               ",
    "              ;          ╭╯    ▓▓▓░░▓▓▓▓▓▓▓▓▓▓▓▓▓░░░▓▓▓         ╰╮           ;              ",
    "              |          │    ▓▓▓░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░▓▓▓          │           |              ",
    "     ⬡  · ─── │ ──────── │   ▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓          │ ──────── · ⬡            ",
    "              |          │    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓           │           |              ",
    "              ;           ╰╮    ▓▓▓░░░░░░░░░░░░░░▓▓▓          ╭╯           ;              ",
    "               \\           ╰╮    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓          ╭╯           /               ",
    "                '.           ╰─╮     ▓▓▓▓▓▓▓▓▓▓▓         ╭─╯          .'                ",
    "                  '.            ╰──────────────────────╯            .'                   ",
    "                    '-.                                            .-'                     ",
    "                       '-.______________________________________. -'                        ",
    "               ·· ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ··                          ",
    "                             \\__/                                                /\\       ",
    "                            .-''-.                                              /__\\      ",
    "                           /|_||_|\\                                            <_[]_>     ",
    "                            /_||_\\                                              /__\\      ",
    "                              /\\                                                           ",
    "        ·        ✦            ·          ✦          ·            ✦        ·               ",
]


def build_parser() -> argparse.ArgumentParser:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Research-Copilot terminal and management CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-conversations", help="List saved research conversations.")
    subparsers.add_parser("show-profile", help="Show the current user research profile.")
    subparsers.add_parser("doctor", help="Run local environment diagnostics.")

    status_parser = subparsers.add_parser("status", help="Show runtime status and optionally a conversation summary.")
    status_parser.add_argument("--conversation-id")

    trajectory = subparsers.add_parser("trajectory", help="Show a conversation trajectory snapshot.")
    trajectory.add_argument("conversation_id")
    trajectory.add_argument("--messages", type=int, default=12)
    trajectory.add_argument("--events", type=int, default=12)

    update_profile = subparsers.add_parser("update-profile", help="Update user research profile.")
    update_profile.add_argument("--topic")
    update_profile.add_argument("--source", action="append", default=[])
    update_profile.add_argument("--keyword", action="append", default=[])
    update_profile.add_argument("--reasoning-style")
    update_profile.add_argument("--note")

    models = subparsers.add_parser("models", help="Inspect or update runtime model preferences.")
    models_subparsers = models.add_subparsers(dest="models_command", required=True)
    models_subparsers.add_parser("show", help="Show current runtime model configuration.")
    models_set = models_subparsers.add_parser("set", help="Set runtime model overrides.")
    models_set.add_argument("--llm-provider")
    models_set.add_argument("--llm-model")
    models_set.add_argument("--embedding-provider")
    models_set.add_argument("--embedding-model")
    models_set.add_argument("--chart-vision-provider")
    models_set.add_argument("--chart-vision-model")

    plugins = subparsers.add_parser("plugins", help="Inspect or configure optional runtime plugins.")
    plugins_subparsers = plugins.add_subparsers(dest="plugins_command", required=True)
    plugins_subparsers.add_parser("list", help="List plugins.")
    plugin_enable = plugins_subparsers.add_parser("enable", help="Enable a plugin.")
    plugin_enable.add_argument("name", choices=sorted(PLUGIN_NAMES))
    plugin_disable = plugins_subparsers.add_parser("disable", help="Disable a plugin.")
    plugin_disable.add_argument("name", choices=sorted(PLUGIN_NAMES))

    agent = subparsers.add_parser("agent", help="Start the interactive terminal agent.")
    agent.add_argument("--conversation-id")
    agent.add_argument("--topic")
    agent.add_argument("--mode", default="auto", choices=["auto", "research", "qa", "import", "document", "chart"])
    agent.add_argument("--days-back", type=int, default=settings.research_default_days_back)
    agent.add_argument("--max-papers", type=int, default=settings.research_default_max_papers)
    agent.add_argument("--source", action="append", default=[])

    return parser


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _terminal_width() -> int:
    return max(72, min(shutil.get_terminal_size(fallback=(120, 30)).columns, 140))


def _visible_len(text: str) -> int:
    length = 0
    in_escape = False
    for char in text:
        if char == "\033":
            in_escape = True
            continue
        if in_escape:
            if char == "m":
                in_escape = False
            continue
        if unicodedata.combining(char):
            continue
        length += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return length


def _pad_visible(text: str, width: int) -> str:
    return text + (" " * max(width - _visible_len(text), 0))


def _clip_visible(text: str, width: int) -> str:
    result: list[str] = []
    in_escape = False
    used = 0
    for char in text:
        if char == "\033":
            in_escape = True
            result.append(char)
            continue
        if in_escape:
            result.append(char)
            if char == "m":
                in_escape = False
            continue
        if unicodedata.combining(char):
            result.append(char)
            continue
        char_width = 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
        if used + char_width > width:
            break
        result.append(char)
        used += char_width
    if in_escape:
        result.append(_RESET)
    return "".join(result)


def _color_line(text: str, color: str) -> str:
    return f"{color}{text}{_RESET}"


def _wrap_visible(text: str, width: int) -> list[str]:
    if width <= 0:
        return [""]
    if not text:
        return [""]

    lines: list[str] = []
    current: list[str] = []
    current_width = 0
    in_escape = False

    def flush() -> None:
        nonlocal current, current_width
        lines.append("".join(current) if current else "")
        current = []
        current_width = 0

    for char in text:
        if char == "\n":
            flush()
            continue
        if char == "\033":
            in_escape = True
            current.append(char)
            continue
        if in_escape:
            current.append(char)
            if char == "m":
                in_escape = False
            continue
        if unicodedata.combining(char):
            current.append(char)
            continue

        char_width = 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
        if current_width > 0 and current_width + char_width > width:
            flush()
        current.append(char)
        current_width += char_width

    if current:
        flush()
    return lines or [""]


def _divider(char: str = "─", *, color: str = _BLUE_SOFT) -> str:
    return f"{color}{char * _terminal_width()}{_RESET}"


def _boxed_lines(lines: list[str], *, title: str | None = None, accent: str = _BLUE, body_color: str = _WHITE) -> None:
    width = _terminal_width()
    inner = max(width - 4, 20)
    if title:
        title_text = f" {title} "
        top_fill = max(inner - _visible_len(title_text), 0)
        print(f"{accent}╭{title_text}{'─' * top_fill}╮{_RESET}")
    else:
        print(f"{accent}╭{'─' * inner}╮{_RESET}")
    for line in lines:
        wrapped_lines = _wrap_visible(line, inner - 1)
        for wrapped in wrapped_lines:
            print(f"{accent}│ {body_color}{_pad_visible(wrapped, inner - 1)}{accent}│{_RESET}")
    print(f"{accent}╰{'─' * inner}╯{_RESET}")


def _render_kepler_banner() -> None:
    art_width = max(len(line) for line in _KEPLER_ART)
    title = f"{_BOLD}{_WHITE}KEPLER{_RESET}"
    subtitle = f"{_BLUE_SOFT}Orbital Research Terminal{_RESET}"
    tagline = f"{_SLATE}Science in motion. Evidence in orbit.{_RESET}"
    lines: list[str] = []
    for index, line in enumerate(_KEPLER_ART):
        centered = line.center(max(art_width, 78))
        if index < len(_KEPLER_LETTER_GRADIENT):
            lines.append(_color_line(centered, _KEPLER_LETTER_GRADIENT[index]))
        else:
            ambient = _BLUE_SOFT if index < 14 else _BLUE_DEEP if index < 26 else _SLATE
            lines.append(_color_line(centered, ambient))
    lines.append("")
    lines.append(f"{title.center(max(art_width, 78) + len(title) - len('KEPLER'))}")
    lines.append(f"{subtitle.center(max(art_width, 78) + _visible_len(subtitle) - len('Orbital Research Terminal'))}")
    lines.append(f"{tagline.center(max(art_width, 78) + _visible_len(tagline) - len('Science in motion. Evidence in orbit.'))}")
    _boxed_lines(lines, title=f"{_BLUE}✦ Kepler Launch Bay{_RESET}", accent=_BLUE_DEEP, body_color=_WHITE)


def _render_welcome_panel() -> None:
    _boxed_lines(
        [
            "你好，我是 KEPLER。",
            "我会帮你做论文检索、候选筛选、导入、Zotero 同步、集合问答和研究轨迹管理。",
        ],
        title=f"{_BLUE}⚗ Welcome{_RESET}",
        accent=_BLUE,
        body_color=_WHITE,
    )


def _progress_bar(value: int, total: int, *, width: int = 10) -> str:
    if total <= 0:
        total = 1
    ratio = max(0.0, min(value / total, 1.0))
    filled = max(1, int(round(ratio * width))) if value > 0 else 0
    return "█" * filled + "░" * (width - filled)


def _format_compact_duration(seconds: float) -> str:
    total = max(int(seconds), 0)
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes}m"
    if minutes > 0:
        return f"{minutes}m"
    return f"{secs}s"


def _format_compact_sources(sources: list[str], *, limit: int = 3) -> str:
    normalized = [str(source or "").strip() for source in sources if str(source or "").strip()]
    if not normalized:
        return "-"
    if len(normalized) > limit:
        return f"{','.join(normalized[:limit])}+{len(normalized) - limit}"
    return ",".join(normalized)


def _render_status_bar(sdk: Any, conversation_id: str, terminal_state: "TerminalSessionState") -> None:
    runtime = sdk.describe_runtime()
    state = sdk.conversation_state(conversation_id)
    summary = state["conversation"].snapshot.context_summary
    paper_count = int(summary.paper_count or 0)
    imported_count = int(summary.imported_document_count or 0)
    selected_count = len(terminal_state.selected_paper_ids)
    source_label = _format_compact_sources(list(state.get("sources") or terminal_state.sources))
    elapsed = time.monotonic() - terminal_state.session_started_monotonic
    context_limit = 128
    usage = min((paper_count * 6) + (imported_count * 8) + (selected_count * 4), context_limit)
    pct = int((usage / context_limit) * 100)
    line = (
        f"{_BLUE}⚗{_RESET} "
        f"{_WHITE}{runtime['llm']['model']}{_RESET} {_SLATE}│{_RESET} "
        f"{_WHITE}{usage:.1f}K/{context_limit}K{_RESET} {_SLATE}│{_RESET} "
        f"{_BLUE}{_progress_bar(usage, context_limit)}{_RESET} {_WHITE}{pct}%{_RESET} {_SLATE}│{_RESET} "
        f"{_WHITE}{terminal_state.turn_count} turns{_RESET} {_SLATE}│{_RESET} "
        f"{_WHITE}{_format_compact_duration(elapsed)}{_RESET} {_SLATE}│{_RESET} "
        f"{_WHITE}⏲ {terminal_state.last_round_seconds:.1f}s{_RESET} {_SLATE}│{_RESET} "
        f"{_WHITE}{summary.current_stage}{_RESET} {_SLATE}│{_RESET} "
        f"{_WHITE}src {source_label}{_RESET} {_SLATE}│{_RESET} "
        f"{_WHITE}papers {paper_count}{_RESET} {_SLATE}│{_RESET} "
        f"{_WHITE}docs {imported_count}{_RESET} {_SLATE}│{_RESET} "
        f"{_WHITE}sel {selected_count}{_RESET}"
    )
    print(_pad_visible(line, _terminal_width()))
    print(_divider())


def _normalize_table_row(line: str) -> str | None:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return None
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if not cells:
        return None
    if all(cell and set(cell) <= {"-", ":"} for cell in cells):
        return None
    return "  │  ".join(cell or " " for cell in cells)


def _normalize_render_text(text: str) -> str:
    value = str(text or "")
    if "\\n" not in value:
        return value
    # Only normalize obvious escaped line breaks from upstream serialization.
    return value.replace("\\r\\n", "\n").replace("\\n", "\n")


def _render_markdown_lines(text: str) -> list[str]:
    normalized_text = _normalize_render_text(text)
    raw_lines = [(line or "").rstrip() for line in normalized_text.splitlines()]
    rendered: list[str] = []
    in_table = False

    for raw in raw_lines:
        stripped = raw.strip()
        if not stripped:
            rendered.append("")
            in_table = False
            continue

        table_row = _normalize_table_row(raw)
        if table_row is not None:
            if not in_table:
                rendered.append(f"{_BLUE_SOFT}┄ table ┄{_RESET}")
                in_table = True
            rendered.append(f"{_WHITE}{table_row}{_RESET}")
            continue

        in_table = False

        if stripped.startswith("### "):
            rendered.append(f"{_CYAN}▸ {stripped[4:]}{_RESET}")
            continue
        if stripped.startswith("## "):
            rendered.append("")
            rendered.append(f"{_BOLD}{_BLUE_SOFT}{stripped[3:]}{_RESET}")
            continue
        if stripped.startswith("# "):
            rendered.append("")
            rendered.append(f"{_BOLD}{_BLUE}{stripped[2:]}{_RESET}")
            continue
        if stripped.startswith("- "):
            rendered.append(f"{_BLUE_SOFT}•{_RESET} {stripped[2:]}")
            continue

        marker, _, rest = stripped.partition(". ")
        if marker.isdigit() and rest:
            rendered.append(f"{_BLUE_SOFT}{marker}.{_RESET} {rest}")
            continue

        rendered.append(raw)

    return rendered or [""]


def _render_assistant_panel(text: str, *, footer: str | None = None) -> None:
    lines = _render_markdown_lines(text)
    if footer:
        lines.extend(["", f"{_SLATE}{footer}{_RESET}"])
    _boxed_lines(lines, title=f"{_BLUE}⚗ KEPLER{_RESET}", accent=_BLUE, body_color=_WHITE)


def _render_system_note(text: str) -> None:
    print(f"{_BLUE_SOFT}{text}{_RESET}")


def _latest_event_info(sdk: Any, conversation_id: str) -> tuple[str | None, str | None]:
    try:
        payload = sdk.conversation_trajectory(conversation_id, message_limit=0, event_limit=1)
    except Exception:
        return None, None
    events = payload.get("recent_events") or []
    if not events:
        return None, None
    event = events[-1]
    event_type = str(event.get("event_type") or "").strip()
    timestamp = str(event.get("timestamp") or "").strip() or None
    event_payload = event.get("payload") or {}
    runtime_event = str(event_payload.get("runtime_event") or "").strip()
    progress_stage = str(event_payload.get("stage") or "").strip()
    progress_node = str(event_payload.get("node") or "").strip()
    progress_status = str(event_payload.get("status") or "").strip()
    progress_summary = str(event_payload.get("summary") or "").strip()
    if runtime_event in {"supervisor_progress", "supervisor_heartbeat"}:
        parts = [part for part in (progress_stage, progress_node, progress_status) if part]
        label = " / ".join(parts) if parts else event_type
        if progress_summary:
            return f"latest event: {label} | {progress_summary}", timestamp
        return f"latest event: {label}", timestamp
    if event_type == "task_completed":
        paper_count = event_payload.get("paper_count")
        if paper_count is not None:
            return f"latest event: task_completed (papers={paper_count})", timestamp
    if event_type:
        return f"latest event: {event_type}", timestamp
    return None, timestamp


def _conversation_failure_message(sdk: Any, conversation_id: str) -> str | None:
    try:
        state = sdk.conversation_state(conversation_id, include_papers=False)
    except Exception:
        return None
    conversation = state.get("conversation")
    snapshot = getattr(conversation, "snapshot", None)
    status_metadata = getattr(conversation, "status_metadata", None)
    last_error = getattr(snapshot, "last_error", None)
    if isinstance(last_error, str) and last_error.strip():
        return last_error.strip()
    lifecycle_status = getattr(status_metadata, "lifecycle_status", None)
    error_message = getattr(status_metadata, "error_message", None)
    if lifecycle_status == "failed":
        return (error_message or "request failed").strip()
    return None


def _completed_conversation_payload(
    sdk: Any,
    conversation_id: str,
    *,
    prior_message_count: int,
) -> tuple[Any, list[Any]] | None:
    try:
        state = sdk.conversation_state(conversation_id, include_papers=False)
    except Exception:
        return None
    conversation = state.get("conversation")
    messages = state.get("messages") or []
    if len(messages) > prior_message_count:
        new_messages = messages[prior_message_count:]
        for message in new_messages:
            if getattr(message, "role", None) == "assistant" and getattr(message, "kind", None) != "welcome":
                return conversation, new_messages
    return None


def _conversation_messages_slice(conversation: Any, *, prior_message_count: int) -> list[Any]:
    messages = getattr(conversation, "messages", None)
    if isinstance(messages, list):
        return messages[prior_message_count:]
    wrapped = getattr(conversation, "conversation", None)
    wrapped_messages = getattr(wrapped, "messages", None)
    if isinstance(wrapped_messages, list):
        return wrapped_messages[prior_message_count:]
    return []


def _render_new_assistant_messages(
    messages: list[Any],
    *,
    footer: str | None = None,
) -> bool:
    rendered_any = False
    footer_text = footer
    for message in messages:
        if getattr(message, "role", None) != "assistant" or getattr(message, "kind", None) == "welcome":
            continue
        kind = getattr(message, "kind", None)
        content = (getattr(message, "content", None) or "").strip()
        title = (getattr(message, "title", None) or "").strip()
        meta = (getattr(message, "meta", None) or "").strip()
        if kind == "notice":
            if content and len(content) > 120:
                _render_assistant_panel(content, footer=footer_text)
                footer_text = None
                rendered_any = True
            elif content:
                _render_system_note(content)
                rendered_any = True
            continue
        if kind == "candidates":
            if meta:
                _render_system_note(meta)
                rendered_any = True
            continue
        if not content and not title:
            continue
        _render_assistant_panel(content or title, footer=footer_text)
        footer_text = None
        rendered_any = True
    return rendered_any


def _format_sources(sources: list[str]) -> str:
    return ", ".join(sources) if sources else "-"


def _summarize_titles(titles: list[str], *, limit: int = 3) -> str:
    if not titles:
        return "-"
    if len(titles) <= limit:
        return " | ".join(titles)
    head = " | ".join(titles[:limit])
    return f"{head} | +{len(titles) - limit} more"


def _find_candidate_paper(rows: list[dict[str, Any]], selector: str) -> dict[str, Any] | None:
    value = str(selector).strip()
    if not value:
        return None
    if value.isdigit():
        index = int(value)
        for row in rows:
            if row["index"] == index:
                return row
    lowered = value.lower()
    for row in rows:
        if row["paper_id"] == value:
            return row
        if lowered in (row["title"] or "").lower():
            return row
    return None


def _print_paper_detail(paper: Any, row: dict[str, Any]) -> None:
    lines = [
        f"{row['index']:>2}. {paper.title}",
        f"paper_id: {paper.paper_id}",
        f"source: {paper.source} | year: {paper.year or '-'}",
        f"authors: {', '.join(paper.authors[:8]) if paper.authors else '-'}",
        f"url: {paper.url or '-'}",
        f"pdf: {paper.pdf_url or '-'}",
        f"citations: {paper.citations if paper.citations is not None else '-'}",
    ]
    flags: list[str] = []
    if row["selected"]:
        flags.append("selected")
    if row["must_read"]:
        flags.append("must-read")
    if row["ingest_candidate"]:
        flags.append("ingest")
    if row["document_id"]:
        flags.append("imported")
    lines.append(f"flags: {', '.join(flags) if flags else '-'}")
    lines.append("")
    lines.append((paper.abstract or "no abstract").strip())
    _boxed_lines(lines, title=f"{_BLUE}📄 Paper Detail{_RESET}", accent=_BLUE_DEEP, body_color=_WHITE)


def _print_status_panel(sdk: Any, *, conversation_id: str | None = None, terminal_state: "TerminalSessionState | None" = None) -> None:
    runtime = sdk.describe_runtime()
    print("=" * 72)
    print("Research-Copilot Runtime")
    print(f"llm: {runtime['llm']['provider']} / {runtime['llm']['model']}")
    print(f"embedding: {runtime['embedding']['provider']} / {runtime['embedding']['model']}")
    enabled_plugins = [item["name"] for item in runtime["plugins"] if item["enabled"]]
    print(f"plugins: {', '.join(enabled_plugins) if enabled_plugins else 'none'}")
    if conversation_id:
        payload = sdk.conversation_trajectory(conversation_id, message_limit=3, event_limit=5)
        state = sdk.conversation_state(conversation_id)
        summary = payload["context_summary"]
        conversation = state["conversation"]
        papers = state["papers"]
        selected_ids = list(terminal_state.selected_paper_ids) if terminal_state is not None else list(state["selected_paper_ids"])
        selected_map = {paper.paper_id: paper.title for paper in papers}
        selected_titles = [selected_map[paper_id] for paper_id in selected_ids if paper_id in selected_map]
        jobs = sdk.list_jobs(conversation_id=conversation_id)
        active_jobs = [job for job in jobs if getattr(job, "status", None) in {"queued", "running"}]
        print("-" * 72)
        print(f"conversation: {conversation_id}")
        print(f"topic: {summary.get('topic') or '-'}")
        print(f"stage: {summary.get('current_stage')}")
        print(f"status: {summary.get('status_summary') or '-'}")
        print(f"sources: {_format_sources(list(state['sources']) or (terminal_state.sources if terminal_state else []))}")
        print(f"task_id: {state['task_id'] or '-'}")
        print(
            f"selected papers: {len(selected_ids)}"
            f" | must-read: {len(state['must_read_paper_ids'])}"
            f" | ingest: {len(state['ingest_candidate_ids'])}"
        )
        print(f"selected titles: {_summarize_titles(selected_titles)}")
        print(f"papers: {summary.get('paper_count')} | imported docs: {summary.get('imported_document_count')}")
        print(
            f"active job: {conversation.snapshot.active_job_id or '-'}"
            f" | running jobs: {len(active_jobs)}"
        )
        print("recent events:")
        for event in payload["recent_events"]:
            print(f"- {event['event_type']} @ {event['timestamp']}")
    print("=" * 72)


def _print_trajectory(payload: dict[str, object]) -> None:
    summary = payload["context_summary"]
    print("=" * 72)
    print("Context Summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("-" * 72)
    print("Recent Events")
    for event in payload["recent_events"]:
        print(f"- {event['event_type']} @ {event['timestamp']} :: {json.dumps(event['payload'], ensure_ascii=False)}")
    print("-" * 72)
    print("Messages")
    for message in payload["messages"]:
        title = message.get("title") or ""
        print(f"[{message['role']}/{message['kind']}] {title}")
        content = (message.get("content") or "").strip()
        if content:
            print(content)
    print("=" * 72)


def _extract_agent_command(line: str) -> tuple[str, str]:
    command, _, rest = line.strip().partition(" ")
    return command, rest.strip()


@dataclass
class TerminalSessionState:
    sources: list[str] = field(default_factory=lambda: ["arxiv", "openalex"])
    selected_paper_ids: list[str] = field(default_factory=list)
    last_figure: dict[str, Any] | None = None
    session_started_monotonic: float = field(default_factory=time.monotonic)
    turn_count: int = 0
    last_round_seconds: float = 0.0


def _refresh_terminal_state_from_conversation(sdk: Any, conversation_id: str, state: TerminalSessionState) -> None:
    payload = sdk.conversation_state(conversation_id, include_papers=False)
    state.sources = list(payload["sources"] or state.sources)
    state.selected_paper_ids = list(payload["selected_paper_ids"] or [])
    conversation = payload.get("conversation")
    workspace_metadata = {}
    if conversation is not None:
        workspace = getattr(getattr(conversation, "snapshot", None), "workspace", None)
        workspace_metadata = dict(getattr(workspace, "metadata", {}) or {})
    cached_figure = workspace_metadata.get("last_visual_anchor_figure")
    state.last_figure = cached_figure if isinstance(cached_figure, dict) else None


def _resolve_conversation_selector(sdk: Any, selector: str) -> str | None:
    value = selector.strip()
    if not value:
        return None
    if value.isdigit():
        conversations = sdk.list_conversations()
        index = int(value)
        if 1 <= index <= len(conversations):
            return conversations[index - 1].conversation_id
        return None
    return value


def _print_candidate_papers(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("当前没有候选论文。")
        return
    for row in rows:
        flags: list[str] = []
        if row["selected"]:
            flags.append("selected")
        if row["must_read"]:
            flags.append("must-read")
        if row["ingest_candidate"]:
            flags.append("ingest")
        if row["document_id"]:
            flags.append("imported")
        suffix = f" [{' | '.join(flags)}]" if flags else ""
        year = f" ({row['year']})" if row["year"] else ""
        print(f"{row['index']:>2}. {row['title']}{year} [{row['source']}] <{row['paper_id']}>{suffix}")


def _print_papers_usage() -> None:
    print("/papers 查看当前候选论文")
    print("/papers filter <keyword> 按标题筛选")
    print("/papers show <index|paper_id|title> 查看单篇详情")
    print("/papers abstract <index|paper_id|title> 查看摘要")


def _figure_from_response(response: Any, conversation: Any | None = None) -> dict[str, Any] | None:
    qa = getattr(response, "qa", None)
    qa_metadata = dict(qa.metadata) if qa is not None and isinstance(getattr(qa, "metadata", None), dict) else {}
    response_metadata = (
        dict(response.metadata)
        if isinstance(getattr(response, "metadata", None), dict)
        else {}
    )
    for metadata in (qa_metadata, response_metadata):
        figure = metadata.get("visual_anchor_figure")
        if isinstance(figure, dict):
            return figure
    chart = getattr(response, "chart", None)
    if chart is not None:
        chart_meta = getattr(chart, "metadata", None) or {}
        if isinstance(chart_meta, dict) and chart_meta.get("image_path"):
            return {
                "figure_id": getattr(chart, "id", None) or getattr(chart, "chart_id", None) or "chart",
                "image_path": str(chart_meta["image_path"]),
                "chart_type": getattr(chart, "chart_type", "unknown"),
                "summary": getattr(chart, "summary", None) or "",
            }
    allow_cached_figure = (
        str(qa_metadata.get("qa_route") or "").strip() == "chart_drilldown"
        or str(response_metadata.get("route_mode") or "").strip() == "chart_drilldown"
        or chart is not None
    )
    if not allow_cached_figure or conversation is None:
        return None
    snapshot = getattr(getattr(conversation, "conversation", None), "snapshot", None)
    workspace = getattr(snapshot, "workspace", None)
    workspace_metadata = getattr(workspace, "metadata", None)
    if isinstance(workspace_metadata, dict):
        figure = workspace_metadata.get("last_visual_anchor_figure")
        if isinstance(figure, dict):
            return figure
    return None


def _quality_hints_from_response(response: Any) -> list[str]:
    qa = getattr(response, "qa", None)
    metadata = getattr(qa, "metadata", None) if qa is not None else None
    if not isinstance(metadata, dict):
        return []
    check = metadata.get("answer_quality_check")
    if not isinstance(check, dict):
        return []
    hints: list[str] = []
    if check.get("needs_recovery"):
        hints.append("答案证据偏弱：可以继续追问，或先导入相关论文全文后再问。")
    if check.get("route") == "chart_drilldown":
        hints.append("本轮走图像问答链路；如有关联图片，可用 /figure 查看，/open-figure 打开。")
    warnings = check.get("warnings")
    if isinstance(warnings, list) and warnings:
        hints.append(f"quality warnings: {', '.join(str(item) for item in warnings[:4])}")
    return hints


def _figure_image_path(figure: dict[str, Any] | None) -> str:
    if not isinstance(figure, dict):
        return ""
    return str(figure.get("image_path") or "").strip()


def _render_figure_panel(figure: dict[str, Any] | None) -> None:
    image_path = _figure_image_path(figure)
    if not image_path:
        return
    lines = [
        f"figure_id: {figure.get('figure_id') or '-'}",
        f"chart_id: {figure.get('chart_id') or '-'}",
        f"page: {figure.get('page_number') or '-'}",
        f"image_path: {image_path}",
        "",
        "打开图片: /open-figure",
    ]
    title = f"{_CYAN}🖼  Linked Figure{_RESET}"
    _boxed_lines(lines, title=title, accent=_CYAN, body_color=_WHITE)


def _summarize_profile_values(values: list[str], *, limit: int = 6, empty: str = "(none)") -> str:
    items = [str(value or "").strip() for value in values if str(value or "").strip()]
    if not items:
        return empty
    if len(items) > limit:
        return f"{', '.join(items[:limit])} (+{len(items) - limit})"
    return ", ".join(items)


def _format_profile_timestamp(value: Any) -> str:
    if value is None:
        return "-"
    if hasattr(value, "isoformat"):
        try:
            rendered = value.isoformat(timespec="seconds")
        except TypeError:
            rendered = value.isoformat()
        return str(rendered).replace("T", " ")
    return str(value)


def _profile_summary_lines(profile: Any) -> list[str]:
    interest_topics = list(getattr(profile, "interest_topics", []) or [])
    latest_recommendation = next(iter(getattr(profile, "recommendation_history", []) or []), None)
    lines = [
        f"user_id: {getattr(profile, 'user_id', '-') or '-'}",
        f"updated_at: {_format_profile_timestamp(getattr(profile, 'updated_at', None))}",
        f"last_active_topic: {getattr(profile, 'last_active_topic', None) or '-'}",
        f"preferred_sources: {_summarize_profile_values(list(getattr(profile, 'preferred_sources', []) or []), limit=5)}",
        f"preferred_keywords: {_summarize_profile_values(list(getattr(profile, 'preferred_keywords', []) or []), limit=6)}",
    ]
    reasoning_style = getattr(profile, "preferred_reasoning_style", None)
    if reasoning_style:
        lines.append(f"reasoning_style: {reasoning_style}")
    answer_language = getattr(profile, "preferred_answer_language", None)
    if answer_language:
        lines.append(f"answer_language: {answer_language}")
    if latest_recommendation:
        lines.append(
            "last_recommendation_topics: "
            + _summarize_profile_values(list(latest_recommendation.get("topics_used", []) or []), limit=4)
        )

    if not interest_topics:
        lines.extend(
            [
                "",
                "还没有学到稳定偏好。",
                "先正常问几轮你关心的论文主题，再用 /profile 或 /preferences 查看长期画像。",
            ]
        )
        return lines

    lines.append("")
    lines.append("top interest topics:")
    for index, topic in enumerate(interest_topics[:5], start=1):
        topic_name = getattr(topic, "topic_name", None) or "-"
        weight = float(getattr(topic, "weight", 0.0) or 0.0)
        mention_count = int(getattr(topic, "mention_count", 0) or 0)
        recent_count = int(getattr(topic, "recent_mention_count", 0) or 0)
        lines.append(
            f"{index}. {topic_name} | weight={weight:.2f} | mentions={mention_count} | recent={recent_count}"
        )
        detail_parts: list[str] = []
        topic_sources = _summarize_profile_values(
            list(getattr(topic, "preferred_sources", []) or []),
            limit=3,
            empty="",
        )
        if topic_sources:
            detail_parts.append(f"sources={topic_sources}")
        topic_keywords = _summarize_profile_values(
            list(getattr(topic, "preferred_keywords", []) or []),
            limit=4,
            empty="",
        )
        if topic_keywords:
            detail_parts.append(f"keywords={topic_keywords}")
        preferred_recency_days = getattr(topic, "preferred_recency_days", None)
        if preferred_recency_days:
            detail_parts.append(f"recency={preferred_recency_days}d")
        if detail_parts:
            lines.append("   " + " | ".join(detail_parts))
    lines.extend(
        [
            "",
            "manage: /profile remove <topic> | /profile clear",
        ]
    )
    return lines


def _render_profile_panel(profile: Any) -> None:
    _boxed_lines(
        _profile_summary_lines(profile),
        title=f"{_BLUE}🧠 Preference Profile{_RESET}",
        accent=_BLUE_DEEP,
        body_color=_WHITE,
    )


def _open_local_image(path_value: str) -> tuple[bool, str]:
    path = Path(path_value).expanduser()
    if not path.exists():
        return False, f"image file not found: {path}"
    openers: list[list[str]] = []
    if shutil.which("xdg-open"):
        openers.append(["xdg-open", str(path)])
    if shutil.which("wslview"):
        openers.append(["wslview", str(path)])
    if shutil.which("explorer.exe"):
        openers.append(["explorer.exe", str(path)])
    if sys.platform == "darwin" and shutil.which("open"):
        openers.append(["open", str(path)])
    if not openers:
        return False, f"no image opener found; open manually: {path}"
    try:
        subprocess.Popen(openers[0], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        return False, f"failed to open image: {exc}"
    return True, f"opening image: {path}"


def _slash_command_entries() -> list[tuple[str, str]]:
    return [
        ("/new [topic]", "Start a new session with an optional research topic"),
        ("/clear", "Delete the current session memory and start a fresh session"),
        ("/use <conversation_id>", "Switch to an existing conversation"),
        ("/status", "Show runtime, scope, selected papers, and active task state"),
        ("/profile", "Show the persisted preference profile learned from past questions"),
        ("/profile remove <topic>", "Remove one noisy long-term interest topic from the profile"),
        ("/profile clear", "Reset the persisted preference profile"),
        ("/preferences", "Alias for /profile"),
        ("/events", "Show recent runtime events"),
        ("/trajectory", "Show compressed context, recent events, and message history"),
        ("/sources", "Show current search sources"),
        ("/sources set <sources...>", "Update search sources for upcoming research turns"),
        ("/papers", "List current candidate papers"),
        ("/papers filter <keyword>", "Filter candidate papers by title keyword"),
        ("/papers show <index|paper_id|title>", "Show detailed metadata and abstract for one paper"),
        ("/papers abstract <index|paper_id|title>", "Show only the abstract for one paper"),
        ("/select <ids|mustread|ingest>", "Select candidate papers for agent context or import"),
        ("/select clear", "Clear the current selected-paper scope"),
        ("/import selected|all|mustread|ingest", "Import papers into the research workspace"),
        ("/zotero selected [collection]", "Sync selected papers to Zotero"),
        ("/figure", "Show the latest linked figure path"),
        ("/open-figure", "Open the latest linked figure image"),
        ("/help", "Show the command reference panel"),
        ("/exit", "Exit KEPLER terminal"),
    ]


def _render_slash_command_palette(prefix: str = "/") -> None:
    prefix = prefix.strip() or "/"
    entries = _slash_command_entries()
    if prefix != "/":
        lowered = prefix.lower()
        filtered = [entry for entry in entries if entry[0].lower().startswith(lowered)]
        entries = filtered or entries
    lines = [f"{_BLUE_SOFT}{cmd:<36}{_RESET} {desc}" for cmd, desc in entries]
    _boxed_lines(
        lines,
        title=f"{_BLUE}⌘ Slash Commands{_RESET}",
        accent=_BLUE_DEEP,
        body_color=_WHITE,
    )


def _prompt_label() -> str:
    return f"{_BLUE}❯{_RESET} "


def _prompt_message() -> Any:
    try:
        from prompt_toolkit.formatted_text import ANSI
    except Exception:
        return _prompt_label()
    return ANSI(_prompt_label())


def _completion_style() -> Any | None:
    try:
        from prompt_toolkit.styles import Style
    except Exception:
        return None
    return Style.from_dict(
        {
            "completion-menu": "bg:#061a2f",
            "completion-menu.completion": "fg:#8fdcff bg:#061a2f",
            "completion-menu.completion.current": "fg:#ffffff bg:#0077d9 bold",
            "completion-menu.meta.completion": "fg:#5fb8ff bg:#06233d",
            "completion-menu.meta.completion.current": "fg:#d7f4ff bg:#0069bf",
            "completion-command": "fg:#b8ecff bold",
            "completion-meta": "fg:#5fb8ff",
            "scrollbar.background": "bg:#06233d",
            "scrollbar.button": "bg:#38bdf8",
        }
    )


def _build_slash_completer() -> Any | None:
    return None


def _build_terminal_completer(sdk: Any, conversation_id: str) -> Any | None:
    try:
        from prompt_toolkit.completion import Completer, Completion
    except Exception:
        return None

    def command_display(command: str):
        return [("class:completion-command", command)]

    def meta_display(meta: str):
        return [("class:completion-meta", meta)]

    entries = _slash_command_entries()
    command_map = {command: description for command, description in entries}
    argument_suggestions: dict[str, list[str]] = {
        "/profile": ["remove", "clear"],
        "/preferences": ["remove", "clear"],
        "/sources": ["set", "arxiv", "openalex", "semantic_scholar", "ieee", "zotero"],
        "/papers": ["filter", "show", "abstract"],
        "/paper": ["filter", "show", "abstract"],
        "/select": ["mustread", "ingest", "clear", "all", "selected"],
        "/import": ["selected", "all", "mustread", "ingest"],
        "/zotero": ["selected"],
        "/new": [],
        "/use": [],
    }

    class SlashCommandCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor.lstrip()
            if not text.startswith("/"):
                return
            lowered = text.lower()
            parts = text.split()
            if len(parts) <= 1 and not text.endswith(" "):
                for command, description in entries:
                    if lowered == "/" or command.lower().startswith(lowered):
                        yield Completion(
                            command,
                            start_position=-len(document.text_before_cursor),
                            display=command_display(command),
                            display_meta=meta_display(description),
                        )
                return

            command = parts[0]
            suggestions = list(argument_suggestions.get(command, []))
            suggestion_meta_overrides: dict[str, str] = {}
            if command == "/use":
                try:
                    conversations = sdk.list_conversations()[:12]
                    suggestions = [str(index) for index, _ in enumerate(conversations, start=1)]
                    suggestions.extend(item.conversation_id for item in conversations)
                    for index, item in enumerate(conversations, start=1):
                        topic = (item.topic or item.title or "").strip() or "untitled"
                        stage = str(getattr(item, "current_stage", "") or "").strip()
                        status = str(getattr(item, "status_summary", "") or "").strip()
                        detail_parts = [topic]
                        if stage:
                            detail_parts.append(stage)
                        if status:
                            detail_parts.append(status)
                        detail = " | ".join(detail_parts)
                        suggestion_meta_overrides[str(index)] = f"{item.conversation_id} | {detail}"
                        suggestion_meta_overrides[item.conversation_id] = detail
                except Exception:
                    suggestions = []
            elif command in {"/profile", "/preferences"} and len(parts) >= 2 and parts[1] == "remove":
                try:
                    profile = sdk.load_user_profile()
                    suggestions = [str(item.topic_name) for item in list(getattr(profile, "interest_topics", []) or [])[:12]]
                    for topic_name in suggestions:
                        suggestion_meta_overrides[topic_name] = "interest topic"
                except Exception:
                    suggestions = []
            elif command in {"/papers", "/paper", "/select"}:
                try:
                    paper_rows = sdk.list_candidate_papers(conversation_id)
                except Exception:
                    paper_rows = []
                dynamic: list[str] = []
                if command in {"/papers", "/paper"} and len(parts) >= 2 and parts[1] in {"show", "abstract"}:
                    dynamic = [str(row["index"]) for row in paper_rows[:20]]
                elif command == "/select":
                    dynamic = [str(row["index"]) for row in paper_rows[:20]]
                for row in paper_rows[:20]:
                    suggestion_meta_overrides[str(row["index"])] = row["title"]
                suggestions = suggestions + [item for item in dynamic if item not in suggestions]
            current = ""
            if not text.endswith(" "):
                current = parts[-1]
            for suggestion in suggestions:
                if current and not suggestion.lower().startswith(current.lower()):
                    continue
                meta = f"{command} option"
                if command == "/sources" and suggestion in {"arxiv", "openalex", "semantic_scholar", "ieee", "zotero"}:
                    meta = "search source"
                elif command in {"/select", "/import"}:
                    meta = "selection scope"
                elif command in {"/papers", "/paper"}:
                    meta = "paper action"
                elif command == "/use":
                    meta = "recent conversation"
                elif suggestion.isdigit():
                    meta = "paper index"
                if suggestion in suggestion_meta_overrides:
                    meta = suggestion_meta_overrides[suggestion]
                replace_len = len(current) if current else 0
                yield Completion(
                    suggestion,
                    start_position=-replace_len,
                    display=command_display(suggestion),
                    display_meta=meta_display(meta),
                )
            if command in command_map and not suggestions and len(parts) == 1:
                yield Completion(
                    command,
                    start_position=-len(document.text_before_cursor),
                    display=command_display(command),
                    display_meta=meta_display(command_map[command]),
                )

    return SlashCommandCompleter()


async def _read_terminal_input(sdk: Any, conversation_id: str) -> str:
    completer = _build_terminal_completer(sdk, conversation_id)
    if completer is None:
        return input(_prompt_label())
    try:
        from prompt_toolkit import PromptSession
    except Exception:
        return input(_prompt_label())
    session = PromptSession()
    style = _completion_style()
    return await session.prompt_async(
        _prompt_message(),
        completer=completer,
        complete_while_typing=True,
        complete_in_thread=True,
        style=style,
    )


async def _run_agent_shell(sdk: Any, args: argparse.Namespace) -> int:
    conversation_id = args.conversation_id
    terminal_state = TerminalSessionState(sources=args.source or ["arxiv", "openalex"])
    if conversation_id is None:
        conversation = sdk.create_conversation(
            topic=args.topic,
            days_back=args.days_back,
            max_papers=args.max_papers,
            sources=terminal_state.sources,
        )
        conversation_id = conversation.conversation.conversation_id
    _refresh_terminal_state_from_conversation(sdk, conversation_id, terminal_state)
    _render_kepler_banner()
    _render_welcome_panel()
    _render_system_note(f"session: {conversation_id}")
    _render_system_note(
        "commands: /help /status /profile /events /trajectory /papers /select /sources /import /zotero /figure /open-figure /new /clear /use /exit"
    )
    print(_divider())

    while True:
        _render_status_bar(sdk, conversation_id, terminal_state)
        try:
            line = (await _read_terminal_input(sdk, conversation_id)).strip()
        except KeyboardInterrupt:
            print()
            return 130
        except EOFError:
            print()
            return 0
        if not line:
            continue
        if line.startswith("/"):
            if line == "/":
                _render_slash_command_palette("/")
                continue
            command, rest = _extract_agent_command(line)
            if command == "/exit":
                return 0
            if command == "/help":
                _render_slash_command_palette("/")
                continue
            if command == "/status":
                _print_status_panel(sdk, conversation_id=conversation_id, terminal_state=terminal_state)
                continue
            if command in {"/profile", "/preferences"}:
                if not rest:
                    _render_profile_panel(sdk.load_user_profile())
                    continue
                if rest.strip().lower() in {"clear", "reset"}:
                    sdk.clear_user_profile()
                    _render_system_note("preference profile cleared")
                    _render_profile_panel(sdk.load_user_profile())
                    continue
                subcommand, _, value = rest.partition(" ")
                if subcommand == "remove":
                    topic_name = value.strip()
                    if not topic_name:
                        _render_system_note("usage: /profile remove <topic>")
                        continue
                    sdk.remove_user_profile_topics([topic_name])
                    _render_system_note(f"removed topic from profile: {topic_name}")
                    _render_profile_panel(sdk.load_user_profile())
                    continue
                _render_system_note("usage: /profile | /profile clear | /profile remove <topic>")
                continue
            if command == "/figure":
                if terminal_state.last_figure is None:
                    _render_system_note("no linked figure yet")
                    continue
                _render_figure_panel(terminal_state.last_figure)
                continue
            if command == "/open-figure":
                image_path = _figure_image_path(terminal_state.last_figure)
                if not image_path:
                    _render_system_note("no linked figure image yet")
                    continue
                ok, message = _open_local_image(image_path)
                _render_system_note(message)
                if not ok:
                    _render_figure_panel(terminal_state.last_figure)
                continue
            if command == "/events":
                payload = sdk.conversation_trajectory(conversation_id, message_limit=0, event_limit=12)
                for event in payload["recent_events"]:
                    print(f"- {event['event_type']} @ {event['timestamp']} :: {json.dumps(event['payload'], ensure_ascii=False)}")
                continue
            if command == "/trajectory":
                _print_trajectory(sdk.conversation_trajectory(conversation_id, message_limit=10, event_limit=10))
                continue
            if command == "/new":
                topic = rest or None
                conversation = sdk.create_conversation(
                    topic=topic,
                    days_back=args.days_back,
                    max_papers=args.max_papers,
                    sources=terminal_state.sources,
                )
                conversation_id = conversation.conversation.conversation_id
                terminal_state.selected_paper_ids = []
                terminal_state.last_figure = None
                terminal_state.session_started_monotonic = time.monotonic()
                terminal_state.turn_count = 0
                terminal_state.last_round_seconds = 0.0
                _refresh_terminal_state_from_conversation(sdk, conversation_id, terminal_state)
                _render_system_note(f"switched to {conversation_id}")
                continue
            if command == "/clear":
                if rest:
                    _render_system_note("usage: /clear")
                    continue
                previous_conversation_id = conversation_id
                try:
                    sdk.clear_conversation_memory(previous_conversation_id)
                except KeyError:
                    _render_system_note(f"conversation not found: {previous_conversation_id}")
                    continue
                conversation = sdk.create_conversation(
                    days_back=args.days_back,
                    max_papers=args.max_papers,
                    sources=terminal_state.sources,
                )
                conversation_id = conversation.conversation.conversation_id
                terminal_state.selected_paper_ids = []
                terminal_state.last_figure = None
                terminal_state.session_started_monotonic = time.monotonic()
                terminal_state.turn_count = 0
                terminal_state.last_round_seconds = 0.0
                _refresh_terminal_state_from_conversation(sdk, conversation_id, terminal_state)
                _render_system_note(f"cleared {previous_conversation_id} and switched to {conversation_id}")
                continue
            if command == "/use":
                if not rest:
                    print("conversation_id is required")
                    continue
                resolved_conversation_id = _resolve_conversation_selector(sdk, rest)
                if resolved_conversation_id is None:
                    _render_system_note(f"conversation not found: {rest}")
                    _render_system_note("Use /use <conversation_id> or /use <number from completion list>.")
                    continue
                try:
                    sdk.get_conversation(resolved_conversation_id)
                except KeyError:
                    _render_system_note(f"conversation not found: {rest}")
                    _render_system_note("Use /use <conversation_id> or /use <number from completion list>.")
                    continue
                conversation_id = resolved_conversation_id
                terminal_state.selected_paper_ids = []
                terminal_state.last_figure = None
                terminal_state.session_started_monotonic = time.monotonic()
                terminal_state.turn_count = 0
                terminal_state.last_round_seconds = 0.0
                _refresh_terminal_state_from_conversation(sdk, conversation_id, terminal_state)
                _render_system_note(f"switched to {conversation_id}")
                continue
            if command == "/sources":
                if not rest:
                    _render_system_note(f"sources: {', '.join(terminal_state.sources)}")
                    continue
                subcommand, _, value = rest.partition(" ")
                if subcommand == "set":
                    next_sources = [item.strip() for item in value.split() if item.strip()]
                    if not next_sources:
                        _render_system_note("usage: /sources set arxiv openalex semantic_scholar ieee zotero")
                        continue
                    terminal_state.sources = next_sources
                    _render_system_note(f"sources updated: {', '.join(terminal_state.sources)}")
                    continue
                _render_system_note("usage: /sources or /sources set ...")
                continue
            if command in {"/papers", "/paper"}:
                rows = sdk.list_candidate_papers(conversation_id)
                if not rest:
                    _print_candidate_papers(rows)
                    continue
                subcommand, _, value = rest.partition(" ")
                value = value.strip()
                if subcommand == "filter":
                    if not value:
                        _render_system_note("usage: /papers filter <keyword>")
                        continue
                    keyword = value.lower()
                    filtered = [row for row in rows if keyword in (row["title"] or "").lower()]
                    _render_system_note(f"matched {len(filtered)} papers")
                    _print_candidate_papers(filtered)
                    continue
                if subcommand == "show":
                    if not value:
                        _render_system_note("usage: /papers show <index|paper_id|title>")
                        continue
                    row = _find_candidate_paper(rows, value)
                    if row is None:
                        _render_system_note("paper not found")
                        continue
                    paper = sdk.conversation_state(conversation_id)["papers"][row["index"] - 1]
                    _print_paper_detail(paper, row)
                    continue
                if subcommand == "abstract":
                    if not value:
                        _render_system_note("usage: /papers abstract <index|paper_id|title>")
                        continue
                    row = _find_candidate_paper(rows, value)
                    if row is None:
                        _render_system_note("paper not found")
                        continue
                    paper = sdk.conversation_state(conversation_id)["papers"][row["index"] - 1]
                    _boxed_lines(
                        [f"{row['index']:>2}. {paper.title}", "", (paper.abstract or "no abstract").strip()],
                        title=f"{_BLUE}📘 Abstract{_RESET}",
                        accent=_BLUE_DEEP,
                        body_color=_WHITE,
                    )
                    continue
                _print_papers_usage()
                continue
            if command == "/select":
                if not rest:
                    _render_system_note(f"selected: {', '.join(terminal_state.selected_paper_ids) or '(empty)'}")
                    continue
                if rest.strip().lower() == "clear":
                    terminal_state.selected_paper_ids = []
                    _render_system_note("selection cleared")
                    continue
                resolved = sdk.resolve_paper_selection(conversation_id, rest.split())
                terminal_state.selected_paper_ids = resolved
                _render_system_note(f"selected: {', '.join(resolved) or '(empty)'}")
                continue
            if command == "/import":
                selectors = rest.split() if rest else ["selected"]
                paper_ids = sdk.resolve_paper_selection(conversation_id, selectors)
                if not paper_ids:
                    _render_system_note("no papers selected for import")
                    continue
                _render_system_note("importing...")
                try:
                    result = await asyncio.wait_for(
                        sdk.import_papers_for_conversation(
                            conversation_id=conversation_id,
                            paper_ids=paper_ids,
                        ),
                        timeout=300,
                    )
                except Exception as exc:
                    _render_system_note(f"import failed: {exc}")
                    continue
                _render_assistant_panel(
                    "\n".join(
                        [
                            f"import completed: imported={result.imported_count} skipped={result.skipped_count} failed={result.failed_count}",
                            *[f"- {item.title} :: {item.status}" for item in result.results[:8]],
                        ]
                    )
                )
                continue
            if command == "/zotero":
                parts = rest.split()
                selectors = [parts[0]] if parts else ["selected"]
                collection_name = " ".join(parts[1:]).strip() or None if parts else None
                paper_ids = sdk.resolve_paper_selection(conversation_id, selectors)
                if not paper_ids:
                    _render_system_note("no papers selected for zotero sync")
                    continue
                _render_system_note("syncing to zotero...")
                try:
                    results = await asyncio.wait_for(
                        sdk.sync_papers_to_zotero(
                            conversation_id=conversation_id,
                            paper_ids=paper_ids,
                            collection_name=collection_name,
                        ),
                        timeout=300,
                    )
                except Exception as exc:
                    _render_system_note(f"zotero sync failed: {exc}")
                    continue
                result_lines: list[str] = []
                for item in results:
                    result_lines.append(f"- {item['title']} :: {item['status']} ({item['action']})")
                    for warning in item.get("warnings", [])[:2]:
                        result_lines.append(f"  warning: {warning}")
                _render_assistant_panel("\n".join(result_lines) if result_lines else "zotero sync completed")
                continue
            _render_system_note(f"unknown command: {command}")
            _render_slash_command_palette(command)
            continue

        _render_system_note("processing...")
        round_started_at = time.monotonic()
        prior_message_count = 0
        try:
            conversation_state = sdk.conversation_state(conversation_id, include_papers=False)
            prior_message_count = len(conversation_state.get("messages") or [])
        except Exception:
            prior_message_count = 0
        agent_task = asyncio.create_task(
            sdk.run_agent_message(
                message=line,
                conversation_id=conversation_id,
                topic=args.topic,
                mode=args.mode,
                days_back=args.days_back,
                max_papers=args.max_papers,
                sources=terminal_state.sources,
                selected_paper_ids=terminal_state.selected_paper_ids,
            )
        )
        poll_interval_seconds = max(0.5, float(sdk.settings.research_cli_poll_initial_seconds))
        next_heartbeat_at = max(1.0, float(sdk.settings.research_cli_heartbeat_seconds))
        last_event_timestamp: str | None = None
        last_event_change_at = round_started_at
        completed_conversation = None
        completed_messages: list[Any] = []
        try:
            while True:
                elapsed = time.monotonic() - round_started_at
                remaining = 300 - elapsed
                if remaining <= 0:
                    agent_task.cancel()
                    raise TimeoutError
                try:
                    response, conversation, prior_message_count = await asyncio.wait_for(
                        asyncio.shield(agent_task),
                        timeout=min(poll_interval_seconds, remaining),
                    )
                    break
                except TimeoutError:
                    elapsed = time.monotonic() - round_started_at
                    poll_interval_seconds = max(
                        poll_interval_seconds,
                        float(sdk.settings.research_cli_poll_steady_seconds),
                    )
                    failure_message = _conversation_failure_message(sdk, conversation_id)
                    if failure_message:
                        agent_task.cancel()
                        raise RuntimeError(failure_message)
                    completed_payload = _completed_conversation_payload(
                        sdk,
                        conversation_id,
                        prior_message_count=prior_message_count,
                    )
                    if completed_payload is not None:
                        completed_conversation, completed_messages = completed_payload
                        agent_task.cancel()
                        response = None
                        conversation = completed_conversation
                        break
                    event_summary, event_timestamp = _latest_event_info(sdk, conversation_id)
                    if event_timestamp and event_timestamp != last_event_timestamp:
                        last_event_timestamp = event_timestamp
                        last_event_change_at = time.monotonic()
                    stall_timeout_seconds = 24.0
                    if event_summary and "latest event: tool_called" in event_summary:
                        stall_timeout_seconds = 90.0
                    if (
                        last_event_timestamp is not None
                        and time.monotonic() - last_event_change_at >= stall_timeout_seconds
                    ):
                        agent_task.cancel()
                        raise RuntimeError(
                            "request stalled after the latest runtime event stopped advancing"
                        )
                    if elapsed >= next_heartbeat_at:
                        note = f"processing... {int(elapsed)}s"
                        if event_summary:
                            note = f"{note} | {event_summary}"
                        _render_system_note(note)
                        next_heartbeat_at += 8.0
        except TimeoutError:
            terminal_state.last_round_seconds = time.monotonic() - round_started_at
            _render_system_note("request timed out after 300s")
            continue
        except Exception as exc:
            terminal_state.last_round_seconds = time.monotonic() - round_started_at
            _render_system_note(f"request failed: {exc}")
            continue
        terminal_state.last_round_seconds = time.monotonic() - round_started_at
        terminal_state.turn_count += 1
        latest_figure = _figure_from_response(response, conversation)
        if latest_figure is not None:
            terminal_state.last_figure = latest_figure
        footer = None
        new_messages: list[Any] = []
        if conversation is not None:
            new_messages = completed_messages or _conversation_messages_slice(
                conversation,
                prior_message_count=prior_message_count,
            )
            _refresh_terminal_state_from_conversation(sdk, conversation_id, terminal_state)
            snapshot_holder = getattr(conversation, "conversation", conversation)
            summary = snapshot_holder.snapshot.context_summary
            footer = f"stage={summary.current_stage} | papers={summary.paper_count} | imported_docs={summary.imported_document_count}"
            if summary.status_summary:
                footer = f"{footer} | {summary.status_summary}"
        rendered = _render_new_assistant_messages(new_messages, footer=footer)
        if not rendered:
            if response is None:
                assistant_text = "research task completed"
            else:
                assistant_text = sdk.latest_assistant_message(
                    conversation,
                    response,
                    prior_message_count=prior_message_count,
                )
            _render_assistant_panel(assistant_text, footer=footer)
        _render_figure_panel(terminal_state.last_figure if latest_figure is not None else None)
        for hint in _quality_hints_from_response(response):
            _render_system_note(hint)


def main() -> int:
    from sdk.client import ResearchCopilotSDK

    args = build_parser().parse_args()
    sdk = ResearchCopilotSDK.from_settings(Settings())

    if args.command == "list-conversations":
        items = sdk.list_conversations()
        _print_json([item.model_dump(mode="json") for item in items])
        return 0

    if args.command == "show-profile":
        profile = sdk.load_user_profile()
        print(profile.model_dump_json(indent=2))
        return 0

    if args.command == "update-profile":
        profile = sdk.update_user_profile(
            topic=args.topic,
            sources=args.source,
            keywords=args.keyword,
            reasoning_style=args.reasoning_style,
            note=args.note,
        )
        print(profile.model_dump_json(indent=2))
        return 0

    if args.command == "doctor":
        payload = sdk.doctor()
        _print_json(payload)
        return 0 if payload["all_required_passed"] else 1

    if args.command == "status":
        _print_status_panel(sdk, conversation_id=args.conversation_id)
        return 0

    if args.command == "trajectory":
        _print_trajectory(
            sdk.conversation_trajectory(
                args.conversation_id,
                message_limit=args.messages,
                event_limit=args.events,
            )
        )
        return 0

    if args.command == "models":
        if args.models_command == "show":
            _print_json(sdk.describe_runtime())
            return 0
        if args.models_command == "set":
            payload = sdk.update_model_profile(
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
                embedding_provider=args.embedding_provider,
                embedding_model=args.embedding_model,
                chart_vision_provider=args.chart_vision_provider,
                chart_vision_model=args.chart_vision_model,
            )
            _print_json(payload)
            return 0

    if args.command == "plugins":
        if args.plugins_command == "list":
            _print_json(sdk.list_plugins())
            return 0
        if args.plugins_command == "enable":
            _print_json(sdk.set_plugin_enabled(args.name, True))
            return 0
        if args.plugins_command == "disable":
            _print_json(sdk.set_plugin_enabled(args.name, False))
            return 0

    if args.command == "agent":
        try:
            return asyncio.run(_run_agent_shell(sdk, args))
        except KeyboardInterrupt:
            print()
            return 130

    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
