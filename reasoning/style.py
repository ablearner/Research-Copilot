from __future__ import annotations


DEFAULT_AGENT_REASONING_STYLE = "cot"


def normalize_reasoning_style(style: str | None) -> str:
    normalized = (style or DEFAULT_AGENT_REASONING_STYLE).strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return DEFAULT_AGENT_REASONING_STYLE
    aliases = {
        "chain_of_thought": "cot",
        "chainofthought": "cot",
        "cot": "cot",
        "planandsolve": "plan_and_solve",
        "plan_and_solve": "plan_and_solve",
        "react": "react",
        "auto": "auto",
    }
    return aliases.get(normalized, normalized)
