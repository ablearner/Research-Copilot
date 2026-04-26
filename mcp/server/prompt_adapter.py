from __future__ import annotations

from core.prompt_resolver import PromptResolver
from mcp.mapping import map_prompt_to_mcp_prompt
from mcp.schemas import MCPPromptContent, MCPPromptSpec


class MCPPromptAdapter:
    def __init__(self, prompt_resolver: PromptResolver | None = None) -> None:
        self.prompt_resolver = prompt_resolver or PromptResolver()

    def list_prompts(self, skill_name: str | None = None) -> list[MCPPromptSpec]:
        mapping = self.prompt_resolver.load_mapping()
        skills = mapping.get("skills") if isinstance(mapping.get("skills"), dict) else {}
        target_skills = [skill_name] if skill_name else ["default", *sorted(skills.keys())]
        seen: set[tuple[str, str]] = set()
        prompts: list[MCPPromptSpec] = []

        for target_skill in target_skills:
            prompt_set = self.prompt_resolver.resolve_prompt_set(skill_name=target_skill)
            for prompt_key, prompt_path in prompt_set.items():
                marker = (target_skill or "default", prompt_key)
                if marker in seen:
                    continue
                seen.add(marker)
                prompts.append(
                    map_prompt_to_mcp_prompt(
                        name=f"{target_skill or 'default'}:{prompt_key}",
                        prompt_key=prompt_key,
                        path=prompt_path,
                        skill_name=target_skill,
                        metadata={"resolved_with_fallback": True},
                    )
                )
        return prompts

    def get_prompt(
        self,
        *,
        prompt_name: str | None = None,
        prompt_key: str | None = None,
        skill_name: str | None = None,
        prompt_path: str | None = None,
    ) -> MCPPromptContent:
        resolved_skill = skill_name
        resolved_key = prompt_key

        if prompt_name and ":" in prompt_name and not resolved_key:
            parsed_skill, parsed_key = prompt_name.split(":", 1)
            resolved_skill = parsed_skill
            resolved_key = parsed_key

        if not resolved_key:
            resolved_key = "answer_prompt_path"

        path = self.prompt_resolver.resolve_prompt_path(
            prompt_key=resolved_key,
            skill_name=resolved_skill,
            explicit_prompt_path=prompt_path,
        )
        content = path.read_text(encoding="utf-8") if path.exists() else ""
        return MCPPromptContent(
            name=prompt_name or f"{resolved_skill or 'default'}:{resolved_key}",
            prompt_key=resolved_key,
            path=str(path),
            content=content,
            skill_name=resolved_skill,
            metadata={"exists": path.exists()},
        )
