from __future__ import annotations

import json
from pathlib import Path

from core.utils import now_iso as _now_iso
from pydantic import BaseModel, Field


class RuntimeModelProfile(BaseModel):
    llm_provider: str | None = None
    llm_model: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    chart_vision_provider: str | None = None
    chart_vision_model: str | None = None


class PluginProfileState(BaseModel):
    enabled: bool = True


class RuntimeProfile(BaseModel):
    models: RuntimeModelProfile = Field(default_factory=RuntimeModelProfile)
    plugins: dict[str, PluginProfileState] = Field(default_factory=dict)
    updated_at: str | None = None


class RuntimeProfileStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> RuntimeProfile:
        if not self.path.exists():
            return RuntimeProfile()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return RuntimeProfile.model_validate(payload)

    def save(self, profile: RuntimeProfile) -> RuntimeProfile:
        updated = profile.model_copy(update={"updated_at": _now_iso()})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(updated.model_dump_json(indent=2), encoding="utf-8")
        return updated

    def update_models(self, **updates: str | None) -> RuntimeProfile:
        profile = self.load()
        next_models = profile.models.model_copy(
            update={key: value for key, value in updates.items() if key in RuntimeModelProfile.model_fields}
        )
        return self.save(profile.model_copy(update={"models": next_models}))

    def clear_models(self) -> RuntimeProfile:
        profile = self.load()
        return self.save(profile.model_copy(update={"models": RuntimeModelProfile()}))

    def set_plugin_enabled(self, name: str, enabled: bool) -> RuntimeProfile:
        profile = self.load()
        next_plugins = dict(profile.plugins)
        next_plugins[name] = PluginProfileState(enabled=enabled)
        return self.save(profile.model_copy(update={"plugins": next_plugins}))

