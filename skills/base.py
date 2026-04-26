from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SkillPromptSet(BaseModel):
    answer_prompt_path: str | None = None
    rewrite_prompt_path: str | None = None
    extra_prompts: dict[str, str] = Field(default_factory=dict)


class SkillRetrievalPolicy(BaseModel):
    mode: Literal["vector", "graph", "hybrid"] = "hybrid"
    top_k: int | None = Field(default=None, ge=1, le=100)
    graph_query_mode: Literal["entity", "subgraph", "summary", "auto"] | None = None
    enable_graph_summary: bool | None = None
    filters: dict[str, Any] = Field(default_factory=dict)


class SkillMemoryPolicy(BaseModel):
    use_session_context: bool = True
    use_task_context: bool = True
    use_preference_context: bool = True
    use_retrieval_cache_summary: bool = True
    include_memory_hints: bool = True


class SkillOutputStyle(BaseModel):
    language: str | None = None
    detail_level: Literal["brief", "normal", "detailed"] = "normal"
    tone: str | None = None
    response_schema: str | None = None


class SkillSpec(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    applicable_tasks: list[str] = Field(default_factory=lambda: ["ask_document"])
    prompt_set: SkillPromptSet = Field(default_factory=SkillPromptSet)
    preferred_tools: list[str] = Field(default_factory=list)
    retrieval_policy: SkillRetrievalPolicy = Field(default_factory=SkillRetrievalPolicy)
    memory_policy: SkillMemoryPolicy = Field(default_factory=SkillMemoryPolicy)
    output_style: SkillOutputStyle = Field(default_factory=SkillOutputStyle)
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillContext(BaseModel):
    name: str
    description: str
    task_type: str
    prompt_set: dict[str, Any] = Field(default_factory=dict)
    preferred_tools: list[str] = Field(default_factory=list)
    retrieval_policy: dict[str, Any] = Field(default_factory=dict)
    memory_policy: dict[str, Any] = Field(default_factory=dict)
    output_style: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_default_skill() -> SkillSpec:
    return SkillSpec(
        name="default",
        description="Default skill that preserves current system behavior.",
        applicable_tasks=[
            "parse_document",
            "index_document",
            "ask_document",
            "understand_chart",
            "function_call",
        ],
        prompt_set=SkillPromptSet(
            answer_prompt_path="prompts/document/answer_question_with_hybrid_rag.txt",
            rewrite_prompt_path="prompts/retrieval/rewrite_query.txt",
        ),
        preferred_tools=[
            "parse_document",
            "understand_chart",
            "hybrid_retrieve",
            "query_graph_summary",
            "answer_with_evidence",
        ],
        retrieval_policy=SkillRetrievalPolicy(
            mode="hybrid",
            top_k=10,
            graph_query_mode="auto",
            enable_graph_summary=True,
        ),
        memory_policy=SkillMemoryPolicy(),
        output_style=SkillOutputStyle(language="zh-CN", detail_level="normal", tone="factual"),
        enabled=True,
    )
