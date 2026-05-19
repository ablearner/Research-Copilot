import pytest

from tools.research.visual_intent import VisualIntentRoutingTool


class VisualIntentLLMStub:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    async def generate_structured(self, prompt: str, input_data: dict, response_model):
        self.calls.append({"prompt": prompt, "input_data": dict(input_data)})
        return response_model.model_validate(self.payload)


@pytest.mark.asyncio
async def test_visual_intent_llm_overrides_marker_hints_for_new_search() -> None:
    router = VisualIntentRoutingTool(
        llm_adapter=VisualIntentLLMStub(
            {
                "intent": "new_visual_search",
                "reuse_current_anchor": False,
                "search_new_figure": True,
                "target_description": "实验结果直方图",
                "exclude_figure_ids": [],
                "confidence": 0.91,
                "rationale": "The user asks to provide a result histogram, not to inspect the current image.",
            }
        )
    )

    decision = await router.decide_async(
        question="给我提供实验结果直方图，并分析",
        current_visual_anchor={"figure_id": "paper-x:old", "image_path": "/tmp/old.png"},
        current_figure={"figure_id": "paper-x:old", "caption": "qualitative examples"},
    )

    assert decision.intent == "new_visual_search"
    assert decision.search_new_figure is True
    assert decision.reuse_current_anchor is False
    assert decision.exclude_figure_ids == ["paper-x:old"]
    assert router.llm_adapter.calls[0]["input_data"]["marker_signals"]["new_visual_search"]


@pytest.mark.asyncio
async def test_visual_intent_fallback_reuses_current_anchor_only_for_current_reference() -> None:
    router = VisualIntentRoutingTool()

    decision = await router.decide_async(
        question="这张图里横轴表示什么？",
        current_visual_anchor={"figure_id": "paper-x:fig-2", "image_path": "/tmp/fig-2.png"},
    )

    assert decision.intent == "current_visual_followup"
    assert decision.reuse_current_anchor is True
    assert decision.search_new_figure is False
