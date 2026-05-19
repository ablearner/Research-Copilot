from __future__ import annotations

import asyncio
import json

import pytest

from adapters.llm.base import BaseLLMAdapter, LLMAdapterError
from agents.general_answer_agent import GeneralAnswerAgent


class SlowGeneralAnswerLLMStub(BaseLLMAdapter):
    async def _generate_structured(self, prompt: str, input_data: dict, response_model: type):
        await asyncio.sleep(1)
        return response_model.model_validate(
            {
                "answer": "late answer",
                "confidence": 0.9,
                "key_points": [],
                "answer_type": "general",
                "warnings": [],
            }
        )

    async def _analyze_image_structured(self, prompt: str, image_path: str, response_model: type):
        raise NotImplementedError

    async def _analyze_pdf_structured(self, prompt: str, file_path: str, response_model: type):
        raise NotImplementedError

    async def _extract_graph_triples(self, prompt: str, input_data: dict, response_model: type):
        raise NotImplementedError


class FailingGeneralAnswerLLMStub(SlowGeneralAnswerLLMStub):
    async def _generate_structured(self, prompt: str, input_data: dict, response_model: type):
        raise LLMAdapterError("relay unavailable")


class StreamingJsonGeneralAnswerLLMStub(SlowGeneralAnswerLLMStub):
    async def _generate_streaming(self, prompt: str, input_data: dict, on_token):
        payload = {
            "answer_type": "direct",
            "warnings": [],
            "answer": "我可以帮你做很多事。",
            "confidence": 0.82,
            "key_points": ["回答问题", "总结资料"],
        }
        text = json.dumps(payload, ensure_ascii=False)
        await on_token(text)
        return text


@pytest.mark.asyncio
async def test_general_answer_agent_returns_provider_timeout_fallback() -> None:
    agent = GeneralAnswerAgent(
        llm_adapter=SlowGeneralAnswerLLMStub(),
        llm_timeout_seconds=0.01,
    )

    result = await agent.answer(question="你好")

    assert result.answer_type == "provider_timeout"
    assert result.warnings == ["llm_timeout"]
    assert result.confidence >= 0.45


@pytest.mark.asyncio
async def test_general_answer_agent_returns_provider_error_fallback() -> None:
    agent = GeneralAnswerAgent(llm_adapter=FailingGeneralAnswerLLMStub())

    result = await agent.answer(question="你是什么")

    assert result.answer_type == "provider_error"
    assert result.warnings == ["llm_provider_error"]
    assert result.confidence >= 0.45


@pytest.mark.asyncio
async def test_general_answer_agent_extracts_answer_from_streaming_json() -> None:
    agent = GeneralAnswerAgent(llm_adapter=StreamingJsonGeneralAnswerLLMStub())
    streamed: list[str] = []

    async def on_token(text: str) -> None:
        streamed.append(text)

    result = await agent.answer(question="你能做什么", on_token=on_token)

    assert streamed
    assert result.answer == "我可以帮你做很多事。"
    assert result.answer_type == "direct"
    assert result.warnings == []
    assert result.confidence == 0.82
    assert result.key_points == ["回答问题", "总结资料"]
