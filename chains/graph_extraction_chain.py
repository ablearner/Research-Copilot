from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from adapters.llm.langchain_binding import ensure_provider_binding
from domain.schemas.chart import ChartSchema
from domain.schemas.document import TextBlock
from domain.schemas.graph import GraphExtractionResult


class GraphExtractionChain:
    def __init__(
        self,
        llm: Any,
        prompt_path: str | Path = "prompts/graph/extract_triples.txt",
    ) -> None:
        self.llm = ensure_provider_binding(llm)
        self.prompt_path = Path(prompt_path)

    async def ainvoke_from_text_blocks(
        self,
        *,
        document_id: str,
        text_blocks: list[TextBlock],
        page_summaries: list[dict[str, Any]] | None = None,
    ) -> GraphExtractionResult:
        return await self._ainvoke(
            payload={
                "document_id": document_id,
                "source_kind": "document_text",
                "instructions": {
                    "do_not_infer_missing_facts": True,
                    "allow_low_confidence_for_uncertain_relations": True,
                    "use_source_references_from_input": True,
                },
                "text_blocks": [block.model_dump(mode="json") for block in text_blocks],
                "page_summaries": page_summaries or [],
            },
            metadata={"document_id": document_id, "source": "text_blocks"},
        )

    async def ainvoke_from_chart(
        self,
        *,
        chart: ChartSchema,
        chart_summary: str | None = None,
    ) -> GraphExtractionResult:
        return await self._ainvoke(
            payload={
                "document_id": chart.document_id,
                "source_kind": "chart",
                "instructions": {
                    "do_not_infer_missing_facts": True,
                    "allow_low_confidence_for_uncertain_relations": True,
                    "use_source_references_from_input": True,
                },
                "chart": chart.model_dump(mode="json"),
                "chart_summary": chart_summary or chart.summary or "",
            },
            metadata={"document_id": chart.document_id, "source": "chart", "chart_id": chart.id},
        )

    async def _ainvoke(
        self,
        *,
        payload: dict[str, Any],
        metadata: dict[str, Any],
    ) -> GraphExtractionResult:
        binding_metadata = {
            **metadata,
            "_input_data_override": payload,
        }
        response = await self.llm.ainvoke_graph_structured(
            messages=[
                SystemMessage(content=self.prompt_path.read_text(encoding="utf-8")),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ],
            response_model=GraphExtractionResult,
            metadata=binding_metadata,
        )
        return GraphExtractionResult.model_validate(response)
