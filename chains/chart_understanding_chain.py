from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from adapters.llm.langchain_binding import ensure_provider_binding
from domain.schemas.chart import ChartSchema


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ChartUnderstandingChain:
    def __init__(
        self,
        llm: Any,
        prompt_path: str | Path = "prompts/chart/parse_chart.txt",
    ) -> None:
        self.llm = ensure_provider_binding(llm)
        resolved = Path(prompt_path)
        if not resolved.is_absolute():
            resolved = _PROJECT_ROOT / resolved
        self.prompt_path = resolved

    async def ainvoke(
        self,
        *,
        image_path: str,
        document_id: str,
        page_id: str,
        page_number: int,
        chart_id: str,
        context: dict[str, Any] | None = None,
    ) -> ChartSchema:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompt_path.read_text(encoding="utf-8")),
                ("human", "{payload}"),
            ]
        )
        payload = {
            "document_id": document_id,
            "page_id": page_id,
            "page_number": page_number,
            "chart_id": chart_id,
            "context": self._compact_context(context or {}),
        }
        messages = await prompt.ainvoke({"payload": payload})
        response = await self.llm.ainvoke_image_structured(
            messages=messages.to_messages(),
            image_path=image_path,
            response_model=ChartSchema,
            metadata={"document_id": document_id, "page_id": page_id, "chart_id": chart_id},
        )
        chart = ChartSchema.model_validate(response)
        return chart.model_copy(
            update={
                "id": chart.id or chart_id,
                "document_id": chart.document_id or document_id,
                "page_id": chart.page_id or page_id,
                "page_number": chart.page_number or page_number,
                "metadata": {
                    **chart.metadata,
                    "image_path": image_path,
                    "context": self._compact_context(context or {}),
                    "parsed_by": "ChartUnderstandingChain",
                },
            }
        )

    def _compact_context(self, context: dict[str, Any]) -> dict[str, Any]:
        allowed_keys = {"document_id", "page_id", "page_number", "chart_id", "user_context"}
        return {key: value for key, value in context.items() if key in allowed_keys}
