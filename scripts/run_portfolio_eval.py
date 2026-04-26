from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domain.schemas.api import QAResponse  # noqa: E402
from domain.schemas.chart import ChartSchema  # noqa: E402
from domain.schemas.document import DocumentPage, ParsedDocument, TextBlock  # noqa: E402
from domain.schemas.evidence import Evidence, EvidenceBundle  # noqa: E402
from domain.schemas.graph import GraphExtractionResult  # noqa: E402
from domain.schemas.retrieval import HybridRetrievalResult, RetrievalHit, RetrievalQuery  # noqa: E402
from rag_runtime.runtime import RagRuntime  # noqa: E402
from rag_runtime.schemas import ChartUnderstandingResult, DocumentIndexResult  # noqa: E402
from rag_runtime.services.embedding_index_service import EmbeddingIndexResult  # noqa: E402
from rag_runtime.services.graph_index_service import GraphIndexStats  # noqa: E402


class EvalDocumentAgent:
    async def parse_document(self, file_path: str, document_id: str | None = None) -> ParsedDocument:
        raise RuntimeError("Portfolio eval uses prebuilt parsed_document fixtures for deterministic evaluation")

    async def summarize_page(self, page: DocumentPage):
        from tools.document_toolkit import PageSummary

        joined = " ".join(block.text for block in page.text_blocks)
        return PageSummary(document_id=page.document_id, page_id=page.id, page_number=page.page_number, summary=joined[:240])


class EvalChartAgent:
    def __init__(self, chart: ChartSchema) -> None:
        self.chart = chart

    async def parse_chart(self, image_path: str, document_id: str, page_id: str, page_number: int, chart_id: str, context=None) -> ChartSchema:
        return self.chart

    def to_graph_text(self, chart: ChartSchema) -> str:
        return f"{chart.title}: {chart.summary}"

    async def ask_chart(self, image_path: str, question: str, context=None, history=None) -> str:
        return self.chart.summary or f"{self.chart.title}: chart evidence available"


class EvalGraphExtractionAgent:
    async def extract_from_text_blocks(self, document_id: str, text_blocks: list[TextBlock], page_summaries=None) -> GraphExtractionResult:
        return GraphExtractionResult(document_id=document_id, status="succeeded")

    async def extract_from_chart(self, chart: ChartSchema, chart_summary: str | None = None) -> GraphExtractionResult:
        return GraphExtractionResult(document_id=chart.document_id, status="succeeded")


class EvalRetrievalAgent:
    def __init__(self, parsed_document: ParsedDocument, chart: ChartSchema) -> None:
        self.parsed_document = parsed_document
        self.chart = chart

    async def retrieve(
        self,
        question: str,
        doc_id=None,
        document_ids=None,
        top_k=10,
        filters=None,
        session_id=None,
        task_id=None,
        memory_hints=None,
        skill_context=None,
    ):
        snippets = [block.text for page in self.parsed_document.pages for block in page.text_blocks]
        hits: list[RetrievalHit] = []
        evidences: list[Evidence] = []
        for index, snippet in enumerate(snippets, start=1):
            evidence = Evidence(
                id=f"ev-{index}",
                document_id=self.parsed_document.id,
                page_id="page-1",
                page_number=1,
                source_type="text_block",
                source_id=f"tb-{index}",
                snippet=snippet,
                score=0.8,
                metadata={},
            )
            evidences.append(evidence)
            hits.append(
                RetrievalHit(
                    id=f"hit-{index}",
                    source_type="text_block",
                    source_id=f"tb-{index}",
                    document_id=self.parsed_document.id,
                    content=snippet,
                    merged_score=0.85,
                    vector_score=0.82,
                    evidence=EvidenceBundle(evidences=[evidence], metadata={}),
                    metadata={},
                )
            )
        chart_evidence = Evidence(
            id="ev-chart",
            document_id=self.chart.document_id,
            page_id=self.chart.page_id,
            page_number=self.chart.page_number,
            source_type="chart",
            source_id=self.chart.id,
            snippet=self.chart.summary,
            score=0.79,
            metadata={},
        )
        hits.append(
            RetrievalHit(
                id="hit-chart",
                source_type="chart",
                source_id=self.chart.id,
                document_id=self.chart.document_id,
                content=self.chart.summary,
                graph_score=0.78,
                merged_score=0.8,
                evidence=EvidenceBundle(evidences=[chart_evidence], metadata={}),
                metadata={},
            )
        )
        bundle = EvidenceBundle(evidences=[*evidences, chart_evidence], metadata={})
        result = HybridRetrievalResult(
            query=RetrievalQuery(query=question, document_ids=document_ids or [self.parsed_document.id], mode="hybrid", top_k=top_k, filters=filters or {}),
            hits=hits,
            evidence_bundle=bundle,
            metadata={
                "graph_hit_count": 1,
                "vector_hit_count": len(snippets),
                "summary_hit_count": 1,
                "cache_hit": False,
            },
        )
        from tools.retrieval_toolkit import RetrievalAgentResult

        return RetrievalAgentResult(
            question=question,
            document_ids=document_ids or [self.parsed_document.id],
            evidence_bundle=bundle,
            retrieval_result=result,
            metadata={"evaluation_mode": True},
        )


class EvalAnswerAgent:
    async def answer(self, question: str, evidence_bundle: EvidenceBundle, retrieval_result=None, metadata=None, session_context=None, **kwargs) -> QAResponse:
        answer = " ".join(e.snippet or "" for e in evidence_bundle.evidences[:2]).strip() or "证据不足"
        return QAResponse(
            answer=answer,
            question=question,
            evidence_bundle=evidence_bundle,
            retrieval_result=retrieval_result,
            confidence=0.76,
            metadata={"evaluation_mode": True},
        )


class EvalGraphIndexService:
    async def index_graph_result(self, graph_result: GraphExtractionResult) -> GraphIndexStats:
        return GraphIndexStats(document_id=graph_result.document_id, status="indexed")


class EvalEmbeddingIndexService:
    async def index_text_blocks(self, document_id: str, text_blocks: list[TextBlock]) -> EmbeddingIndexResult:
        return EmbeddingIndexResult(document_id=document_id, status="indexed", record_count=len(text_blocks))

    async def index_pages(self, document_id: str, pages: list[DocumentPage]) -> EmbeddingIndexResult:
        return EmbeddingIndexResult(document_id=document_id, status="indexed", record_count=len(pages))

    async def index_charts(self, document_id: str, charts: list[ChartSchema]) -> EmbeddingIndexResult:
        return EmbeddingIndexResult(document_id=document_id, status="indexed", record_count=len(charts))


def load_samples(sample_path: Path) -> tuple[ParsedDocument, ChartSchema, list[dict[str, Any]]]:
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    parsed_document = ParsedDocument.model_validate(payload["document"])
    chart = ChartSchema.model_validate(payload["chart"])
    qa_cases = list(payload.get("qa_cases", []))
    return parsed_document, chart, qa_cases


async def run_eval(sample_path: Path) -> dict[str, Any]:
    parsed_document, chart, qa_cases = load_samples(sample_path)
    runtime = RagRuntime(
        document_tools=EvalDocumentAgent(),
        chart_tools=EvalChartAgent(chart),
        graph_extraction_tools=EvalGraphExtractionAgent(),
        retrieval_tools=EvalRetrievalAgent(parsed_document, chart),
        answer_tools=EvalAnswerAgent(),
        graph_index_service=EvalGraphIndexService(),
        embedding_index_service=EvalEmbeddingIndexService(),
        llm_adapter=None,
    )

    index_result: DocumentIndexResult = await runtime.handle_index_document(parsed_document, charts=[chart])
    chart_result: ChartUnderstandingResult = await runtime.handle_understand_chart(
        image_path=str(chart.metadata.get("image_path", "synthetic-chart.png")),
        document_id=chart.document_id,
        page_id=chart.page_id,
        page_number=chart.page_number,
        chart_id=chart.id,
        context={"extract_chart_graph": True},
    )

    qa_results: list[dict[str, Any]] = []
    fused_results: list[dict[str, Any]] = []
    for case in qa_cases:
        qa = await runtime.handle_ask_document(case["question"], doc_id=parsed_document.id, session_id="portfolio-eval")
        expected_keywords = case.get("expected_keywords", [])
        matched = [keyword for keyword in expected_keywords if keyword.lower() in qa.answer.lower()]
        qa_results.append(
            {
                "question": case["question"],
                "answer": qa.answer,
                "confidence": qa.confidence,
                "matched_keywords": matched,
                "keyword_recall": round(len(matched) / max(len(expected_keywords), 1), 2),
                "evidence_count": len(qa.evidence_bundle.evidences),
                "warnings": qa.metadata.get("warnings", []),
            }
        )
        fused = await runtime.handle_ask_fused(
            question=case["question"],
            image_path=str(chart.metadata.get("image_path", "synthetic-chart.png")),
            doc_id=parsed_document.id,
            document_ids=[parsed_document.id],
            page_id=chart.page_id,
            page_number=chart.page_number,
            chart_id=chart.id,
            session_id="portfolio-eval-fused",
        )
        fused_matched = [keyword for keyword in expected_keywords if keyword.lower() in fused.qa.answer.lower()]
        fused_results.append(
            {
                "question": case["question"],
                "answer": fused.qa.answer,
                "chart_answer": fused.chart_answer,
                "chart_confidence": fused.chart_confidence,
                "matched_keywords": fused_matched,
                "keyword_recall": round(len(fused_matched) / max(len(expected_keywords), 1), 2),
                "evidence_count": len(fused.qa.evidence_bundle.evidences),
                "warnings": fused.qa.metadata.get("warnings", []),
            }
        )

    return {
        "index": {
            "status": index_result.status,
            "graph_index_status": getattr(index_result.graph_index, "status", None),
            "text_records": getattr(index_result.text_embedding_index, "record_count", 0),
        },
        "chart": {
            "chart_id": chart_result.chart.id,
            "title": chart_result.chart.title,
            "graph_text": chart_result.graph_text,
        },
        "ask": qa_results,
        "ask_fused": fused_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a low-resource portfolio evaluation for Research-Copilot.")
    parser.add_argument(
        "--samples",
        default=str(Path(__file__).resolve().parents[1] / "docs" / "portfolio_eval_samples.json"),
        help="Path to the evaluation samples JSON file.",
    )
    args = parser.parse_args()
    sample_path = Path(args.samples)
    result = __import__("asyncio").run(run_eval(sample_path))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
