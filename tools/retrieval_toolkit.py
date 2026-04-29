import logging
from typing import Any

from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from domain.schemas.evidence import EvidenceBundle
from domain.schemas.retrieval import HybridRetrievalResult, RetrievalQuery
from retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverError

logger = logging.getLogger(__name__)


class RetrievalAgentError(RuntimeError):
    """Raised when retrieval evidence preparation fails."""


class RetrievalAgentResult(BaseModel):
    question: str
    document_ids: list[str] = Field(default_factory=list)
    evidence_bundle: EvidenceBundle
    retrieval_result: HybridRetrievalResult
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalInput(BaseModel):
    question: str
    doc_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    top_k: int = 10
    filters: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None
    task_id: str | None = None
    memory_hints: dict[str, Any] = Field(default_factory=dict)


def resolve_document_ids(
    doc_id: str | None,
    document_ids: list[str] | None,
) -> list[str]:
    resolved = list(document_ids or [])
    if doc_id and doc_id not in resolved:
        resolved.append(doc_id)
    return resolved


async def retrieve_run(
    *,
    hybrid_retriever: HybridRetriever,
    question: str,
    doc_id: str | None = None,
    document_ids: list[str] | None = None,
    top_k: int = 10,
    filters: dict[str, Any] | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
    memory_hints: dict[str, Any] | None = None,
) -> RetrievalAgentResult:
    resolved_document_ids = resolve_document_ids(doc_id, document_ids)
    resolved_memory_hints = dict(memory_hints or {})
    query_filters = {
        **(filters or {}),
        "session_id": session_id,
        "task_id": task_id,
        "session_context": resolved_memory_hints,
        "task_context": {"task_id": task_id} if task_id else {},
        "memory_hints": resolved_memory_hints,
    }
    query = RetrievalQuery(
        query=question,
        document_ids=resolved_document_ids,
        mode="hybrid",
        top_k=top_k,
        filters=query_filters,
    )
    try:
        retrieval_result = await hybrid_retriever.retrieve(query)
        retrieval_result = retrieval_result.model_copy(
            update={
                "metadata": {
                    **retrieval_result.metadata,
                    "cache_key": None,
                }
            }
        )
        logger.info(
            "Retrieval agent prepared evidence",
            extra={
                "document_ids": resolved_document_ids,
                "hit_count": len(retrieval_result.hits),
                "evidence_count": len(retrieval_result.evidence_bundle.evidences),
                },
        )
        return RetrievalAgentResult(
            question=question,
            document_ids=resolved_document_ids,
            evidence_bundle=retrieval_result.evidence_bundle,
            retrieval_result=retrieval_result,
            metadata={
                "top_k": top_k,
                "cache_hit": bool(retrieval_result.metadata.get("cache_hit")),
                "cache_key": retrieval_result.metadata.get("cache_key"),
            },
        )
    except HybridRetrieverError as exc:
        logger.exception("Hybrid retriever failed in retrieval agent")
        raise RetrievalAgentError("Hybrid retriever failed in retrieval agent") from exc
    except Exception as exc:
        logger.exception("Unexpected retrieval agent failure")
        raise RetrievalAgentError("Unexpected retrieval agent failure") from exc


class RetrievalAgent:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.retrieve_tool = StructuredTool.from_function(
            coroutine=self.retrieve,
            name="hybrid_retrieve",
            description="Retrieve vector and graph evidence for a question.",
            args_schema=RetrievalInput,
        )
        self.retrieve_chain = RunnableLambda(lambda payload: payload)

    async def retrieve(
        self,
        question: str,
        doc_id: str | None = None,
        document_ids: list[str] | None = None,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        memory_hints: dict[str, Any] | None = None,
    ) -> RetrievalAgentResult:
        return await retrieve_run(
            hybrid_retriever=self.hybrid_retriever,
            question=question,
            doc_id=doc_id,
            document_ids=document_ids,
            top_k=top_k,
            filters=filters,
            session_id=session_id,
            task_id=task_id,
            memory_hints=memory_hints,
        )

    async def tool_hybrid_retrieve(
        self,
        question: str,
        doc_id: str | None = None,
        document_ids: list[str] | None = None,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        memory_hints: dict[str, Any] | None = None,
    ) -> RetrievalAgentResult:
        return await self.retrieve(
            question=question,
            doc_id=doc_id,
            document_ids=document_ids,
            top_k=top_k,
            filters=filters,
            session_id=session_id,
            task_id=task_id,
            memory_hints=memory_hints,
        )


RetrievalTools = RetrievalAgent
RetrievalToolsError = RetrievalAgentError
RetrievalResult = RetrievalAgentResult
