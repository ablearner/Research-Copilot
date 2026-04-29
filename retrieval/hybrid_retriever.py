from __future__ import annotations

import asyncio
import logging
import hashlib
import json
from typing import Any

from langchain_core.runnables import RunnableLambda

from domain.schemas.evidence import EvidenceBundle
from domain.schemas.retrieval import HybridRetrievalResult, RetrievalHit, RetrievalQuery
from retrieval.cross_encoder import BaseCrossEncoderReranker
from retrieval.evidence_builder import build_evidence_bundle
from retrieval.fusion import apply_rrf, merge_hits
from retrieval.graph_retriever import GraphRetriever, GraphRetrieverError
from retrieval.graph_summary_retriever import GraphSummaryRetriever
from retrieval.ranking import rerank_hits
from retrieval.sparse_retriever import SparseRetriever, SparseRetrieverError
from retrieval.vector_retriever import VectorRetriever, VectorRetrieverError

logger = logging.getLogger(__name__)


class HybridRetrieverError(RuntimeError):
    """Raised when hybrid retrieval fails."""


class _NoopSparseRetriever:
    async def ainvoke(self, query: RetrievalQuery | str) -> list[RetrievalHit]:
        return []


class HybridRetriever:
    def __init__(
        self,
        graph_retriever: GraphRetriever,
        vector_retriever: VectorRetriever,
        sparse_retriever: SparseRetriever | None = None,
        graph_summary_retriever: GraphSummaryRetriever | None = None,
        reranker: BaseCrossEncoderReranker | None = None,
    ) -> None:
        self.graph_retriever = graph_retriever
        self.vector_retriever = vector_retriever
        fallback_sparse_retriever = sparse_retriever
        if fallback_sparse_retriever is None:
            vector_store = getattr(vector_retriever, "vector_store", None)
            fallback_sparse_retriever = (
                SparseRetriever(vector_store) if vector_store is not None else _NoopSparseRetriever()
            )
        self.sparse_retriever = fallback_sparse_retriever
        self.graph_summary_retriever = graph_summary_retriever
        if reranker is None:
            raise ValueError("HybridRetriever requires a Cross-Encoder reranker")
        self.reranker = reranker
        self.runnable = RunnableLambda(self.ainvoke)
        self._session_cache: dict[str, HybridRetrievalResult] = {}

    async def ainvoke(self, query: RetrievalQuery | str) -> HybridRetrievalResult:
        return await self.retrieve(query)

    def invoke(self, query: RetrievalQuery | str) -> HybridRetrievalResult:
        raise NotImplementedError("Use async retrieval for hybrid retrieval")

    def as_runnable(self) -> RunnableLambda:
        return self.runnable

    async def retrieve(self, query: RetrievalQuery | str) -> HybridRetrievalResult:
        retrieval_query = self._coerce_query(query)
        requested_mode = self._requested_retrieval_mode(retrieval_query)
        session_id = retrieval_query.filters.get("session_id")
        cache_key = self._cache_key(retrieval_query, session_id)
        cached = self._session_cache.get(cache_key) if cache_key else None
        if cached is not None:
            return cached.model_copy(
                update={
                    "metadata": {
                        **cached.metadata,
                        "cache_hit": True,
                        "cache_layer": self.__class__.__name__,
                    }
                }
            )

        try:
            task_specs: list[tuple[str, asyncio.Future | asyncio.Task | Any]] = []
            if requested_mode in {"hybrid", "graph"}:
                task_specs.append(("graph", self.graph_retriever.ainvoke(retrieval_query)))
            if requested_mode in {"hybrid", "sparse"}:
                task_specs.append(("sparse", self.sparse_retriever.ainvoke(retrieval_query)))
            if requested_mode in {"hybrid", "vector"}:
                task_specs.append(("vector", self.vector_retriever.ainvoke(retrieval_query)))
            if (
                self.graph_summary_retriever
                and (
                    requested_mode == "graphrag_summary"
                    or (requested_mode == "hybrid" and self._summary_enabled(retrieval_query))
                )
            ):
                task_specs.append(("summary", self.graph_summary_retriever.ainvoke(retrieval_query)))

            task_results = (
                await asyncio.gather(*(task for _name, task in task_specs), return_exceptions=True)
                if task_specs
                else []
            )
            sparse_hits: list[RetrievalHit] = []
            graph_hits: list[RetrievalHit] = []
            vector_hits: list[RetrievalHit] = []
            summary_hits: list[RetrievalHit] = []
            failed_sources: list[str] = []
            failures: list[Exception] = []
            for (source_name, _task), task_result in zip(task_specs, task_results):
                if isinstance(task_result, Exception):
                    failed_sources.append(source_name)
                    failures.append(task_result)
                    logger.error(
                        "Selected retrieval source failed",
                        extra={"query": retrieval_query.query, "retrieval_source": source_name},
                        exc_info=(type(task_result), task_result, task_result.__traceback__),
                    )
                    continue
                if source_name == "graph":
                    graph_hits = task_result
                elif source_name == "sparse":
                    sparse_hits = task_result
                elif source_name == "vector":
                    vector_hits = task_result
                elif source_name == "summary":
                    summary_hits = task_result

            if failures and not self._can_degrade(
                requested_mode=requested_mode,
                sparse_hits=sparse_hits,
                graph_hits=graph_hits,
                vector_hits=vector_hits,
                summary_hits=summary_hits,
            ):
                primary_exc = failures[0]
                if isinstance(primary_exc, (GraphRetrieverError, VectorRetrieverError)):
                    raise HybridRetrieverError("Hybrid retrieval dependency failed") from primary_exc
                raise HybridRetrieverError("Unexpected hybrid retrieval failure") from primary_exc

            merged_hits = merge_hits(
                sparse_hits=sparse_hits,
                graph_hits=graph_hits,
                vector_hits=vector_hits,
                summary_hits=summary_hits,
            )
            rrf_hits = apply_rrf(merged_hits)
            reranked_hits = (
                await rerank_hits(
                    retrieval_query.query,
                    rrf_hits,
                    reranker=self.reranker,
                )
            )[: retrieval_query.top_k]
            evidence_bundle = build_evidence_bundle(reranked_hits)

            result = HybridRetrievalResult(
                query=retrieval_query,
                hits=reranked_hits,
                evidence_bundle=evidence_bundle,
                metadata={
                    "graph_hit_count": len(graph_hits),
                    "sparse_hit_count": len(sparse_hits),
                    "vector_hit_count": len(vector_hits),
                    "summary_hit_count": len(summary_hits),
                    "fusion_strategy": "rrf",
                    "rerank_strategy": "cross_encoder",
                    "prefilter_strategy": "rrf",
                    "rrf_candidate_count": len(rrf_hits),
                    "requested_retrieval_mode": requested_mode,
                    "graph_summary_enabled": any(source_name == "summary" for source_name, _task in task_specs),
                    "partial_failure": bool(failed_sources),
                    "failed_sources": failed_sources,
                    "sparse_hits": [hit.model_dump(mode="json") for hit in sparse_hits],
                    "vector_hits": [hit.model_dump(mode="json") for hit in vector_hits],
                    "graph_hits": [hit.model_dump(mode="json") for hit in graph_hits],
                    "graph_summary_hits": [hit.model_dump(mode="json") for hit in summary_hits],
                },
            )
            if cache_key:
                self._session_cache[cache_key] = result
            logger.info(
                "Hybrid retrieval completed",
                extra={
                    "query": retrieval_query.query,
                    "graph_hit_count": len(graph_hits),
                    "sparse_hit_count": len(sparse_hits),
                    "vector_hit_count": len(vector_hits),
                    "summary_hit_count": len(summary_hits),
                    "merged_hit_count": len(reranked_hits),
                    "requested_retrieval_mode": requested_mode,
                    "failed_sources": failed_sources,
                },
            )
            return result
        except (GraphRetrieverError, SparseRetrieverError, VectorRetrieverError) as exc:
            logger.exception("Hybrid retrieval dependency failed", extra={"query": retrieval_query.query})
            raise HybridRetrieverError("Hybrid retrieval dependency failed") from exc
        except Exception as exc:
            logger.exception("Unexpected hybrid retrieval failure", extra={"query": retrieval_query.query})
            raise HybridRetrieverError("Unexpected hybrid retrieval failure") from exc

    async def rerank_hits(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        return await rerank_hits(query, hits, reranker=self.reranker)

    def build_evidence_bundle(self, hits: list[RetrievalHit]) -> EvidenceBundle:
        return build_evidence_bundle(hits)


    def _summary_enabled(self, query: RetrievalQuery) -> bool:
        if query.graph_query_mode == "summary":
            return True
        if query.graph_query_mode == "auto":
            return bool(query.filters.get("enable_graph_summary", True))
        return bool(query.filters.get("enable_graph_summary", False))

    def _requested_retrieval_mode(self, query: RetrievalQuery) -> str:
        requested = query.filters.get("retrieval_mode") or query.mode
        if requested in {"vector", "graph", "sparse", "hybrid", "graphrag_summary"}:
            return str(requested)
        return "hybrid"

    def _can_degrade(
        self,
        *,
        requested_mode: str,
        sparse_hits: list[RetrievalHit],
        graph_hits: list[RetrievalHit],
        vector_hits: list[RetrievalHit],
        summary_hits: list[RetrievalHit],
    ) -> bool:
        if requested_mode != "hybrid":
            return False
        return bool(sparse_hits or graph_hits or vector_hits or summary_hits)

    def _coerce_query(self, query: RetrievalQuery | str) -> RetrievalQuery:
        if isinstance(query, RetrievalQuery):
            return query.model_copy(update={"mode": "hybrid"})
        return RetrievalQuery(query=query, mode="hybrid")

    def _cache_key(self, query: RetrievalQuery, session_id: str | None) -> str | None:
        if not session_id:
            return None
        raw = json.dumps(
            {
                "session_id": session_id,
                "query": query.query,
                "document_ids": sorted(query.document_ids),
                "filters": query.filters,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
