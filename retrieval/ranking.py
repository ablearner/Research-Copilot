from __future__ import annotations

from domain.schemas.retrieval import RetrievalHit
from retrieval.cross_encoder import BaseCrossEncoderReranker


async def rerank_hits(
    query: str,
    hits: list[RetrievalHit],
    *,
    reranker: BaseCrossEncoderReranker,
) -> list[RetrievalHit]:
    if not hits:
        return []
    fallback_rerank_hits = getattr(reranker, "rerank_hits", None)
    if callable(fallback_rerank_hits):
        return await fallback_rerank_hits(query, hits)
    documents = [hit.content or "" for hit in hits]
    scores = await reranker.score(query, documents)
    reranked = [
        hit.model_copy(update={"merged_score": float(score)})
        for hit, score in zip(hits, scores, strict=True)
    ]
    return sorted(reranked, key=lambda item: item.merged_score or 0.0, reverse=True)
