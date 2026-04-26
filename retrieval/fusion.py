from __future__ import annotations

from domain.schemas.evidence import EvidenceBundle
from domain.schemas.graph import GraphTriple
from domain.schemas.retrieval import RetrievalHit


def apply_rrf(
    hits: list[RetrievalHit],
    *,
    k: int = 60,
) -> list[RetrievalHit]:
    reranked: list[RetrievalHit] = []
    for hit in hits:
        source_ranks = hit.metadata.get("source_ranks")
        rrf_score = _rrf_score(source_ranks, k=k)
        reranked.append(
            hit.model_copy(
                update={
                    "merged_score": rrf_score,
                    "metadata": {
                        **hit.metadata,
                        "rrf_score": rrf_score,
                        "rrf_k": k,
                    },
                }
            )
        )
    return sorted(reranked, key=lambda item: item.merged_score or 0.0, reverse=True)


def merge_hits(
    *,
    sparse_hits: list[RetrievalHit],
    vector_hits: list[RetrievalHit],
    graph_hits: list[RetrievalHit],
    summary_hits: list[RetrievalHit] | None = None,
) -> list[RetrievalHit]:
    merged: dict[str, RetrievalHit] = {}

    for hits, source in (
        (sparse_hits, "sparse"),
        (vector_hits, "vector"),
        ((summary_hits or []), "graph_summary"),
        (graph_hits, "graph"),
    ):
        for rank, hit in enumerate(hits, start=1):
            key = merge_key(hit)
            source_ranks = _merge_source_ranks(hit.metadata.get("source_ranks"), source=source, rank=rank)
            if key not in merged:
                merged[key] = hit.model_copy(
                    update={
                        "metadata": {
                            **hit.metadata,
                            "retrieval_sources": sorted({*hit.metadata.get("retrieval_sources", []), source}),
                            "source_ranks": source_ranks,
                            "original_source_type": hit.source_type,
                        }
                    }
                )
                continue
            existing = merged[key]
            merged[key] = existing.model_copy(
                update={
                    "sparse_score": hit.sparse_score if hit.sparse_score is not None else existing.sparse_score,
                    "vector_score": hit.vector_score if hit.vector_score is not None else existing.vector_score,
                    "graph_score": hit.graph_score if hit.graph_score is not None else existing.graph_score,
                    "graph_nodes": merge_by_id(existing.graph_nodes, hit.graph_nodes),
                    "graph_edges": merge_by_id(existing.graph_edges, hit.graph_edges),
                    "graph_triples": merge_triples(existing.graph_triples, hit.graph_triples),
                    "evidence": merge_evidence_bundles(existing.evidence, hit.evidence),
                    "metadata": {
                        **existing.metadata,
                        **hit.metadata,
                        "retrieval_sources": sorted(
                            {
                                *existing.metadata.get("retrieval_sources", []),
                                *hit.metadata.get("retrieval_sources", []),
                                source,
                            }
                        ),
                        "source_ranks": _merge_source_ranks(
                            existing.metadata.get("source_ranks"),
                            incoming=hit.metadata.get("source_ranks"),
                            source=source,
                            rank=rank,
                        ),
                    },
                }
            )
    return list(merged.values())


def _merge_source_ranks(
    existing: object,
    *,
    incoming: object | None = None,
    source: str,
    rank: int,
) -> dict[str, int]:
    merged: dict[str, int] = {}
    for candidate in (existing, incoming):
        if not isinstance(candidate, dict):
            continue
        for key, value in candidate.items():
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if key not in merged or parsed < merged[key]:
                merged[key] = parsed
    if source not in merged or rank < merged[source]:
        merged[source] = rank
    return merged


def _rrf_score(source_ranks: object, *, k: int) -> float:
    if not isinstance(source_ranks, dict):
        return 0.0
    score = 0.0
    for rank in source_ranks.values():
        try:
            parsed_rank = int(rank)
        except (TypeError, ValueError):
            continue
        if parsed_rank <= 0:
            continue
        score += 1.0 / float(k + parsed_rank)
    return score


def merge_key(hit: RetrievalHit) -> str:
    if hit.document_id and hit.source_type in {"text_block", "page", "page_image", "chart", "graph_summary"}:
        return f"{hit.source_type}:{hit.document_id}:{hit.source_id}"
    if hit.document_id and hit.graph_nodes:
        node_ids = ",".join(sorted(node.id for node in hit.graph_nodes))
        return f"{hit.document_id}:nodes:{node_ids}"
    return f"{hit.source_type}:{hit.source_id}"


def merge_by_id(existing: list, incoming: list) -> list:
    merged = {item.id: item for item in existing}
    for item in incoming:
        merged[item.id] = item
    return list(merged.values())


def merge_triples(existing: list[GraphTriple], incoming: list[GraphTriple]) -> list[GraphTriple]:
    merged = {
        f"{triple.subject.id}:{triple.predicate.id}:{triple.object.id}": triple for triple in existing
    }
    for triple in incoming:
        merged[f"{triple.subject.id}:{triple.predicate.id}:{triple.object.id}"] = triple
    return list(merged.values())


def merge_evidence_bundles(
    left: EvidenceBundle | None,
    right: EvidenceBundle | None,
) -> EvidenceBundle | None:
    if left is None:
        return right
    if right is None:
        return left
    merged = {evidence.id: evidence for evidence in left.evidences}
    for evidence in right.evidences:
        merged[evidence.id] = evidence
    return EvidenceBundle(
        evidences=list(merged.values()),
        summary=left.summary or right.summary,
        metadata={**left.metadata, **right.metadata},
    )
