from __future__ import annotations

from domain.schemas.evidence import Evidence, EvidenceBundle
from domain.schemas.retrieval import RetrievalHit


def _raw_hit_score(hit: RetrievalHit) -> float | None:
    return hit.merged_score or hit.vector_score or hit.graph_score or hit.sparse_score


def _non_negative_score(score: float | None) -> float | None:
    if score is None:
        return None
    return max(float(score), 0.0)


def build_evidence_bundle(hits: list[RetrievalHit]) -> EvidenceBundle:
    evidences: dict[str, Evidence] = {}
    for hit in hits:
        raw_score = _raw_hit_score(hit)
        normalized_score = _non_negative_score(raw_score)
        if hit.evidence:
            for evidence in hit.evidence.evidences:
                evidence_metadata = dict(evidence.metadata)
                if raw_score is not None:
                    evidence_metadata["raw_retrieval_score"] = float(raw_score)
                evidences[evidence.id] = evidence.model_copy(
                    update={
                        "score": _non_negative_score(evidence.score) or normalized_score,
                        "metadata": evidence_metadata,
                    }
                )
        else:
            evidence = hit_to_evidence(hit)
            evidences[evidence.id] = evidence
    return EvidenceBundle(
        evidences=list(evidences.values()),
        summary=f"{len(evidences)} evidence items from {len(hits)} retrieval hits",
        metadata={"hit_count": len(hits)},
    )


def hit_to_evidence(hit: RetrievalHit) -> Evidence:
    raw_score = _raw_hit_score(hit)
    return Evidence(
        id=f"ev:{hit.id}",
        document_id=hit.document_id,
        source_type=evidence_source_type(hit.source_type),
        source_id=hit.source_id,
        snippet=hit.content,
        score=_non_negative_score(raw_score),
        graph_node_ids=[node.id for node in hit.graph_nodes],
        graph_edge_ids=[edge.id for edge in hit.graph_edges],
        metadata={
            **hit.metadata,
            "hit_id": hit.id,
            "hit_source_type": hit.source_type,
            **({"raw_retrieval_score": float(raw_score)} if raw_score is not None else {}),
        },
    )


def evidence_source_type(hit_source_type: str) -> str:
    if hit_source_type == "text_block":
        return "text_block"
    if hit_source_type in {"page", "page_image", "image_region"}:
        return "page_image"
    if hit_source_type == "chart":
        return "chart"
    if hit_source_type == "graph_node":
        return "graph_node"
    if hit_source_type in {"graph_edge", "graph_triple", "graph_subgraph", "graph_summary"}:
        return "graph_edge"
    return "document"
