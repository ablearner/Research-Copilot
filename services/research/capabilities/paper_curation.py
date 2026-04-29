from __future__ import annotations

from typing import TYPE_CHECKING

from domain.schemas.research import PaperCandidate

if TYPE_CHECKING:
    from services.research.paper_search_service import PaperSearchService


class PaperCurator:
    """Turn raw paper hits into a ranked, ingest-ready candidate set."""

    name = "PaperCurator"

    def __init__(self, paper_search_service: PaperSearchService) -> None:
        self.paper_search_service = paper_search_service

    def curate(
        self,
        *,
        topic: str,
        raw_papers: list[PaperCandidate],
        max_papers: int,
    ) -> tuple[list[PaperCandidate], list[str], list[str]]:
        deduped_papers = self.paper_search_service._dedupe(raw_papers)
        ranked_papers = self.paper_search_service.paper_ranker.rank(
            topic=topic,
            papers=deduped_papers,
            max_papers=max(max_papers, len(deduped_papers)),
        )
        must_read_ids = [paper.paper_id for paper in ranked_papers[: min(5, len(ranked_papers))]]
        ingest_candidate_ids = [
            paper.paper_id
            for paper in ranked_papers
            if paper.pdf_url and paper.ingest_status not in {"ingested", "unavailable"}
        ][: min(5, len(ranked_papers))]
        curated: list[PaperCandidate] = []
        for index, paper in enumerate(ranked_papers[:max_papers], start=1):
            curated.append(
                paper.model_copy(
                    update={
                        "ingest_status": "selected"
                        if paper.paper_id in ingest_candidate_ids
                        else paper.ingest_status,
                        "metadata": {
                            **paper.metadata,
                            "curation_rank": index,
                            "curation_skill": self.name,
                            "must_read": paper.paper_id in must_read_ids,
                            "selected_for_ingest": paper.paper_id in ingest_candidate_ids,
                        },
                    }
                )
            )
        return curated, must_read_ids, ingest_candidate_ids


# Compatibility alias for older imports/tests that still use the previous name.
PaperCuratorAgent = PaperCurator
