from tools.paper_figure_toolkit import PaperFigureTools
from domain.schemas.document import DocumentPage, TextBlock


def test_page_fallback_candidate_infers_figure_caption_and_title(tmp_path) -> None:
    toolkit = PaperFigureTools(storage_root=tmp_path)
    page = DocumentPage(
        id="p2",
        document_id="doc1",
        page_number=2,
        image_uri="/tmp/page.png",
        text_blocks=[
            TextBlock(
                id="tb1",
                document_id="doc1",
                page_id="p2",
                page_number=2,
                text="Figure 2. Overall system architecture for the proposed pipeline.",
                block_type="caption",
            )
        ],
    )

    candidate = toolkit._page_fallback_candidate(page)

    assert candidate is not None
    assert candidate.metadata["caption"] == "Figure 2. Overall system architecture for the proposed pipeline."
    assert candidate.metadata["title"] == "Overall system architecture for the proposed pipeline."
    assert candidate.metadata["fallback"] == "page_image"


def test_page_fallback_candidate_uses_figure_marker_from_paragraph(tmp_path) -> None:
    toolkit = PaperFigureTools(storage_root=tmp_path)
    page = DocumentPage(
        id="p3",
        document_id="doc1",
        page_number=3,
        image_uri="/tmp/page.png",
        text_blocks=[
            TextBlock(
                id="tb1",
                document_id="doc1",
                page_id="p3",
                page_number=3,
                text="Fig. 3: Three-stage workflow with encoder, planner, and executor.",
                block_type="paragraph",
            )
        ],
    )

    candidate = toolkit._page_fallback_candidate(page)

    assert candidate is not None
    assert candidate.metadata["caption"] == "Fig. 3: Three-stage workflow with encoder, planner, and executor."
    assert candidate.metadata["title"] == "Three-stage workflow with encoder, planner, and executor."
