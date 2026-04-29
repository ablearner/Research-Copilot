from services.research.capabilities import PaperRanker
from domain.schemas.research import PaperCandidate


def test_paper_ranker_filters_off_topic_openalex_noise_for_chinese_llm_topic() -> None:
    ranker = PaperRanker()
    relevant = PaperCandidate(
        paper_id="arxiv:2601.00001",
        title="Efficient Reasoning for Large Language Models",
        authors=["Alice"],
        abstract="We study inference-time reasoning and evaluation for LLM systems.",
        year=2026,
        venue="arXiv",
        source="arxiv",
        pdf_url="https://arxiv.org/pdf/2601.00001.pdf",
        is_open_access=True,
        published_at="2026-03-01T00:00:00+00:00",
    )
    noise = PaperCandidate(
        paper_id="openalex:noise",
        title="A Trialogue of Science, Humanities, and Lifelong Friendship",
        authors=["Bob"],
        abstract="A collection of essays on science and humanities.",
        year=2026,
        venue="OpenAlex",
        source="openalex",
        pdf_url="https://example.com/noise.pdf",
        is_open_access=True,
        published_at="2026-03-01",
    )

    ranked = ranker.rank(topic="最近 6 个月大模型方向有哪些值得关注的论文？", papers=[noise, relevant], max_papers=10)

    assert [paper.paper_id for paper in ranked] == ["arxiv:2601.00001"]
    assert ranked[0].relevance_score is not None
    assert ranked[0].metadata["rank_keyword_overlap"] > 0
