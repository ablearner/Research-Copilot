import pytest

from agents.preference_memory_agent import PreferenceMemoryAgent
from domain.schemas.research import PaperCandidate, ResearchReport, ResearchTopicPlan
from memory.long_term_memory import JsonLongTermMemoryStore, LongTermMemory
from memory.memory_manager import MemoryManager
from tools.research.paper_search import SearchResultBundle


class PreferenceSearchServiceStub:
    def __init__(self, *, llm_adapter=None) -> None:
        self.llm_adapter = llm_adapter

    async def search(self, *, topic: str, days_back: int, max_papers: int, sources: list[str], task_id=None):
        del days_back, max_papers, sources, task_id
        normalized = topic.lower()
        if "graphrag" in normalized:
            papers = [
                PaperCandidate(
                    paper_id="arxiv:graphrag-1",
                    title="Recent GraphRAG Agents for Literature Review",
                    authors=["Alice"],
                    abstract="GraphRAG and agent memory for research workflows.",
                    year=2026,
                    source="arxiv",
                    url="https://arxiv.org/abs/graphrag-1",
                    pdf_url="https://arxiv.org/pdf/graphrag-1.pdf",
                    published_at="2026-04-10T00:00:00+00:00",
                )
            ]
        else:
            papers = [
                PaperCandidate(
                    paper_id="arxiv:memory-1",
                    title="Long-Term Agent Memory for Scientific Copilots",
                    authors=["Bob"],
                    abstract="Persistent preference memory for research agents.",
                    year=2026,
                    source="arxiv",
                    url="https://arxiv.org/abs/memory-1",
                    pdf_url="https://arxiv.org/pdf/memory-1.pdf",
                    published_at="2026-04-08T00:00:00+00:00",
                )
            ]
        return SearchResultBundle(
            plan=ResearchTopicPlan(topic=topic, normalized_topic=topic.lower(), queries=[topic], sources=["arxiv"]),
            papers=papers,
            report=ResearchReport(
                report_id=f"report_{topic}",
                topic=topic,
                generated_at="2026-04-25T00:00:00+00:00",
                markdown=f"# {topic}",
                paper_count=len(papers),
            ),
            warnings=[],
        )


class SourceSelectionLLMStub:
    def __init__(self, sources: list[str]) -> None:
        self.sources = list(sources)
        self.calls: list[dict] = []

    async def generate_structured(self, prompt: str, input_data: dict, response_model):
        self.calls.append({"prompt": prompt, "input_data": dict(input_data)})
        return response_model(sources=self.sources, rationale="Pick broader scholarly sources for this query.")


@pytest.mark.asyncio
async def test_preference_memory_agent_persists_profile_and_recommends_across_restarts(tmp_path) -> None:
    long_term_root = tmp_path / "memory" / "long_term"
    memory_manager = MemoryManager(
        long_term_memory=LongTermMemory(JsonLongTermMemoryStore(long_term_root))
    )
    agent = PreferenceMemoryAgent(
        memory_manager=memory_manager,
        paper_search_service=PreferenceSearchServiceStub(),
        storage_root=tmp_path,
    )

    observed = agent.observe_user_message(
        message="我最近经常问 GraphRAG 和 agent memory 的论文，优先看 arXiv。",
        sources=["arxiv"],
    )

    assert observed.interest_topics
    assert observed.interest_topics[0].topic_name in {"GraphRAG", "agent memory"}

    reloaded_agent = PreferenceMemoryAgent(
        memory_manager=MemoryManager(
            long_term_memory=LongTermMemory(JsonLongTermMemoryStore(long_term_root))
        ),
        paper_search_service=PreferenceSearchServiceStub(),
        storage_root=tmp_path,
    )

    output = await reloaded_agent.recommend_recent_papers(
        question="给我推荐最近值得看的论文",
        days_back=30,
        top_k=3,
        sources=["arxiv"],
    )

    assert output.recommendations
    assert output.metadata["topics_used"]
    assert (tmp_path / "notifications" / "queue.json").exists()


def test_extract_preference_signal_strips_recency_and_source_names_from_topics(tmp_path) -> None:
    memory_manager = MemoryManager(
        long_term_memory=LongTermMemory(JsonLongTermMemoryStore(tmp_path / "memory" / "long_term"))
    )
    agent = PreferenceMemoryAgent(
        memory_manager=memory_manager,
        paper_search_service=PreferenceSearchServiceStub(),
        storage_root=tmp_path,
    )

    signal = agent.extract_preference_signal("给我推荐最近三个月 GraphRAG 值得看的论文，优先看 arXiv。")

    assert "GraphRAG" in signal["topics"]
    assert all("arxiv" not in topic.lower() for topic in signal["topics"])
    assert all("三个" not in topic and "三个月" not in topic for topic in signal["topics"])
    assert signal["sources"] == ["arxiv"]


def test_extract_preference_signal_ignores_request_wrapped_topics_and_session_titles(tmp_path) -> None:
    memory_manager = MemoryManager(
        long_term_memory=LongTermMemory(JsonLongTermMemoryStore(tmp_path / "memory" / "long_term"))
    )
    agent = PreferenceMemoryAgent(
        memory_manager=memory_manager,
        paper_search_service=PreferenceSearchServiceStub(),
        storage_root=tmp_path,
    )

    signal = agent.extract_preference_signal("科研助手里的长期记忆方向，有没有比较新的论文？")

    assert "科研助手里的长期记忆方向，有没有比较新的论文" not in signal["topics"]
    assert "未命名研究会话" not in agent.extract_preference_signal("未命名研究会话")["topics"]


def test_observe_user_message_does_not_learn_session_sources_as_long_term_preference(tmp_path) -> None:
    memory_manager = MemoryManager(
        long_term_memory=LongTermMemory(JsonLongTermMemoryStore(tmp_path / "memory" / "long_term"))
    )
    agent = PreferenceMemoryAgent(
        memory_manager=memory_manager,
        paper_search_service=PreferenceSearchServiceStub(),
        storage_root=tmp_path,
    )

    profile = agent.observe_user_message(
        message="给我推荐最近值得看的论文",
        sources=["arxiv"],
    )

    assert profile.preferred_sources == []


@pytest.mark.asyncio
async def test_recommend_recent_papers_prefers_llm_selected_sources_over_current_session_scope(tmp_path) -> None:
    llm = SourceSelectionLLMStub(["openalex", "semantic_scholar"])
    long_term_root = tmp_path / "memory" / "long_term"
    memory_manager = MemoryManager(
        long_term_memory=LongTermMemory(JsonLongTermMemoryStore(long_term_root))
    )
    agent = PreferenceMemoryAgent(
        memory_manager=memory_manager,
        paper_search_service=PreferenceSearchServiceStub(llm_adapter=llm),
        storage_root=tmp_path,
    )
    agent.observe_user_message(message="我一直关注 GraphRAG 和 agent memory。")

    output = await agent.recommend_recent_papers(
        question="给我推荐最近值得看的论文",
        days_back=30,
        top_k=3,
        sources=["arxiv"],
    )

    assert output.metadata["resolved_sources"] == ["openalex", "semantic_scholar"]
    assert output.metadata["source_selection"]["mode"] == "llm"
    assert output.metadata["topic_groups"]
    assert llm.calls


@pytest.mark.asyncio
async def test_recommend_recent_papers_ignores_noisy_profile_topics_and_emits_grouped_paper_details(tmp_path) -> None:
    memory_manager = MemoryManager(
        long_term_memory=LongTermMemory(JsonLongTermMemoryStore(tmp_path / "memory" / "long_term"))
    )
    agent = PreferenceMemoryAgent(
        memory_manager=memory_manager,
        paper_search_service=PreferenceSearchServiceStub(),
        storage_root=tmp_path,
    )
    memory_manager.update_user_profile(topics=["未命名研究会话", "优先 arXiv", "GraphRAG"])

    output = await agent.recommend_recent_papers(
        question="给我推荐一下论文",
        days_back=30,
        top_k=2,
        sources=["arxiv"],
    )

    groups = list(output.metadata["topic_groups"])
    assert groups
    assert groups[0]["topic"] == "GraphRAG"
    assert all(group["topic"] not in {"未命名研究会话", "优先 arXiv"} for group in groups)
    assert groups[0]["papers"][0]["url"] == "https://arxiv.org/abs/graphrag-1"
    assert groups[0]["papers"][0]["explanation"]
