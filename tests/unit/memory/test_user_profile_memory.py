from domain.schemas.research_memory import UserResearchProfile
from memory.long_term_memory import InMemoryLongTermMemoryStore, LongTermMemory
from memory.user_profile_memory import UserProfileMemory


def test_user_profile_memory_updates_and_loads_profile() -> None:
    memory = UserProfileMemory(LongTermMemory(InMemoryLongTermMemoryStore()))

    updated = memory.update_profile(
        user_id="user-1",
        topic="GraphRAG",
        sources=["arxiv", "openalex"],
        keywords=["groundedness", "retrieval"],
        reasoning_style="cot",
        note="prefers evidence-backed answers",
    )

    loaded = memory.load_profile(user_id="user-1")

    assert isinstance(updated, UserResearchProfile)
    assert loaded.last_active_topic == "GraphRAG"
    assert "arxiv" in loaded.preferred_sources
    assert "retrieval" in loaded.preferred_keywords
    assert loaded.preferred_reasoning_style == "cot"
    assert loaded.interest_topics
    assert loaded.interest_topics[0].normalized_topic == "graphrag"


def test_user_profile_memory_observe_query_accumulates_interest_weights() -> None:
    memory = UserProfileMemory(LongTermMemory(InMemoryLongTermMemoryStore()))

    first = memory.observe_query(
        user_id="user-2",
        topics=["GraphRAG", "agent memory"],
        sources=["arxiv"],
        keywords=["groundedness"],
    )
    second = memory.observe_query(
        user_id="user-2",
        topics=["GraphRAG"],
        sources=["openalex"],
        keywords=["retrieval"],
        signal_strength=1.4,
    )

    assert first.interest_topics
    assert second.interest_topics[0].topic_name == "GraphRAG"
    assert second.interest_topics[0].mention_count >= 2
    assert second.interest_topics[0].weight > second.interest_topics[1].weight


def test_user_profile_memory_can_remove_noisy_topics() -> None:
    memory = UserProfileMemory(LongTermMemory(InMemoryLongTermMemoryStore()))

    memory.observe_query(
        user_id="user-3",
        topics=["GraphRAG", "未命名研究会话"],
        sources=["arxiv"],
        keywords=["retrieval"],
    )
    updated = memory.remove_topics(user_id="user-3", topics=["未命名研究会话"])

    assert [item.topic_name for item in updated.interest_topics] == ["GraphRAG"]
    assert updated.last_active_topic == "GraphRAG"


def test_user_profile_memory_can_clear_profile() -> None:
    memory = UserProfileMemory(LongTermMemory(InMemoryLongTermMemoryStore()))

    memory.update_profile(
        user_id="user-4",
        topic="GraphRAG",
        sources=["arxiv"],
        keywords=["retrieval"],
        reasoning_style="cot",
    )
    cleared = memory.clear_profile(user_id="user-4")

    assert cleared.interest_topics == []
    assert cleared.preferred_sources == []
    assert cleared.preferred_keywords == []
    assert cleared.last_active_topic is None
