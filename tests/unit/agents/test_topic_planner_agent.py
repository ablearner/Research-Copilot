from services.research.capabilities import TopicPlanner


def test_topic_planner_agent_simplifies_natural_language_research_request() -> None:
    planner = TopicPlanner()

    plan = planner.plan(
        topic="最近 6 个月无人机路径规划方向有哪些值得关注的论文？",
        days_back=180,
        max_papers=12,
        sources=["arxiv", "openalex", "semantic_scholar"],
    )

    assert plan.metadata["simplified_topic"] == "无人机路径规划"
    assert "无人机路径规划" in plan.queries
    assert "UAV path planning" in plan.queries


def test_topic_planner_agent_expands_chinese_llm_topic_to_english_queries() -> None:
    planner = TopicPlanner()

    plan = planner.plan(
        topic="最近 6 个月大模型方向有哪些值得关注的论文？",
        days_back=180,
        max_papers=12,
        sources=["arxiv", "openalex", "semantic_scholar"],
    )

    assert plan.metadata["simplified_topic"] == "大模型"
    assert plan.metadata["query_language_policy"] == "english_for_scholarly_sources"
    assert "large language model" in plan.queries
    assert "LLM" in plan.queries


def test_topic_planner_agent_uses_english_queries_for_english_literature_sources() -> None:
    planner = TopicPlanner()
    plan = planner.plan(
        topic="最近 6 个月大模型方向有哪些值得关注的论文？",
        days_back=180,
        max_papers=12,
        sources=["arxiv", "openalex", "semantic_scholar"],
    )

    arxiv_queries = planner.queries_for_source(source="arxiv", queries=plan.queries)
    semantic_queries = planner.queries_for_source(source="semantic_scholar", queries=plan.queries)
    openalex_queries = planner.queries_for_source(source="openalex", queries=plan.queries)

    assert arxiv_queries == ["large language model", "foundation model"]
    assert semantic_queries == ["large language model"]
    assert openalex_queries == ["large language model", "foundation model", "LLM"]
    assert all(not any("\u4e00" <= char <= "\u9fff" for char in query) for query in arxiv_queries + semantic_queries + openalex_queries)
