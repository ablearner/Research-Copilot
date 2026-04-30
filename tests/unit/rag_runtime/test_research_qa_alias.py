from rag_runtime.runtime import RagRuntime


class DummyAnswerTools:
    llm_adapter = None


def test_rag_runtime_exposes_research_qa_agent_alias() -> None:
    react_agent = object()
    runtime = RagRuntime(
        document_tools=object(),
        chart_tools=object(),
        graph_extraction_tools=object(),
        retrieval_tools=object(),
        answer_tools=DummyAnswerTools(),
        graph_index_service=object(),
        embedding_index_service=object(),
        react_reasoning_agent=react_agent,
    )

    assert runtime.react_reasoning_agent is react_agent
    assert runtime.research_qa_agent is react_agent
