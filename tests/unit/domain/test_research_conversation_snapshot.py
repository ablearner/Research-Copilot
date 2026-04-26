from domain.schemas.research import ResearchConversation


def test_research_conversation_allows_negative_merged_rerank_scores_in_snapshot() -> None:
    payload = {
        "conversation_id": "conv_test_negative_merged_score",
        "title": "Negative merged score snapshot",
        "created_at": "2026-04-21T00:00:00+00:00",
        "updated_at": "2026-04-21T00:00:00+00:00",
        "message_count": 1,
        "snapshot": {
            "topic": "VLN",
            "ask_result": {
                "task_id": "task_1",
                "paper_ids": [],
                "document_ids": [],
                "scope_mode": "all_imported",
                "warnings": [],
                "todo_items": [],
                "qa": {
                    "question": "哪些论文值得看？",
                    "answer": "可优先查看最新结果。",
                    "confidence": 0.4,
                    "metadata": {},
                    "evidence_bundle": {
                        "evidences": [],
                        "summary": "0 evidence items",
                        "metadata": {},
                    },
                    "retrieval_result": {
                        "query": {
                            "query": "VLN recent papers",
                            "document_ids": [],
                            "mode": "hybrid",
                            "modalities": [],
                            "top_k": 10,
                            "filters": {},
                            "graph_query_mode": "auto",
                        },
                        "hits": [
                            {
                                "id": "hit_1",
                                "source_type": "text_block",
                                "source_id": "block_1",
                                "document_id": "doc_1",
                                "content": "Instruction-as-State proposes a new embodied navigation method.",
                                "merged_score": -5.23,
                                "graph_nodes": [],
                                "graph_edges": [],
                                "graph_triples": [],
                                "metadata": {},
                            }
                        ],
                        "evidence_bundle": {
                            "evidences": [],
                            "summary": "0 evidence items",
                            "metadata": {},
                        },
                        "metadata": {},
                    },
                },
            },
        },
        "metadata": {},
    }

    conversation = ResearchConversation.model_validate(payload)

    assert conversation.snapshot.ask_result is not None
    assert conversation.snapshot.ask_result.qa.retrieval_result is not None
    assert conversation.snapshot.ask_result.qa.retrieval_result.hits[0].merged_score == -5.23


def test_research_conversation_supports_context_summary_and_runtime_events() -> None:
    payload = {
        "conversation_id": "conv_runtime_summary",
        "title": "Runtime summary snapshot",
        "created_at": "2026-04-23T00:00:00+00:00",
        "updated_at": "2026-04-23T00:00:00+00:00",
        "message_count": 0,
        "status_metadata": {
            "lifecycle_status": "running",
            "updated_at": "2026-04-23T00:00:00+00:00",
            "correlation_id": "corr_1",
        },
        "snapshot": {
            "topic": "GraphRAG",
            "context_summary": {
                "objective": "GraphRAG literature review",
                "current_stage": "qa",
                "topic": "GraphRAG",
                "paper_count": 4,
                "imported_document_count": 3,
                "selected_paper_count": 2,
                "key_findings": ["Graph summaries improve recall."],
                "evidence_gaps": ["Need more long-document benchmarks."],
                "next_actions": ["Compare summary retrievers."],
                "status_summary": "Collection QA in progress.",
                "last_user_message": "总结一下当前缺口",
                "last_updated_at": "2026-04-23T00:00:00+00:00",
            },
            "recent_events": [
                {
                    "event_id": "evt_1",
                    "event_type": "memory_updated",
                    "conversation_id": "conv_runtime_summary",
                    "task_id": "task_1",
                    "correlation_id": "corr_1",
                    "timestamp": "2026-04-23T00:00:00+00:00",
                    "payload": {"workspace_stage": "qa"},
                }
            ],
        },
    }

    conversation = ResearchConversation.model_validate(payload)

    assert conversation.snapshot.context_summary.paper_count == 4
    assert conversation.snapshot.context_summary.key_findings == ["Graph summaries improve recall."]
    assert conversation.snapshot.recent_events[0].event_type == "memory_updated"
    assert conversation.status_metadata.correlation_id == "corr_1"
