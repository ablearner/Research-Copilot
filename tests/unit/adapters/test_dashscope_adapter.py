import pytest

from adapters.llm.dashscope_adapter import DashScopeLLMAdapter
from domain.schemas.chart import ChartSchema


def build_adapter() -> DashScopeLLMAdapter:
    return DashScopeLLMAdapter(api_key="test-key")


def test_extract_structured_payload_fills_required_chart_fields() -> None:
    adapter = build_adapter()

    payload = adapter._extract_structured_payload(
        '{"chart_type": "Network and Line Charts", "summary": "demo"}',
        ChartSchema,
    )

    assert payload["id"] == "chart"
    assert payload["document_id"] == "document"
    assert payload["page_id"] == "page"
    assert payload["page_number"] == 1
    assert payload["chart_type"] == "line"
    assert payload["metadata"]["raw_chart_type"] == "Network and Line Charts"


def test_extract_structured_payload_rejects_returned_json_schema() -> None:
    adapter = build_adapter()

    schema_like = '{"type": "object", "title": "ChartSchema", "properties": {"id": {"type": "string"}}}'
    with pytest.raises(Exception, match="JSON Schema"):
        adapter._extract_structured_payload(schema_like, ChartSchema)


def test_parse_json_recovers_wrapped_json_text() -> None:
    adapter = build_adapter()

    wrapped = 'Here is the result:\n```json\n{"id":"c1","document_id":"d1","page_id":"p1","page_number":1,"chart_type":"bar"}\n```'
    payload = adapter._parse_json(wrapped)

    assert payload["id"] == "c1"
    assert payload["chart_type"] == "bar"


def test_extract_structured_payload_normalizes_axis_lists_and_null_points() -> None:
    adapter = build_adapter()

    payload = adapter._extract_structured_payload(
        '{"id":"c1","document_id":"d1","page_id":"p1","page_number":1,"chart_type":"line",'
        '"x_axis":[{"label":"x"}],"y_axis":[{"label":"y"}],"series":[{"name":"s1","points":null}]}',
        ChartSchema,
    )

    assert payload["x_axis"] == {"label": "x"}
    assert payload["y_axis"] == {"label": "y"}
    assert payload["series"][0]["points"] == []
