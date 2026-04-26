import pytest
import httpx
from pydantic import BaseModel

from adapters.llm.base import ImageNotVisibleLLMAdapterError, LLMAdapterError
from adapters.llm.openai_relay_adapter import IMAGE_NOT_VISIBLE_SENTINEL, OpenAIRelayAdapter
from domain.schemas.chart import ChartSchema


def build_adapter() -> OpenAIRelayAdapter:
    return OpenAIRelayAdapter(api_key="test-key", model="relay-model")


def test_extract_structured_payload_normalizes_chart_fields() -> None:
    adapter = build_adapter()

    payload = adapter._extract_structured_payload(
        '{"chart_type":"Network and Line Charts","x_axis":[{"label":"x"}],"series":[{"name":"s1","points":null}]}',
        ChartSchema,
    )

    assert payload["id"] == "chart"
    assert payload["document_id"] == "document"
    assert payload["page_id"] == "page"
    assert payload["page_number"] == 1
    assert payload["chart_type"] == "line"
    assert payload["x_axis"] == {"label": "x"}
    assert payload["series"][0]["points"] == []


def test_extract_structured_payload_rejects_json_schema() -> None:
    adapter = build_adapter()

    with pytest.raises(LLMAdapterError, match="JSON Schema"):
        adapter._extract_structured_payload(
            '{"type":"object","title":"ChartSchema","properties":{"id":{"type":"string"}}}',
            ChartSchema,
        )


def test_parse_json_recovers_fenced_json() -> None:
    adapter = build_adapter()

    payload = adapter._parse_json('```json\n{"id":"c1","document_id":"d1","page_id":"p1","page_number":1,"chart_type":"bar"}\n```')

    assert payload["id"] == "c1"
    assert payload["chart_type"] == "bar"


def test_parse_json_repairs_missing_comma_between_fields() -> None:
    adapter = build_adapter()

    payload = adapter._parse_json('{"answer":"ok"\n"confidence":0.7}')

    assert payload["answer"] == "ok"
    assert payload["confidence"] == 0.7


def test_parse_json_recovers_truncated_json_object() -> None:
    adapter = build_adapter()

    payload = adapter._parse_json('{"answer":"ok","metadata":{"source":"relay"}')

    assert payload["answer"] == "ok"
    assert payload["metadata"]["source"] == "relay"


def test_parse_json_ignores_trailing_text_after_json_object() -> None:
    adapter = build_adapter()

    payload = adapter._parse_json('{"answer":"ok"}\nThis extra explanation should be ignored.')

    assert payload["answer"] == "ok"


def test_indicates_missing_image_sentinel_and_text() -> None:
    adapter = build_adapter()

    assert adapter._indicates_missing_image(IMAGE_NOT_VISIBLE_SENTINEL)
    assert adapter._indicates_missing_image("I don't see an image attached.")
    assert not adapter._indicates_missing_image('{"chart_type":"bar"}')


def test_describe_image_input_reports_basic_metadata(tmp_path) -> None:
    adapter = build_adapter()
    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    description = adapter._describe_image_input(str(image_path))

    assert description["image_path"] == str(image_path)
    assert description["image_exists"] is True
    assert description["image_size_bytes"] == 8
    assert description["image_mime_type"] == "image/png"
    assert description["image_transport"] == "data_uri:image_url"


def test_extract_structured_payload_maps_lightweight_chart_json() -> None:
    adapter = build_adapter()

    payload = adapter._extract_structured_payload(
        '{"chart_type":"Network graph and line charts","summary":"A network graph is shown with two line charts.",'
        '"x_axis_label":"Node number","y_axis_label":"Variance","series_names":["blue line","red line"],'
        '"visible_trends":["Blue and red lines vary by node."],"confidence":"High"}',
        ChartSchema,
    )

    assert payload["chart_type"] == "line"
    assert payload["summary"] == "A network graph is shown with two line charts."
    assert payload["x_axis"]["label"] == "Node number"
    assert payload["y_axis"]["label"] == "Variance"
    assert [series["name"] for series in payload["series"]] == ["blue line", "red line"]
    assert payload["metadata"]["visible_trends"] == ["Blue and red lines vary by node."]
    assert payload["confidence"] == 0.9


def test_extract_structured_payload_flattens_required_optional_contract_shape() -> None:
    adapter = build_adapter()

    payload = adapter._extract_structured_payload(
        '{"type":"Lightweight chart JSON","required":{"chart_type":"mixed","summary":"A network graph and line charts."},'
        '"optional":{"x_axis_label":"Node number k","y_axis_label":"σ²_k","series_names":["blue","red"],"confidence":0.85}}',
        ChartSchema,
    )

    assert payload["chart_type"] == "mixed"
    assert payload["summary"] == "A network graph and line charts."
    assert payload["x_axis"]["label"] == "Node number k"
    assert payload["y_axis"]["label"] == "σ²_k"
    assert [series["name"] for series in payload["series"]] == ["blue", "red"]
    assert payload["confidence"] == 0.85


class DummyStructuredResponse(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_generate_structured_retries_without_response_format_when_relay_rejects_it(monkeypatch) -> None:
    adapter = build_adapter()
    sent_payloads: list[dict] = []

    async def fake_post_chat_completion(payload: dict) -> str:
        sent_payloads.append(dict(payload))
        if len(sent_payloads) == 1:
            request = httpx.Request("POST", "https://relay.invalid/v1/chat/completions")
            response = httpx.Response(
                400,
                request=request,
                text='{"error":{"message":"response_format is not supported"}}',
            )
            raise httpx.HTTPStatusError("Bad Request", request=request, response=response)
        return '{"answer":"ok"}'

    monkeypatch.setattr(adapter, "_post_chat_completion", fake_post_chat_completion)

    response = await adapter._generate_structured(
        prompt="Return JSON",
        input_data={"question": "what happened"},
        response_model=DummyStructuredResponse,
    )

    assert response.answer == "ok"
    assert "response_format" in sent_payloads[0]
    assert "response_format" not in sent_payloads[1]


@pytest.mark.asyncio
async def test_chat_completion_surfaces_http_status_and_body(monkeypatch) -> None:
    adapter = build_adapter()

    async def fake_post_chat_completion(payload: dict) -> str:
        del payload
        request = httpx.Request("POST", "https://relay.invalid/v1/chat/completions")
        response = httpx.Response(
            429,
            request=request,
            text='{"error":{"message":"rate limit exceeded"}}',
        )
        raise httpx.HTTPStatusError("Too Many Requests", request=request, response=response)

    monkeypatch.setattr(adapter, "_post_chat_completion", fake_post_chat_completion)

    with pytest.raises(LLMAdapterError, match="status=429"):
        await adapter._chat_completion(
            model="relay-model",
            messages=[{"role": "user", "content": "hello"}],
        )


@pytest.mark.asyncio
async def test_analyze_image_structured_raises_dedicated_missing_image_error(monkeypatch, tmp_path) -> None:
    adapter = build_adapter()
    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    async def fake_chat_completion(*, model: str, messages: list[dict], json_mode: bool = False) -> str:
        return IMAGE_NOT_VISIBLE_SENTINEL

    async def fake_upload_file_and_get_url(file_path: str) -> str | None:
        assert file_path == str(image_path)
        return None

    monkeypatch.setattr(adapter, "_chat_completion", fake_chat_completion)
    monkeypatch.setattr(adapter, "_upload_file_and_get_url", fake_upload_file_and_get_url)

    with pytest.raises(ImageNotVisibleLLMAdapterError, match="not visible"):
        await adapter._analyze_image_structured(
            prompt="Analyze chart",
            image_path=str(image_path),
            response_model=ChartSchema,
        )


@pytest.mark.asyncio
async def test_analyze_image_structured_retries_with_uploaded_file_url(monkeypatch, tmp_path) -> None:
    adapter = build_adapter()
    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    seen_urls: list[str] = []

    async def fake_request_chart_payload(*, prompt: str, image_url: str, output_contract: dict) -> str:
        seen_urls.append(image_url)
        if image_url.startswith("data:"):
            return IMAGE_NOT_VISIBLE_SENTINEL
        return '{"chart_type":"bar","summary":"Uploaded file url worked."}'

    async def fake_upload_file_and_get_url(file_path: str) -> str | None:
        assert file_path == str(image_path)
        return "https://cdn.example.com/chart.png"

    monkeypatch.setattr(adapter, "_request_chart_payload", fake_request_chart_payload)
    monkeypatch.setattr(adapter, "_upload_file_and_get_url", fake_upload_file_and_get_url)

    response = await adapter._analyze_image_structured(
        prompt="Analyze chart",
        image_path=str(image_path),
        response_model=ChartSchema,
    )

    assert response.chart_type == "bar"
    assert response.summary == "Uploaded file url worked."
    assert seen_urls[0].startswith("data:")
    assert seen_urls[1] == "https://cdn.example.com/chart.png"
