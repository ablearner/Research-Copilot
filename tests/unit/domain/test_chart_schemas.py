from domain.schemas.chart import AxisSchema, ChartSchema, SeriesPoint, SeriesSchema


def test_chart_schema_json_schema() -> None:
    assert ChartSchema.model_json_schema()["title"] == "ChartSchema"


def test_chart_models_validate() -> None:
    chart = ChartSchema(
        id="chart1",
        document_id="doc1",
        page_id="p1",
        page_number=1,
        chart_type="line",
        x_axis=AxisSchema(label="Year"),
        y_axis=AxisSchema(label="Revenue", unit="USD"),
        series=[SeriesSchema(name="Revenue", chart_role="line", points=[SeriesPoint(x="2025", y=10)])],
    )
    assert chart.series[0].points[0].y == 10
