from domain.schemas.research import ResearchReport
from agents.research_writer_agent import ResearchWriterAgent


def test_research_writer_quality_counts_chinese_content_without_space_tokenization() -> None:
    section = (
        "On-Policy Distillation 相关工作围绕在线策略更新、教师学生反馈、奖励对齐和推理数据构造展开。"
        "这些论文虽然可以先基于标题和摘要形成摘要级综述，但全文导入前仍需保留证据边界说明。"
    )
    report = ResearchReport(
        report_id="report-cjk",
        task_id="task-cjk",
        topic="On-Policy Distillation",
        generated_at="2026-05-19T00:00:00+00:00",
        markdown=(
            "# 文献调研报告\n\n"
            "## 研究背景\n"
            f"{section * 4} [P1][P2]\n\n"
            "## 核心问题\n"
            f"{section * 3} [P3]\n\n"
            "## 方法对比\n"
            "| 论文 | 方法 |\n| --- | --- |\n| [P1] | 在线蒸馏 |\n\n"
            "## 关键发现\n"
            f"{section * 3} [P1][P3]\n"
        ),
    )

    quality = ResearchWriterAgent()._quality_metrics(report)

    assert quality["passed"] is True
    assert quality["raw_word_count"] < 250
    assert quality["cjk_char_count"] >= 500
