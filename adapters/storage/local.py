"""Local JSON file storage — alias for ResearchReportService."""

from services.research.research_report_service import ResearchReportService as JsonFileStore

__all__ = ["JsonFileStore"]
