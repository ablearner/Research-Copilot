from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from domain.schemas.research_functions import (
    ResearchDateRange,
    ResearchSortBy,
    SearchPaperResult,
)


class AcademicSearchToolInput(BaseModel):
    source: list[str] = Field(default_factory=list)
    query: str = Field(min_length=1, max_length=1000)
    date_range: ResearchDateRange | None = None
    max_results: int = Field(default=10, ge=1, le=100)
    sort_by: ResearchSortBy = "relevance"


class LocalFileEntry(BaseModel):
    path: str
    entry_type: Literal["file", "directory"]
    size_bytes: int | None = Field(default=None, ge=0)


class LocalFileToolInput(BaseModel):
    operation: Literal["read", "write", "append", "delete", "list"]
    path: str = Field(min_length=1)
    content: str | None = None
    encoding: str = "utf-8"


class LocalFileToolOutput(BaseModel):
    operation: str
    path: str
    success: bool = True
    content: str | None = None
    entries: list[LocalFileEntry] = Field(default_factory=list)
    existed: bool | None = None


class CodeExecutionToolInput(BaseModel):
    code: str = Field(min_length=1)
    timeout_seconds: int = Field(default=10, ge=1, le=60)
    working_directory: str | None = None


class CodeExecutionToolOutput(BaseModel):
    success: bool
    return_code: int
    stdout: str = ""
    stderr: str = ""
    executed_with: str


class WebSearchResultItem(BaseModel):
    title: str
    url: str
    snippet: str = ""
    published_at: str | None = None


class WebSearchToolInput(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    provider: Literal["auto", "tavily", "brave"] = "auto"
    max_results: int = Field(default=5, ge=1, le=20)


class WebSearchToolOutput(BaseModel):
    provider: str
    results: list[WebSearchResultItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class NotificationItem(BaseModel):
    notification_id: str
    message: str
    channel: Literal["system", "queue"] = "queue"
    trigger_at: datetime | None = None
    status: Literal["queued", "dismissed"] = "queued"
    metadata: dict[str, Any] = Field(default_factory=dict)


class NotificationToolInput(BaseModel):
    operation: Literal["enqueue", "list", "dismiss"] = "enqueue"
    message: str = ""
    channel: Literal["system", "queue"] = "queue"
    trigger_at: datetime | None = None
    notification_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NotificationToolOutput(BaseModel):
    status: str
    queue_size: int = Field(default=0, ge=0)
    items: list[NotificationItem] = Field(default_factory=list)


class LibrarySyncToolInput(BaseModel):
    provider: Literal["filesystem", "zotero", "notion"] = "filesystem"
    operation: Literal["export", "sync"] = "export"
    paper_ids: list[str] = Field(default_factory=list)
    target_collection: str | None = None


class LibrarySyncToolOutput(BaseModel):
    provider: str
    status: str
    exported_count: int = Field(default=0, ge=0)
    output_path: str | None = None
    warnings: list[str] = Field(default_factory=list)


class SearchOrImportPaperToolInput(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    source: list[str] = Field(default_factory=list)
    date_range: ResearchDateRange | None = None
    max_results: int = Field(default=5, ge=1, le=20)
    sort_by: ResearchSortBy = "relevance"
    candidate_index: int = Field(default=0, ge=0, le=19)
    collection_name: str | None = None
    ingest_to_workspace: bool = False


class SearchOrImportPaperToolOutput(BaseModel):
    status: Literal["imported", "reused", "not_found", "failed", "not_configured"]
    action: Literal["imported", "reused", "none"] = "none"
    selected_paper_id: str | None = None
    selected_paper_title: str | None = None
    candidate_index: int | None = Field(default=None, ge=0)
    candidates: list[SearchPaperResult] = Field(default_factory=list)
    zotero_item_key: str | None = None
    workspace_document_id: str | None = None
    workspace_status: Literal["imported", "skipped", "failed"] | None = None
    matched_by: Literal["doi", "title", "url", "pdf_url"] | None = None
    collection_name: str | None = None
    attachment_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)
