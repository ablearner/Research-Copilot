from __future__ import annotations

import json
import shutil
from pathlib import Path

from domain.schemas.research import (
    PaperCandidate,
    ResearchConversation,
    ResearchJob,
    ResearchMessage,
    ResearchReport,
    ResearchTask,
)


class ResearchReportService:
    """Persist research tasks, reports, and paper snapshots to local storage."""

    def __init__(self, storage_root: Path) -> None:
        self.storage_root = storage_root
        self.tasks_root = storage_root / "tasks"
        self.reports_root = storage_root / "reports"
        self.papers_root = storage_root / "papers"
        self.conversations_root = storage_root / "conversations"
        self.messages_root = storage_root / "messages"
        self.jobs_root = storage_root / "jobs"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        self.tasks_root.mkdir(parents=True, exist_ok=True)
        self.reports_root.mkdir(parents=True, exist_ok=True)
        self.papers_root.mkdir(parents=True, exist_ok=True)
        self.conversations_root.mkdir(parents=True, exist_ok=True)
        self.messages_root.mkdir(parents=True, exist_ok=True)
        self.jobs_root.mkdir(parents=True, exist_ok=True)

    def save_task(self, task: ResearchTask) -> None:
        self._write_json(self.tasks_root / f"{task.task_id}.json", task.model_dump(mode="json"))

    def load_task(self, task_id: str) -> ResearchTask | None:
        payload = self._read_json(self.tasks_root / f"{task_id}.json")
        return ResearchTask.model_validate(payload) if payload else None

    def save_papers(self, task_id: str, papers: list[PaperCandidate]) -> None:
        self._write_json(
            self.papers_root / f"{task_id}.json",
            [paper.model_dump(mode="json") for paper in papers],
        )

    def load_papers(self, task_id: str) -> list[PaperCandidate]:
        payload = self._read_json(self.papers_root / f"{task_id}.json") or []
        return [PaperCandidate.model_validate(item) for item in payload]

    def save_report(self, report: ResearchReport) -> None:
        if report.task_id:
            task_dir = self.reports_root / report.task_id
        else:
            task_dir = self.reports_root / "adhoc"
        task_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(task_dir / f"{report.report_id}.json", report.model_dump(mode="json"))

    def load_report(self, task_id: str, report_id: str | None = None) -> ResearchReport | None:
        task_dir = self.reports_root / task_id
        if not task_dir.exists():
            return None
        candidate_path = task_dir / f"{report_id}.json" if report_id else None
        if candidate_path and candidate_path.exists():
            payload = self._read_json(candidate_path)
            return ResearchReport.model_validate(payload) if payload else None
        candidates = sorted(task_dir.glob("*.json"))
        if not candidates:
            return None
        payload = self._read_json(candidates[-1])
        return ResearchReport.model_validate(payload) if payload else None

    def save_conversation(self, conversation: ResearchConversation) -> None:
        self._write_json(
            self.conversations_root / f"{conversation.conversation_id}.json",
            conversation.model_dump(mode="json"),
        )

    def load_conversation(self, conversation_id: str) -> ResearchConversation | None:
        payload = self._read_json(self.conversations_root / f"{conversation_id}.json")
        return ResearchConversation.model_validate(payload) if payload else None

    def list_conversations(self) -> list[ResearchConversation]:
        conversations: list[ResearchConversation] = []
        for path in sorted(self.conversations_root.glob("*.json")):
            payload = self._read_json(path)
            if payload:
                conversations.append(ResearchConversation.model_validate(payload))
        conversations.sort(key=lambda item: item.updated_at, reverse=True)
        return conversations

    def delete_conversation(self, conversation_id: str) -> None:
        for path in (
            self.conversations_root / f"{conversation_id}.json",
            self.messages_root / f"{conversation_id}.json",
        ):
            if path.exists():
                path.unlink()

    def delete_task_artifacts(self, task_id: str) -> None:
        for path in (
            self.tasks_root / f"{task_id}.json",
            self.papers_root / f"{task_id}.json",
        ):
            if path.exists():
                path.unlink()
        report_dir = self.reports_root / task_id
        if report_dir.exists():
            shutil.rmtree(report_dir)

    def delete_jobs(self, *, conversation_id: str | None = None, task_id: str | None = None) -> None:
        for job in self.list_jobs(conversation_id=conversation_id, task_id=task_id):
            path = self.jobs_root / f"{job.job_id}.json"
            if path.exists():
                path.unlink()

    def save_messages(self, conversation_id: str, messages: list[ResearchMessage]) -> None:
        self._write_json(
            self.messages_root / f"{conversation_id}.json",
            [message.model_dump(mode="json") for message in messages],
        )

    def load_messages(self, conversation_id: str) -> list[ResearchMessage]:
        payload = self._read_json(self.messages_root / f"{conversation_id}.json") or []
        return [ResearchMessage.model_validate(item) for item in payload]

    def save_job(self, job: ResearchJob) -> None:
        self._write_json(self.jobs_root / f"{job.job_id}.json", job.model_dump(mode="json"))

    def load_job(self, job_id: str) -> ResearchJob | None:
        payload = self._read_json(self.jobs_root / f"{job_id}.json")
        return ResearchJob.model_validate(payload) if payload else None

    def list_jobs(self, *, conversation_id: str | None = None, task_id: str | None = None) -> list[ResearchJob]:
        jobs: list[ResearchJob] = []
        for path in sorted(self.jobs_root.glob("*.json")):
            payload = self._read_json(path)
            if not payload:
                continue
            job = ResearchJob.model_validate(payload)
            if conversation_id and job.conversation_id != conversation_id:
                continue
            if task_id and job.task_id != task_id:
                continue
            jobs.append(job)
        jobs.sort(key=lambda item: item.updated_at, reverse=True)
        return jobs

    def clear_all(self) -> None:
        if self.storage_root.exists():
            for child in self.storage_root.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        self._ensure_directories()

    def _write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_json(self, path: Path) -> object | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
