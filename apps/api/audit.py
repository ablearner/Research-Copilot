from __future__ import annotations

import logging
from typing import Any

from fastapi import Request

logger = logging.getLogger("doc_chart_copilot.audit")


def audit_api_call(
    request: Request,
    *,
    route: str,
    trace_id: str,
    task_type: str,
    status: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    settings = request.app.state.settings
    if not settings.audit_log_enabled:
        return
    logger.info(
        "api_call",
        extra={
            "route": route,
            "trace_id": trace_id,
            "task_type": task_type,
            "status": status,
            "metadata": metadata or {},
        },
    )
