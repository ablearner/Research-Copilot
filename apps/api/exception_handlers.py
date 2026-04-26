from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def build_error_detail(
    request: Request | None,
    *,
    fallback: str,
    exc: Exception,
) -> str:
    if request is None:
        return fallback

    settings = getattr(getattr(request, "app", None), "state", None)
    app_settings = getattr(settings, "settings", None)
    app_env = str(getattr(app_settings, "app_env", "") or "").lower()
    if app_env != "local":
        return fallback

    message = str(exc).strip()
    if not message:
        return f"{fallback}: {exc.__class__.__name__}"
    return f"{fallback}: {exc.__class__.__name__}: {message}"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled API exception")
        return JSONResponse(
            status_code=500,
            content={"detail": build_error_detail(request, fallback="Internal server error", exc=exc)},
        )
