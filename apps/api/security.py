from __future__ import annotations

from fastapi import Header, HTTPException, Request, status


async def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    settings = request.app.state.settings
    if not settings.api_key_enabled:
        return
    if not settings.api_key or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


async def build_quota_context(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, str]:
    quota_subject = x_api_key[-6:] if x_api_key else "anonymous"
    context = {
        "quota_subject": quota_subject,
        "quota_bucket": f"{request.url.path}:{request.method.lower()}",
    }
    request.state.quota_context = context
    return context
