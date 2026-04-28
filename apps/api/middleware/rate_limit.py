"""Simple in-process rate limiter middleware for FastAPI."""

from __future__ import annotations

import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class RateLimiter:
    """Sliding-window rate limiter keyed by client IP."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._max = max_requests
        self._window = window_seconds

    def check(self, client_ip: str) -> bool:
        now = time.monotonic()
        timestamps = self._requests[client_ip]
        self._requests[client_ip] = [t for t in timestamps if now - t < self._window]
        if len(self._requests[client_ip]) >= self._max:
            return False
        self._requests[client_ip].append(now)
        return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that rejects requests over the rate limit."""

    def __init__(self, app: object, max_requests: int = 60, window_seconds: int = 60) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._limiter = RateLimiter(max_requests=max_requests, window_seconds=window_seconds)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        if not self._limiter.check(client_ip):
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please slow down."},
            )
        return await call_next(request)
