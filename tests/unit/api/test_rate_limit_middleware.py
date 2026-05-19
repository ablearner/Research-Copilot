from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from apps.api.middleware.rate_limit import RateLimitMiddleware


async def ok_endpoint(request):
    return PlainTextResponse("ok")


def test_rate_limit_middleware_can_be_disabled() -> None:
    app = Starlette(routes=[Route("/ping", ok_endpoint)])
    app.add_middleware(RateLimitMiddleware, max_requests=1, window_seconds=60, enabled=False)
    client = TestClient(app)

    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 200


def test_rate_limit_middleware_skips_options_requests() -> None:
    app = Starlette(routes=[Route("/ping", ok_endpoint, methods=["GET", "OPTIONS"])])
    app.add_middleware(RateLimitMiddleware, max_requests=1, window_seconds=60, enabled=True)
    client = TestClient(app)

    assert client.options("/ping").status_code == 200
    assert client.options("/ping").status_code == 200
    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 429
