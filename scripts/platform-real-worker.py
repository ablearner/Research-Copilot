"""Test-only Worker observer: record provider metadata without prompts or credentials."""

import json
import os
import runpy
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
from core.config import Settings

s = Settings()
hosts = {
    urlsplit(x).hostname
    for x in [s.openai_base_url, s.chart_vision_base_url, s.embedding_base_url]
    if x
}
send = httpx.AsyncClient.send


async def observed(self, request, *args, **kwargs):
    started = time.monotonic()
    if request.url.host not in hosts:
        return await send(self, request, *args, **kwargs)
    response = None
    error = None
    try:
        response = await send(self, request, *args, **kwargs)
        return response
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        row = {
            "host": request.url.host,
            "path": request.url.path,
            "status": response.status_code if response is not None else None,
            "error": error,
            "recorded_at": time.time(),
            "seconds": round(time.monotonic() - started, 3),
        }
        try:
            body = json.loads(request.content)
            row["model"] = body.get("model")
            row["image_parts"] = sum(
                1
                for m in body.get("messages", [])
                if isinstance(m.get("content"), list)
                for part in m["content"]
                if part.get("type") == "image_url"
            )
        except Exception:
            pass
        try:
            row["usage"] = response.json().get("usage")
        except Exception:
            pass
        with open(os.environ["REAL_E2E_CALLS"], "a") as f:
            f.write(json.dumps(row) + "\n")


httpx.AsyncClient.send = observed
runpy.run_module("apps.worker", run_name="__main__")
