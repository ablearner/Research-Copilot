"""Real HTTP -> Outbox -> Worker -> remote models -> persisted answer acceptance."""

import asyncio
import hashlib
import json
import re
import secrets
import time
import sys
import os
import argparse
import signal
import subprocess
from pathlib import Path

import fitz
import httpx
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / ".data/real-e2e"
ROOT.mkdir(parents=True, exist_ok=True)


async def main():
    report = {"started_at": time.time(), "steps": [], "tasks": []}
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:58000/api/v1", timeout=30, trust_env=False
    ) as c:

        async def req(method, path, **kw):
            headers = kw.pop("headers", {})
            if c.cookies.get("rc_csrf"):
                headers["X-CSRF-Token"] = c.cookies["rc_csrf"]
            r = await c.request(method, path, headers=headers, **kw)
            if r.status_code >= 400:
                raise RuntimeError(f"{method} {path}: {r.status_code} {r.text[:400]}")
            return r.json() if r.content else None

        async def account(client_email):
            password = secrets.token_urlsafe(18)
            await req("POST", "/auth/register", json={"email": client_email, "password": password})
            async with httpx.AsyncClient(
                base_url="http://127.0.0.1:58025", trust_env=False
            ) as mail:
                for _ in range(20):
                    entries = (await mail.get("/api/v1/messages")).json()["messages"]
                    msg = next(
                        (m for m in entries if any(r["Address"] == client_email for r in m["To"])),
                        None,
                    )
                    if msg:
                        break
                    await asyncio.sleep(0.5)
                content = (await mail.get("/api/v1/message/" + msg["ID"])).json()["Text"]
                token = re.search(r"token=([A-Za-z0-9_-]+)", content).group(1)
            await req("POST", "/auth/verify/confirm", json={"token": token})
            return await req(
                "POST", "/auth/login", json={"email": client_email, "password": password}
            )

        email = "real-e2e-" + secrets.token_hex(5) + "@example.test"
        login = await account(email)
        wid = login["workspace"]["id"]
        wp = f"/workspaces/{wid}"
        project = await req(
            "POST", wp + "/projects", json={"name": "Real provider E2E synthetic evidence"}
        )
        pp = wp + "/projects/" + project["id"]
        report.update(workspace_id=wid, project_id=project["id"])
        report["steps"].append("register_email_verify_login_project")
        img = Image.new("RGB", (640, 420), "white")
        d = ImageDraw.Draw(img)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
        d.text((30, 15), "Figure 1: Measured score", fill="black", font=font)
        for x, value, color, name in [(100, 37, "royalblue", "Alpha"), (350, 82, "orange", "Beta")]:
            d.rectangle((x, 340 - value * 3, x + 100, 340), fill=color)
            d.text((x, 300 - value * 3), str(value), fill="black", font=font)
            d.text((x, 360), name, fill="black", font=font)
        image_path = ROOT / "chart.png"
        img.save(image_path)
        pdf = fitz.open()
        page = pdf.new_page(width=640, height=620)
        page.insert_text(
            (25, 35), "Synthetic research evidence for API end-to-end validation.", fontsize=16
        )
        page.insert_text(
            (25, 65), "Method Orion achieved accuracy 92 percent on dataset TestSet.", fontsize=14
        )
        page.insert_text(
            (25, 90),
            "Figure 1 compares Alpha and Beta. Read numerical scores from the chart.",
            fontsize=12,
        )
        page.insert_image(fitz.Rect(0, 130, 640, 550), filename=str(image_path))
        data = pdf.tobytes()
        pdf.close()
        (ROOT / "evidence.pdf").write_bytes(data)
        grant = await req(
            "POST",
            pp + "/uploads/init",
            json={
                "filename": "real-evidence.pdf",
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            },
        )
        async with httpx.AsyncClient(trust_env=False) as objects:
            upload = await objects.post(
                grant["url"],
                data=grant["fields"],
                files={"file": ("real-evidence.pdf", data, "application/pdf")},
            )
            upload.raise_for_status()
        doc = await req("POST", wp + "/uploads/" + grant["id"] + "/complete")
        report["document"] = doc

        async def wait(task):
            started = time.monotonic()
            while time.monotonic() - started < 650:
                state = await req("GET", wp + "/tasks/" + task["id"])
                if state["status"] in ["succeeded", "failed", "cancelled", "partially_succeeded"]:
                    events = (await c.get(wp + "/tasks/" + task["id"] + "/events")).text
                    (ROOT / (task["id"] + "-events.txt")).write_text(events)
                    report["tasks"].append(
                        {
                            "id": task["id"],
                            "status": state["status"],
                            "seconds": round(time.monotonic() - started, 2),
                            "error_code": state.get("error_code"),
                        }
                    )
                    print(json.dumps(report["tasks"][-1]), flush=True)
                    if state["status"] not in ["succeeded", "partially_succeeded"]:
                        raise RuntimeError("Task failed: " + str(state.get("error_code")))
                    return state
                await asyncio.sleep(3)
            await req("POST", wp + "/tasks/" + task["id"] + "/cancel")
            raise TimeoutError("task deadline")

        try:
            task = await req(
                "POST",
                pp + "/imports",
                json={"document_ids": [doc["document_id"]]},
                headers={"Idempotency-Key": secrets.token_hex(16)},
            )
            await wait(task)
            report["steps"].append("signed_upload_queue_parse_embed_milvus")
            conv = await req("POST", pp + "/conversations")
            cid = conv["id"]
            report["conversation_id"] = cid
            for label, question, expected in [
                (
                    "text",
                    "Using only the selected paper, what accuracy did Method Orion achieve on TestSet? Cite the paper and answer concisely.",
                    ["92"],
                ),
                (
                    "vision",
                    "Inspect Figure 1 in the selected paper using image understanding, not guesses from surrounding text. What numerical scores are printed above Alpha and Beta, and which is larger?",
                    ["37", "82"],
                ),
            ]:
                task = await req(
                    "POST",
                    wp + "/conversations/" + cid + "/messages",
                    json={"message": question, "document_ids": [doc["document_id"]]},
                    headers={"Idempotency-Key": secrets.token_hex(16)},
                )
                await wait(task)
                messages = await req("GET", wp + "/conversations/" + cid + "/messages?latest=true")
                answer = messages[-1]["content"]["text"]
                report[label + "_answer"] = answer
                assert all(x in answer for x in expected), (
                    label + " answer missing expected evidence"
                )
                report["steps"].append(label + "_answer_persisted")
                (ROOT / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            saved = await req("GET", wp + "/conversations/" + cid + "/messages?latest=true")
            async with httpx.AsyncClient(
                base_url=c.base_url, cookies=c.cookies, trust_env=False
            ) as fresh:
                reload = await fresh.get(wp + "/conversations/" + cid + "/messages?latest=true")
                assert reload.json() == saved
            report["steps"].append("new_http_client_history_restored")
            reports = await req("GET", pp + "/reports")
            assert reports and "92" in json.dumps(reports), "Missing persisted research report"
            report["steps"].append("research_report_persisted")
            download = await c.get(wp + "/files/" + doc["file_id"] + "/content")
            assert download.content == data
            other = await account("stranger-" + secrets.token_hex(5) + "@example.test")
            assert other["workspace"]["id"] != wid
            denied = await c.get(wp + "/files/" + doc["file_id"] + "/content")
            assert denied.status_code == 404
            report["steps"].append("cross_user_file_denied")
            calls = [
                json.loads(line)
                for line in (ROOT / "provider-calls.jsonl").read_text().splitlines()
            ]
            calls = [row for row in calls if row.get("recorded_at", 0) >= report["started_at"]]
            report["provider_calls"] = calls
            assert any(row.get("image_parts", 0) > 0 and row["status"] == 200 for row in calls), (
                "No real image request observed"
            )
            assert any(
                row.get("model", "").startswith("deepseek") and row["status"] == 200
                for row in calls
            ), "No real DeepSeek call observed"
            assert any(
                row["path"].endswith("/embeddings") and row["status"] == 200 for row in calls
            ), "No real embedding request observed"
            report["success"] = True
        except Exception as exc:
            report.update(success=False, error=type(exc).__name__ + ": " + str(exc))
            raise
        finally:
            report["finished_at"] = time.time()
            (ROOT / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


async def verify_restored():
    """Run after restarting serve; recover only this harness's disposable account."""
    sys.path.insert(0, str(REPO))
    from sqlalchemy import select
    from adapters.storage.mysql import Database
    from adapters.storage.mysql import tables as t
    from apps.platform.config import PlatformSettings

    report = json.loads((ROOT / "result.json").read_text())
    assert report.get("success"), "Complete test before checking restart persistence"
    db = Database(PlatformSettings().database_url)
    try:
        async with db.transaction() as conn:
            email = (
                await conn.execute(
                    select(t.users.c.email)
                    .join(t.workspace_members, t.workspace_members.c.user_id == t.users.c.id)
                    .where(t.workspace_members.c.workspace_id == report["workspace_id"])
                )
            ).scalar_one()
    finally:
        await db.close()
    assert email.startswith("real-e2e-") and email.endswith("@example.test")
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:58000/api/v1", trust_env=False, timeout=30
    ) as client:
        response = await client.post("/auth/reset/request", json={"email": email})
        response.raise_for_status()
        async with httpx.AsyncClient(base_url="http://127.0.0.1:58025", trust_env=False) as mail:
            entries = (await mail.get("/api/v1/messages")).json()["messages"]
            msg = next(m for m in entries if any(r["Address"] == email for r in m["To"]))
            content = (await mail.get("/api/v1/message/" + msg["ID"])).json()["Text"]
            token = re.search(r"token=([A-Za-z0-9_-]+)", content).group(1)
        password = secrets.token_urlsafe(18)
        response = await client.post(
            "/auth/reset/confirm", json={"token": token, "password": password}
        )
        response.raise_for_status()
        response = await client.post("/auth/login", json={"email": email, "password": password})
        response.raise_for_status()
        wp = "/workspaces/" + report["workspace_id"]
        response = await client.get(
            wp + "/conversations/" + report["conversation_id"] + "/messages?latest=true"
        )
        response.raise_for_status()
        answers = [
            row["content"].get("text") for row in response.json() if row["role"] == "assistant"
        ]
        assert report["text_answer"] in answers and report["vision_answer"] in answers
        response = await client.get(wp + "/projects/" + report["project_id"] + "/reports")
        response.raise_for_status()
        reports = response.json()
        assert reports and "92" in json.dumps(reports), "Research report did not survive restart"
        response = await client.get(wp + "/files/" + report["document"]["file_id"] + "/content")
        response.raise_for_status()
        assert response.content == (ROOT / "evidence.pdf").read_bytes()
        report["restart_verification"] = {
            "verified_at": time.time(),
            "success": True,
            "checks": [
                "password_reset_login",
                "text_and_vision_history",
                "research_report",
                "original_pdf",
            ],
            "report_count": len(reports),
        }
        (ROOT / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report["restart_verification"]), flush=True)


def serve():
    os.chdir(REPO)
    env = dict(
        os.environ,
        PLATFORM_PUBLIC_ORIGIN="http://127.0.0.1:58000",
        PLATFORM_REQUIRE_EMAIL_VERIFICATION="true",
        PLATFORM_RATE_LIMIT_ENABLED="true",
        PLATFORM_WORKER_ROOT=str(ROOT / "worker"),
        PLATFORM_TASK_TIMEOUT_SECONDS="600",
        GRAPH_TEXT_TIMEOUT_SECONDS="60",
        OPENAI_MAX_RETRIES="0",
        CHART_VISION_MAX_RETRIES="0",
        RERANKER_ALLOW_DOWNLOAD="false",
        REAL_E2E_CALLS=str(ROOT / "provider-calls.jsonl"),
    )
    commands = [
        (
            "api",
            [
                sys.executable,
                "-m",
                "uvicorn",
                "apps.platform.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "58000",
            ],
        ),
        ("worker", [sys.executable, str(REPO / "scripts/platform-real-worker.py")]),
        ("scheduler", [sys.executable, "-m", "apps.scheduler"]),
    ]
    children = []
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        for name, command in commands:
            with (ROOT / (name + ".log")).open("w") as log:
                children.append(
                    subprocess.Popen(
                        command,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
        print(
            "Real-provider test services at http://127.0.0.1:58000; Ctrl-C stops children",
            flush=True,
        )
        while not stopping:
            if any(p.poll() is not None for p in children):
                raise RuntimeError("A test service exited; inspect .data/real-e2e logs")
            time.sleep(1)
    finally:
        for p in children:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in children:
            try:
                p.wait(timeout=35)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Opt-in real-provider E2E on dedicated local services. Uses paid API calls and synthetic PDF. Keeps test accounts/artifacts; never use a production database."
    )
    parser.add_argument("action", choices=["serve", "test", "verify-restored"])
    args = parser.parse_args()
    if args.action == "serve":
        serve()
    elif args.action == "verify-restored":
        asyncio.run(verify_restored())
    else:
        asyncio.run(main())
