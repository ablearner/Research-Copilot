"""Thin online client. Token comes from RESEARCH_PLATFORM_TOKEN, never command history."""

import argparse
import json
import os
from uuid import uuid4
import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("workspaces")
    projects = sub.add_parser("projects")
    projects.add_argument("workspace")
    task = sub.add_parser("task")
    task.add_argument("workspace")
    task.add_argument("task_id")
    ask = sub.add_parser("ask")
    ask.add_argument("workspace")
    ask.add_argument("conversation")
    ask.add_argument("message")
    ask.add_argument("--document", action="append", default=[])
    ask.add_argument("--idempotency-key", default=None)
    args = parser.parse_args()
    token = os.environ.get("RESEARCH_PLATFORM_TOKEN")
    if not token:
        parser.error("Set RESEARCH_PLATFORM_TOKEN to a scoped personal credential")
    with httpx.Client(
        base_url=args.url.rstrip("/") + "/api/v1/",
        headers={"Authorization": "Bearer " + token},
        timeout=30,
    ) as client:
        if args.command == "workspaces":
            response = client.get("workspaces")
        elif args.command == "projects":
            response = client.get(f"workspaces/{args.workspace}/projects")
        elif args.command == "task":
            response = client.get(f"workspaces/{args.workspace}/tasks/{args.task_id}")
        else:
            response = client.post(
                f"workspaces/{args.workspace}/conversations/{args.conversation}/messages",
                json={"message": args.message, "document_ids": args.document},
                headers={"Idempotency-Key": args.idempotency_key or str(uuid4())},
            )
        response.raise_for_status()
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
