import argparse
import json
import socket
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import Settings


def _can_connect(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Research-Copilot local environment.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    settings = Settings()
    checks = {
        "research_storage_root": settings.resolve_path(settings.research_storage_root).exists(),
        "upload_dir_parent": settings.resolve_path(settings.upload_dir).parent.exists(),
        "neo4j_bolt": _can_connect("127.0.0.1", 7687) if settings.neo4j_uri else False,
        "milvus_http": _can_connect("127.0.0.1", 19530) if settings.milvus_uri else False,
        "llm_api_key_present": bool(settings.dashscope_api_key or settings.openai_api_key),
    }
    payload = {
        "app_env": settings.app_env,
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
        "checks": checks,
        "all_required_passed": checks["research_storage_root"] and checks["upload_dir_parent"],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for name, value in payload["checks"].items():
            print(f"{name}: {'ok' if value else 'missing'}")
        print(f"all_required_passed: {payload['all_required_passed']}")
    return 0 if payload["all_required_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
