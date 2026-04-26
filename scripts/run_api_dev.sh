#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/myc/miniconda3/envs/Research-Copilot/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

RELOAD_DIRS=(
  "apps"
  "adapters"
  "agents"
  "chains"
  "core"
  "domain"
  "rag_runtime"
  "mcp"
  "memory"
  "planners"
  "prompts"
  "repositories"
  "retrieval"
  "services"
  "skills"
  "tooling"
  "tools"
)

ARGS=(
  -m uvicorn
  apps.api.main:app
  --host "$HOST"
  --port "$PORT"
  --reload
  --reload-include "*.py"
  --reload-include "*.md"
  --reload-include "*.yaml"
  --reload-include "*.yml"
)

for dir in "${RELOAD_DIRS[@]}"; do
  ARGS+=(--reload-dir "$ROOT_DIR/$dir")
done

cd "$ROOT_DIR"
exec "$PYTHON_BIN" "${ARGS[@]}"
