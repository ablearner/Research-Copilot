#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_DIR="${PROJECT_ROOT}/.data/runtime"
PID_FILE="${RUNTIME_DIR}/zotero-bridge.pid"
LOG_FILE="${RUNTIME_DIR}/zotero-bridge.log"

LOCAL_HOST="${LOCAL_HOST:-127.0.0.1}"
LOCAL_PORT="${LOCAL_PORT:-23119}"
REMOTE_PORT="${REMOTE_PORT:-23119}"

mkdir -p "${RUNTIME_DIR}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <start|stop|restart|status> [remote_host]

Environment overrides:
  LOCAL_HOST   Local bind host (default: 127.0.0.1)
  LOCAL_PORT   Local bind port (default: 23119)
  REMOTE_PORT  Remote Zotero connector port (default: 23119)

Examples:
  $(basename "$0") start
  $(basename "$0") start 172.23.160.1
  $(basename "$0") status
EOF
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

http_probe() {
  local host="$1"
  local port="$2"
  python3 - <<PY
import http.client
import sys

host = ${host@Q}
port = int(${port@Q})

try:
    conn = http.client.HTTPConnection(host, port, timeout=2)
    conn.request("GET", "/connector/ping")
    response = conn.getresponse()
    print(response.status)
except Exception:
    print("ERR")
finally:
    try:
        conn.close()
    except Exception:
        pass
PY
}

resolve_windows_host() {
  if [[ $# -ge 1 && -n "${1}" ]]; then
    printf '%s\n' "$1"
    return 0
  fi

  local gateway=""
  gateway="$(ip route show default 2>/dev/null | awk '/default/ {print $3; exit}')"
  if [[ -n "${gateway}" ]]; then
    printf '%s\n' "${gateway}"
    return 0
  fi

  echo "unable to detect Windows host IP automatically; pass it explicitly" >&2
  exit 1
}

is_pid_running() {
  local pid="$1"
  kill -0 "${pid}" >/dev/null 2>&1
}

pid_from_file() {
  if [[ -f "${PID_FILE}" ]]; then
    tr -d '[:space:]' < "${PID_FILE}"
  fi
}

find_existing_bridge_pid() {
  pgrep -f "socat TCP-LISTEN:${LOCAL_PORT},bind=${LOCAL_HOST},reuseaddr,fork TCP:.*:${REMOTE_PORT}" | head -n 1 || true
}

current_status() {
  local pid=""
  pid="$(pid_from_file)"
  if [[ -n "${pid}" && "$(is_pid_running "${pid}"; echo $?)" -eq 0 ]]; then
    echo "running pid=${pid}"
    return 0
  fi

  pid="$(find_existing_bridge_pid)"
  if [[ -n "${pid}" ]]; then
    echo "${pid}" > "${PID_FILE}"
    echo "running pid=${pid}"
    return 0
  fi

  echo "stopped"
  return 1
}

start_bridge() {
  require_command socat
  local remote_host="$1"

  if current_status >/dev/null 2>&1; then
    echo "bridge already running: $(current_status)"
    return 0
  fi

  nohup socat \
    "TCP-LISTEN:${LOCAL_PORT},bind=${LOCAL_HOST},reuseaddr,fork" \
    "TCP:${remote_host}:${REMOTE_PORT}" \
    >> "${LOG_FILE}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${PID_FILE}"

  sleep 1
  if ! is_pid_running "${pid}"; then
    echo "failed to start bridge; check ${LOG_FILE}" >&2
    rm -f "${PID_FILE}"
    exit 1
  fi

  local probe_status=""
  probe_status="$(http_probe "${LOCAL_HOST}" "${LOCAL_PORT}")"
  if [[ "${probe_status}" == "ERR" ]]; then
    echo "warning: bridge started but ping probe failed; check Windows Zotero connector state" >&2
  else
    echo "probe_status=${probe_status}"
  fi

  echo "started pid=${pid} local=${LOCAL_HOST}:${LOCAL_PORT} remote=${remote_host}:${REMOTE_PORT}"
}

stop_bridge() {
  local pid=""
  pid="$(pid_from_file)"
  if [[ -z "${pid}" ]]; then
    pid="$(find_existing_bridge_pid)"
  fi

  if [[ -z "${pid}" ]]; then
    echo "bridge already stopped"
    rm -f "${PID_FILE}"
    return 0
  fi

  if is_pid_running "${pid}"; then
    kill "${pid}"
    sleep 1
  fi

  if is_pid_running "${pid}"; then
    echo "bridge did not stop cleanly; sending SIGKILL to pid=${pid}"
    kill -9 "${pid}"
  fi

  rm -f "${PID_FILE}"
  echo "stopped pid=${pid}"
}

command="${1:-}"
case "${command}" in
  start)
    start_bridge "$(resolve_windows_host "${2:-}")"
    ;;
  stop)
    stop_bridge
    ;;
  restart)
    stop_bridge
    start_bridge "$(resolve_windows_host "${2:-}")"
    ;;
  status)
    current_status
    ;;
  *)
    usage
    exit 1
    ;;
esac
