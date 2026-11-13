#!/usr/bin/env python3
"""Foreground local platform supervisor; Ctrl-C stops every child process group."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def compose(*args):
    prefix = ['docker', 'compose']
    if subprocess.run(prefix + ['version'], capture_output=True).returncode:
        prefix = ['docker-compose']
    return prefix + ['-p', os.environ.get('PLATFORM_COMPOSE_PROJECT', 'research-copilot-platform'),
                     '-f', str(ROOT / 'compose.platform.yml'),
                     '-f', str(ROOT / 'deploy/platform-mail.compose.yml')] + list(args)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['infra', 'migrate', 'run', 'doctor'])
    parser.add_argument('--offline', action='store_true', help='Local model and memory vector providers; no real LLM quality claim')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == 'infra':
        subprocess.run(compose('up', '-d'), check=True)
        return
    if args.action == 'migrate':
        # Alembic reads environment directly, unlike PlatformSettings.
        from apps.platform.config import PlatformSettings
        env = dict(os.environ, PLATFORM_DATABASE_URL=PlatformSettings().database_url)
        subprocess.run([sys.executable, '-m', 'alembic', '-c', 'adapters/storage/mysql/alembic.ini', 'upgrade', 'head'], env=env, check=True)
        return
    if args.action == 'doctor':
        subprocess.run(compose('ps'), check=True)
        subprocess.run([sys.executable, '-c', 'import fastapi, sqlalchemy, asyncmy, aio_pika, boto3; print("Python dependencies import OK")'], check=True)
        subprocess.run(['node', '--version'], check=True)
        return
    env = dict(os.environ)
    if args.offline:
        env.update(RUNTIME_BACKEND='local', LLM_PROVIDER='local', EMBEDDING_PROVIDER='local',
                   VECTOR_STORE_PROVIDER='memory', GRAPH_STORE_PROVIDER='memory',
                   RERANKER_ALLOW_DOWNLOAD='false', RERANKER_UNAVAILABLE_POLICY='heuristic')
    commands = [
        ([sys.executable, '-m', 'uvicorn', 'apps.platform.main:app', '--host', '127.0.0.1', '--port', '8000'], ROOT),
        ([sys.executable, '-m', 'apps.worker'], ROOT),
        ([sys.executable, '-m', 'apps.scheduler'], ROOT),
        (['npm', 'run', 'start'], ROOT / 'web'),
    ]
    children = []
    stopping = False
    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        for command, cwd in commands:
            children.append(subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True))
        print('Platform running: http://localhost:3000; mailbox: http://localhost:58025', flush=True)
        while not stopping:
            for process in children:
                if process.poll() is not None:
                    raise RuntimeError(f'Child {process.pid} exited with {process.returncode}')
            time.sleep(0.5)
    finally:
        for process in children:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 40
        for process in children:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

if __name__ == '__main__':
    main()
