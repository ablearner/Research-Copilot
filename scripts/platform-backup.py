#!/usr/bin/env python3
"""Offline backup/restore for the local Compose profile, with SHA256 verification.
Stop all API/worker/scheduler/web writers before using --quiesced.
Restore refuses any nonempty target volume; it never overwrites an installation.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import shutil

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('platform_local', ROOT / 'scripts/platform-local.py')
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)
IMAGE = 'mysql:8.4'

def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)

def volume(service, destination):
    cid = run(local.compose('ps', '-q', service), capture_output=True, text=True).stdout.strip()
    if not cid:
        # docker-compose v1 ps -q includes stopped containers; fail closed otherwise.
        raise RuntimeError(f'Compose container unavailable: {service}')
    mounts = json.loads(run(['docker', 'inspect', cid], capture_output=True, text=True).stdout)[0]['Mounts']
    return next(m['Name'] for m in mounts if m['Destination'] == destination and m['Type'] == 'volume')

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['backup', 'verify', 'restore'])
    parser.add_argument('directory', type=Path)
    parser.add_argument('--quiesced', action='store_true')
    args = parser.parse_args()
    folder = args.directory.resolve()
    from apps.platform.config import PlatformSettings
    configured_root = Path(PlatformSettings().worker_root)
    if not configured_root.is_absolute():
        configured_root = ROOT / configured_root
    worker_root = configured_root.resolve()
    if args.action == 'verify':
        manifest = json.loads((folder / 'manifest.json').read_text())
        for name, checksum in manifest['sha256'].items():
            if Path(name).name != name or sha(folder / name) != checksum:
                raise RuntimeError(f'Backup checksum mismatch: {name}')
        print('Backup checksums verified')
        return
    if not args.quiesced:
        parser.error('Stop all application writers, then explicitly pass --quiesced')
    if args.action == 'backup':
        folder.mkdir(mode=0o700, parents=True, exist_ok=False)
        run(local.compose('stop'))
        for service, path in [('mysql', '/var/lib/mysql'), ('minio', '/data')]:
            name = volume(service, path)
            with (folder / f'{service}.tar').open('wb') as output:
                run(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'tar',
                     '-v', f'{name}:/source:ro', IMAGE, '-C', '/source', '-cf', '-', '.'], stdout=output)
        with tarfile.open(folder / 'worker.tar', 'w') as archive:
            if worker_root.exists():
                archive.add(worker_root, arcname='platform-worker')
        manifest = {'format': 1, 'mysql_image': IMAGE,
                    'sha256': {name: sha(folder / name) for name in ['mysql.tar', 'minio.tar', 'worker.tar']}}
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print('Backup complete; infrastructure remains stopped. Config/secrets must be escrowed separately.')
        return
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest.get('format') != 1 or manifest.get('mysql_image') != IMAGE:
        raise RuntimeError('Unsupported backup format/image')
    for name in ['mysql.tar', 'minio.tar', 'worker.tar']:
        if sha(folder / name) != manifest['sha256'][name]:
            raise RuntimeError(f'Backup checksum mismatch: {name}')
    if worker_root.exists() and any(worker_root.iterdir()):
        raise RuntimeError('Restore needs an empty .data/platform-worker directory')
    run(local.compose('create'))
    run(local.compose('stop'))
    volumes = [(service, volume(service, path)) for service, path in [('mysql', '/var/lib/mysql'), ('minio', '/data')]]
    # Verify every target before writing any of them.
    for _, name in volumes:
        run(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'sh', '-v', f'{name}:/target', IMAGE,
             '-c', 'test -z "$(ls -A /target)"'])
    for service, name in volumes:
        with (folder / f'{service}.tar').open('rb') as source:
            run(['docker', 'run', '--rm', '-i', '--network', 'none', '--entrypoint', 'tar',
                 '-v', f'{name}:/target', IMAGE, '-C', '/target', '-xf', '-'], stdin=source)
    with tempfile.TemporaryDirectory(prefix="platform-restore-") as staging:
        with tarfile.open(folder / 'worker.tar') as archive:
            archive.extractall(staging, filter='data')
        restored = Path(staging) / 'platform-worker'
        worker_root.parent.mkdir(parents=True, exist_ok=True)
        if worker_root.exists():
            worker_root.rmdir()
        if restored.exists():
            shutil.move(str(restored), str(worker_root))
        else:
            worker_root.mkdir()
    print('Restored; start infrastructure, migrate head, run deploy/platform-replay-outbox.sql before starting workers. See runbook.')

if __name__ == '__main__':
    main()
