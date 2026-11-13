import asyncio
from pathlib import Path


class LocalObjectStore:
    """Private local adapter for tests/offline development, never a public static directory."""

    def __init__(self, root):
        self.root = Path(root).resolve()

    def _path(self, key):
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Invalid object key")
        return path

    async def put(self, key, data, content_type="application/octet-stream"):
        path = self._path(key)
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, data)

    async def get(self, key):
        return await asyncio.to_thread(self._path(key).read_bytes)

    async def delete(self, key):
        await asyncio.to_thread(self._path(key).unlink, missing_ok=True)
