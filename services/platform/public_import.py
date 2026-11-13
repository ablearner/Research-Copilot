"""Import public arXiv identifiers through a fixed origin, never caller-supplied URLs."""

import re
import httpx


def arxiv_pdf_url(identifier):
    if not isinstance(identifier, str) or not re.fullmatch(
        r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", identifier
    ):
        raise ValueError("A valid arXiv identifier is required for automatic download")
    return "https://arxiv.org/pdf/" + identifier


async def download_arxiv(identifier, max_bytes):
    async with httpx.AsyncClient(timeout=60, follow_redirects=False) as client:
        async with client.stream("GET", arxiv_pdf_url(identifier)) as response:
            response.raise_for_status()
            data = bytearray()
            async for block in response.aiter_bytes():
                data.extend(block)
                if len(data) > max_bytes:
                    raise ValueError("Public PDF exceeds upload limit")
            return bytes(data)
