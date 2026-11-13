"""Isolated PDF parsing process. No network, model, queue, or database work."""

import asyncio
import resource
import sys
from pathlib import Path


async def main():
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_FSIZE, (512 * 1024 * 1024, 512 * 1024 * 1024))
    import fitz
    from adapters.local_runtime import LocalDocumentParser

    source, document_id, storage, output = sys.argv[1:]
    with fitz.open(source) as pdf:
        if pdf.page_count > 500:
            raise ValueError("PDF exceeds 500-page processing limit")
    parsed = await LocalDocumentParser(storage_root=storage).parse_document(source, document_id)
    Path(output).write_text(parsed.model_dump_json())


if __name__ == "__main__":
    asyncio.run(main())
