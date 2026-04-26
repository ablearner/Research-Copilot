from typing import Protocol, runtime_checkable

from domain.schemas.document import ParsedDocument


@runtime_checkable
class PdfService(Protocol):
    async def parse_document(self, file_path: str, document_id: str | None = None) -> ParsedDocument:
        raise NotImplementedError
