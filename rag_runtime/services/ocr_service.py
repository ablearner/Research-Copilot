from typing import Protocol, runtime_checkable

from domain.schemas.document import DocumentPage, TextBlock


@runtime_checkable
class OcrService(Protocol):
    async def extract_text_blocks(self, page: DocumentPage) -> list[TextBlock]:
        raise NotImplementedError
