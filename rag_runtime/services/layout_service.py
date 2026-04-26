from typing import Protocol, runtime_checkable

from domain.schemas.document import DocumentPage


@runtime_checkable
class LayoutService(Protocol):
    async def locate_chart_candidates(self, page: DocumentPage) -> list[dict]:
        raise NotImplementedError
