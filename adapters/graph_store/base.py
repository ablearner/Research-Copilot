from abc import ABC, abstractmethod

from domain.schemas.graph import GraphEdge, GraphNode, GraphQueryRequest, GraphQueryResult, GraphTriple


class GraphStoreError(RuntimeError):
    """Raised when a graph store operation fails."""


class BaseGraphStore(ABC):
    @abstractmethod
    async def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def upsert_nodes(self, nodes: list[GraphNode]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def upsert_edges(self, edges: list[GraphEdge]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def upsert_triples(self, triples: list[GraphTriple]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def query_subgraph(self, query_request: GraphQueryRequest) -> GraphQueryResult:
        raise NotImplementedError

    @abstractmethod
    async def get_neighbors(self, node_id: str, depth: int) -> GraphQueryResult:
        raise NotImplementedError

    @abstractmethod
    async def search_entities(
        self,
        keyword: str,
        document_ids: list[str] | None = None,
    ) -> GraphQueryResult:
        raise NotImplementedError
