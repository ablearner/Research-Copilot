from __future__ import annotations

from typing import Any

from mcp.mapping import map_resource_to_mcp_resource
from mcp.schemas import MCPResourceContent, MCPResourceSpec


class MCPResourceAdapter:
    def __init__(self) -> None:
        self._document_summaries: dict[str, dict[str, Any]] = {}
        self._chart_summaries: dict[str, dict[str, Any]] = {}
        self._graph_community_summaries: dict[str, dict[str, Any]] = {}
        self._schema_info: dict[str, dict[str, Any]] = {}
        self._config_info: dict[str, dict[str, Any]] = {}

    def set_document_summary(self, document_id: str, summary: str, metadata: dict[str, Any] | None = None) -> None:
        self._document_summaries[document_id] = {"summary": summary, "metadata": metadata or {}}

    def set_chart_summary(self, chart_id: str, summary: str, metadata: dict[str, Any] | None = None) -> None:
        self._chart_summaries[chart_id] = {"summary": summary, "metadata": metadata or {}}

    def set_graph_community_summary(
        self,
        community_id: str,
        summary: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._graph_community_summaries[community_id] = {"summary": summary, "metadata": metadata or {}}

    def set_schema_info(self, schema_name: str, schema_payload: dict[str, Any]) -> None:
        self._schema_info[schema_name] = schema_payload

    def set_config_info(self, config_name: str, config_payload: dict[str, Any]) -> None:
        self._config_info[config_name] = config_payload

    def list_resources(self) -> list[MCPResourceSpec]:
        resources: list[MCPResourceSpec] = []
        for document_id in sorted(self._document_summaries.keys()):
            resources.append(
                map_resource_to_mcp_resource(
                    resource_type="document_summary",
                    resource_id=document_id,
                    name=f"document_summary:{document_id}",
                    description="Read-only document summary resource",
                )
            )
        for chart_id in sorted(self._chart_summaries.keys()):
            resources.append(
                map_resource_to_mcp_resource(
                    resource_type="chart_summary",
                    resource_id=chart_id,
                    name=f"chart_summary:{chart_id}",
                    description="Read-only chart summary resource",
                )
            )
        for community_id in sorted(self._graph_community_summaries.keys()):
            resources.append(
                map_resource_to_mcp_resource(
                    resource_type="graph_community_summary",
                    resource_id=community_id,
                    name=f"graph_community_summary:{community_id}",
                    description="Read-only graph community summary resource",
                )
            )
        for schema_name in sorted(self._schema_info.keys()):
            resources.append(
                map_resource_to_mcp_resource(
                    resource_type="schema",
                    resource_id=schema_name,
                    name=f"schema:{schema_name}",
                    description="Read-only schema information",
                )
            )
        for config_name in sorted(self._config_info.keys()):
            resources.append(
                map_resource_to_mcp_resource(
                    resource_type="config",
                    resource_id=config_name,
                    name=f"config:{config_name}",
                    description="Read-only config information",
                )
            )
        return resources

    def read_resource(self, uri: str) -> MCPResourceContent:
        prefix = "mcp://resource/"
        if not uri.startswith(prefix):
            return MCPResourceContent(
                uri=uri,
                mime_type="application/json",
                content={"error": "invalid_resource_uri"},
                read_only=True,
                metadata={"exists": False},
            )

        remainder = uri[len(prefix):]
        parts = remainder.split("/", 1)
        if len(parts) != 2:
            return MCPResourceContent(
                uri=uri,
                mime_type="application/json",
                content={"error": "invalid_resource_uri"},
                read_only=True,
                metadata={"exists": False},
            )

        resource_type, resource_id = parts
        payload = self._lookup(resource_type, resource_id)
        exists = payload is not None
        return MCPResourceContent(
            uri=uri,
            mime_type="application/json",
            content=payload if exists else {"error": "resource_not_found"},
            read_only=True,
            metadata={"exists": exists, "resource_type": resource_type, "resource_id": resource_id},
        )

    def _lookup(self, resource_type: str, resource_id: str) -> Any | None:
        if resource_type == "document_summary":
            return self._document_summaries.get(resource_id)
        if resource_type == "chart_summary":
            return self._chart_summaries.get(resource_id)
        if resource_type == "graph_community_summary":
            return self._graph_community_summaries.get(resource_id)
        if resource_type == "schema":
            return self._schema_info.get(resource_id)
        if resource_type == "config":
            return self._config_info.get(resource_id)
        return None
