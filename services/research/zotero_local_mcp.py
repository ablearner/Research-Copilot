from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
import ipaddress
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from typing import Any
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field, model_validator

from tooling.executor import ToolExecutor
from tooling.registry import ToolRegistry
from tooling.schemas import ToolSpec


class EmptyToolInput(BaseModel):
    pass


class ZoteroCollectionRecord(BaseModel):
    key: str | None = None
    name: str
    parent_collection: str | None = None
    item_count: int | None = Field(default=None, ge=0)


class ZoteroAttachmentRecord(BaseModel):
    key: str | None = None
    title: str
    content_type: str | None = None
    local_path: str | None = None
    url: str | None = None


class ZoteroItemRecord(BaseModel):
    key: str | None = None
    item_type: str
    title: str
    creators: list[str] = Field(default_factory=list)
    abstract: str | None = None
    year: str | None = None
    url: str | None = None
    doi: str | None = None
    collections: list[str] = Field(default_factory=list)
    attachments: list[ZoteroAttachmentRecord] = Field(default_factory=list)


class ZoteroStatusToolOutput(BaseModel):
    status: str
    base_url: str
    message: str = ""


class ZoteroSelectedCollectionToolOutput(BaseModel):
    editable: bool = False
    library_id: int | None = None
    library_name: str | None = None
    collection_id: int | None = None
    collection_key: str | None = None
    collection_name: str | None = None


class ZoteroCollectionListToolInput(BaseModel):
    top_level_only: bool = False


class ZoteroCollectionListToolOutput(BaseModel):
    collections: list[ZoteroCollectionRecord] = Field(default_factory=list)


class ZoteroSearchItemsToolInput(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=50)
    include_attachments: bool = True
    collection_key: str | None = None
    collection_name: str | None = None
    current_collection_only: bool = False

    @model_validator(mode="after")
    def validate_collection_filters(self) -> "ZoteroSearchItemsToolInput":
        scoped_filters = [
            bool(self.collection_key and self.collection_key.strip()),
            bool(self.collection_name and self.collection_name.strip()),
            self.current_collection_only,
        ]
        if sum(scoped_filters) > 1:
            raise ValueError(
                "Use at most one of collection_key, collection_name, or current_collection_only."
            )
        return self


class ZoteroSearchItemsToolOutput(BaseModel):
    items: list[ZoteroItemRecord] = Field(default_factory=list)


class ZoteroItemChildrenToolInput(BaseModel):
    item_key: str = Field(min_length=1)


class ZoteroItemChildrenToolOutput(BaseModel):
    item_key: str
    attachments: list[ZoteroAttachmentRecord] = Field(default_factory=list)


class ZoteroImportPaperToolInput(BaseModel):
    title: str = Field(min_length=1, max_length=1000)
    item_type: str = "journalArticle"
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    year: int | None = Field(default=None, ge=0, le=3000)
    url: str | None = None
    doi: str | None = None
    publication_title: str | None = None
    pdf_url: str | None = None
    collection_name: str | None = None


class ZoteroImportPaperToolOutput(BaseModel):
    status: str
    imported_item_key: str | None = None
    attachment_title: str | None = None
    selected_collection: ZoteroSelectedCollectionToolOutput | None = None
    warnings: list[str] = Field(default_factory=list)


class ZoteroAttachPdfToItemToolInput(BaseModel):
    item_key: str = Field(min_length=1)
    pdf_url: str = Field(min_length=1, max_length=4000)
    title: str | None = None
    source_url: str | None = None


class ZoteroAttachPdfToItemToolOutput(BaseModel):
    status: str
    item_key: str
    attachment_title: str | None = None
    attachment_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class ZoteroLocalServerConfig:
    base_url: str = "http://127.0.0.1:23119"
    user_id: str = "0"
    timeout_seconds: float = 20.0


class ZoteroLocalGateway:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:23119",
        user_id: str = "0",
        timeout_seconds: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_id = str(user_id or "0")
        self.timeout_seconds = timeout_seconds

    def _build_async_client(self, **kwargs: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.timeout_seconds, **kwargs)

    def _is_wsl(self) -> bool:
        try:
            osrelease = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").strip().casefold()
        except OSError:
            return False
        return "microsoft" in osrelease or "wsl" in osrelease

    def _windows_host_gateway(self) -> str | None:
        try:
            for line in Path("/proc/net/route").read_text(encoding="utf-8").splitlines()[1:]:
                columns = line.split()
                if len(columns) < 3:
                    continue
                destination, gateway = columns[1], columns[2]
                if destination != "00000000":
                    continue
                gateway_bytes = bytes.fromhex(gateway)
                gateway_ip = ipaddress.IPv4Address(gateway_bytes[::-1])
                return str(gateway_ip)
        except OSError:
            return None
        return None

    def _candidate_base_urls(self) -> list[str]:
        candidates = [self.base_url]
        parsed = urlparse(self.base_url)
        host = (parsed.hostname or "").strip().casefold()
        if host not in {"127.0.0.1", "localhost"} or not self._is_wsl():
            return candidates
        gateway_host = self._windows_host_gateway()
        if not gateway_host:
            return candidates
        fallback_url = urlunparse(
            (
                parsed.scheme or "http",
                f"{gateway_host}:{parsed.port or 23119}",
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        ).rstrip("/")
        if fallback_url not in candidates:
            candidates.append(fallback_url)
        return candidates

    async def _resolve_base_url(self) -> str:
        candidates = self._candidate_base_urls()
        if len(candidates) <= 1:
            return self.base_url
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                async with self._build_async_client() as client:
                    response = await client.get(f"{candidate}/connector/ping")
                    response.raise_for_status()
                self.base_url = candidate
                return candidate
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.NetworkError) as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        return self.base_url

    async def ping(self) -> ZoteroStatusToolOutput:
        await self._resolve_base_url()
        async with self._build_async_client() as client:
            response = await client.get(f"{self.base_url}/connector/ping")
            response.raise_for_status()
            message = response.text.strip()
        return ZoteroStatusToolOutput(
            status="ok",
            base_url=self.base_url,
            message=message or "Zotero connector reachable",
        )

    async def get_selected_collection(self) -> ZoteroSelectedCollectionToolOutput:
        await self._resolve_base_url()
        async with self._build_async_client() as client:
            payload = await self._fetch_selected_collection_payload(client)
        return ZoteroSelectedCollectionToolOutput(
            editable=bool(payload.get("editable")),
            library_id=payload.get("libraryID"),
            library_name=payload.get("libraryName"),
            collection_id=payload.get("id"),
            collection_key=self._coerce_string(payload.get("key")),
            collection_name=payload.get("name"),
        )

    async def list_collections(self, *, top_level_only: bool = False) -> ZoteroCollectionListToolOutput:
        await self._resolve_base_url()
        endpoint = "collections/top" if top_level_only else "collections"
        async with self._build_async_client() as client:
            response = await client.get(f"{self.base_url}/api/users/{self.user_id}/{endpoint}")
            response.raise_for_status()
            payload = response.json() if response.content else []
        collections = [self._parse_collection(item) for item in payload if isinstance(item, dict)]
        return ZoteroCollectionListToolOutput(collections=collections)

    async def search_items(
        self,
        *,
        query: str,
        limit: int = 10,
        include_attachments: bool = True,
        collection_key: str | None = None,
        collection_name: str | None = None,
        current_collection_only: bool = False,
    ) -> ZoteroSearchItemsToolOutput:
        await self._resolve_base_url()
        # Prefer top-level library items so searches return parent papers rather than
        # attachments, snapshots, and annotation children.
        params = {
            "q": query,
            "limit": str(limit),
            "qmode": "titleCreatorYear",
        }
        async with self._build_async_client() as client:
            resolved_collection_key = await self._resolve_collection_key(
                client,
                collection_key=collection_key,
                collection_name=collection_name,
                current_collection_only=current_collection_only,
            )
            items_endpoint = (
                f"{self.base_url}/api/users/{self.user_id}/collections/{resolved_collection_key}/items/top"
                if resolved_collection_key
                else f"{self.base_url}/api/users/{self.user_id}/items/top"
            )
            response = await client.get(items_endpoint, params=params)
            response.raise_for_status()
            payload = response.json() if response.content else []
            items = [item for item in payload if isinstance(item, dict)]

            attachments_by_parent: dict[str, list[ZoteroAttachmentRecord]] = {}
            if include_attachments:
                for item in items:
                    item_key = self._data_from_entry(item).get("key")
                    if not isinstance(item_key, str) or not item_key:
                        continue
                    attachments_by_parent[item_key] = await self._fetch_item_attachments(client, item_key)

        return ZoteroSearchItemsToolOutput(
            items=[
                self._parse_item(
                    item,
                    attachments=attachments_by_parent.get(self._data_from_entry(item).get("key") or "", []),
                )
                for item in items
            ]
        )

    async def get_item_attachments(self, *, item_key: str) -> ZoteroItemChildrenToolOutput:
        async with self._build_async_client() as client:
            attachments = await self._fetch_item_attachments(client, item_key)
        return ZoteroItemChildrenToolOutput(item_key=item_key, attachments=attachments)

    async def _resolve_collection_key(
        self,
        client: httpx.AsyncClient,
        *,
        collection_key: str | None,
        collection_name: str | None,
        current_collection_only: bool,
    ) -> str | None:
        if collection_key and collection_key.strip():
            return collection_key.strip()
        if current_collection_only:
            payload = await self._fetch_selected_collection_payload(client)
            selected_key = await self._resolve_selected_collection_key(client, payload)
            if selected_key is not None:
                return selected_key
            raise ValueError("No Zotero collection is currently selected.")
        if collection_name and collection_name.strip():
            response = await client.get(f"{self.base_url}/api/users/{self.user_id}/collections")
            response.raise_for_status()
            payload = response.json() if response.content else []
            matches = [
                self._parse_collection(item)
                for item in payload
                if isinstance(item, dict) and self._collection_name_matches(item, collection_name)
            ]
            if not matches:
                raise ValueError(f"Zotero collection '{collection_name}' was not found.")
            if len(matches) > 1:
                raise ValueError(
                    f"Multiple Zotero collections named '{collection_name}' were found; use collection_key instead."
                )
            return matches[0].key
        return None

    async def _fetch_selected_collection_payload(
        self,
        client: httpx.AsyncClient,
    ) -> dict[str, Any]:
        endpoint = f"{self.base_url}/connector/getSelectedCollection"
        response = await client.get(endpoint)
        if response.status_code < 400:
            payload = response.json() if response.content else {}
            return payload if isinstance(payload, dict) else {}

        # Zotero 7 connector variants may reject GET for this endpoint and require
        # a POST request with an explicit zero-length body.
        fallback_response = await client.post(endpoint, content=b"")
        fallback_response.raise_for_status()
        payload = fallback_response.json() if fallback_response.content else {}
        return payload if isinstance(payload, dict) else {}

    async def _resolve_selected_collection_key(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, Any],
    ) -> str | None:
        selected_key = payload.get("key")
        if isinstance(selected_key, str) and selected_key.strip():
            return selected_key.strip()
        selected_name = payload.get("name")
        if not isinstance(selected_name, str) or not selected_name.strip():
            return None
        try:
            response = await client.get(f"{self.base_url}/api/users/{self.user_id}/collections")
            response.raise_for_status()
            items = response.json() if response.content else []
        except Exception:
            return None
        matches = [
            self._parse_collection(item)
            for item in items
            if isinstance(item, dict) and self._collection_name_matches(item, selected_name)
        ]
        if len(matches) == 1 and matches[0].key:
            return matches[0].key
        return None

    async def import_paper(
        self,
        *,
        title: str,
        item_type: str = "journalArticle",
        authors: list[str] | None = None,
        abstract: str | None = None,
        year: int | None = None,
        url: str | None = None,
        doi: str | None = None,
        publication_title: str | None = None,
        pdf_url: str | None = None,
        collection_name: str | None = None,
    ) -> ZoteroImportPaperToolOutput:
        await self._resolve_base_url()
        selected_collection = await self.get_selected_collection()
        warnings: list[str] = []
        if collection_name and selected_collection.collection_name and collection_name != selected_collection.collection_name:
            warnings.append(
                f"Current selected Zotero collection is '{selected_collection.collection_name}', not '{collection_name}'."
            )

        session_id = f"rc_{uuid4().hex}"
        connector_item_id = f"item_{uuid4().hex}"
        item_payload = {
            "id": connector_item_id,
            "itemType": item_type,
            "title": title,
            "creators": [self._creator_from_name(name) for name in (authors or []) if name.strip()],
            "abstractNote": abstract or "",
            "url": url or "",
            "DOI": doi or "",
            "publicationTitle": publication_title or "",
        }
        if year is not None:
            item_payload["date"] = str(year)

        async with self._build_async_client() as client:
            save_response = await client.post(
                f"{self.base_url}/connector/saveItems",
                json={"sessionID": session_id, "items": [item_payload]},
            )
            save_response.raise_for_status()
            try:
                save_payload = save_response.json() if save_response.content else {}
            except Exception:
                save_payload = {}
            imported_item_key = self._extract_imported_item_key(save_payload, connector_item_id)
            if imported_item_key is None:
                imported_item_key = await self._find_imported_item_key(
                    client,
                    title=title,
                    url=url,
                    doi=doi,
                    year=year,
                )

            attachment_title = None
            if pdf_url:
                pdf_response = await client.get(pdf_url, follow_redirects=True)
                pdf_response.raise_for_status()
                attachment_title = f"{title} PDF"
                metadata = {
                    "sessionID": session_id,
                    "title": attachment_title,
                    "url": pdf_url,
                    "parentItemID": connector_item_id,
                }
                attachment_response = await client.post(
                    f"{self.base_url}/connector/saveAttachment",
                    content=pdf_response.content,
                    headers={
                        "Content-Type": pdf_response.headers.get("content-type", "application/pdf"),
                        "X-Metadata": json.dumps(metadata),
                    },
                )
                attachment_response.raise_for_status()

        return ZoteroImportPaperToolOutput(
            status="imported",
            imported_item_key=imported_item_key,
            attachment_title=attachment_title,
            selected_collection=selected_collection,
            warnings=warnings,
        )

    async def _find_imported_item_key(
        self,
        client: httpx.AsyncClient,
        *,
        title: str,
        url: str | None,
        doi: str | None,
        year: int | None,
    ) -> str | None:
        response = await client.get(
            f"{self.base_url}/api/users/{self.user_id}/items/top",
            params={"q": title, "limit": "10", "qmode": "titleCreatorYear"},
        )
        response.raise_for_status()
        payload = response.json() if response.content else []
        if not isinstance(payload, list):
            return None
        normalized_title = self._normalize_match_text(title)
        normalized_url = url.strip().rstrip("/") if isinstance(url, str) and url.strip() else None
        normalized_doi = doi.strip().casefold() if isinstance(doi, str) and doi.strip() else None
        expected_year = str(year) if isinstance(year, int) else None
        for item in payload:
            if not isinstance(item, dict):
                continue
            data = self._data_from_entry(item)
            item_title = self._coerce_string(data.get("title"))
            if not item_title or self._normalize_match_text(item_title) != normalized_title:
                continue
            item_url = self._coerce_string(data.get("url"))
            item_doi = self._coerce_string(data.get("DOI"))
            item_year = self._extract_year(data)
            if normalized_doi and item_doi and item_doi.casefold() == normalized_doi:
                return self._coerce_string(data.get("key"))
            if normalized_url and item_url and item_url.rstrip("/") == normalized_url:
                return self._coerce_string(data.get("key"))
            if expected_year is None or item_year == expected_year:
                return self._coerce_string(data.get("key"))
        return None

    async def attach_pdf_to_item(
        self,
        *,
        item_key: str,
        pdf_url: str,
        title: str | None = None,
        source_url: str | None = None,
    ) -> ZoteroAttachPdfToItemToolOutput:
        await self._resolve_base_url()
        session_id = f"rc_attach_{uuid4().hex}"
        attachment_title = (title.strip() if isinstance(title, str) and title.strip() else None) or "PDF Attachment"

        async with self._build_async_client() as client:
            pdf_response = await client.get(pdf_url, follow_redirects=True)
            pdf_response.raise_for_status()
            metadata = {
                "sessionID": session_id,
                "title": attachment_title,
                "url": source_url or pdf_url,
                "parentItemKey": item_key,
            }
            attachment_response = await client.post(
                f"{self.base_url}/connector/saveAttachment",
                content=pdf_response.content,
                headers={
                    "Content-Type": pdf_response.headers.get("content-type", "application/pdf"),
                    "X-Metadata": json.dumps(metadata),
                },
            )
            attachment_response.raise_for_status()
            attachments = await self._fetch_item_attachments(client, item_key)

        matched_attachment = next(
            (
                attachment
                for attachment in attachments
                if (attachment.url and attachment.url.rstrip("/") == (source_url or pdf_url).rstrip("/"))
                or attachment.title == attachment_title
            ),
            None,
        )
        warnings: list[str] = []
        status = "attached"
        if matched_attachment is None:
            status = "failed"
            warnings.append("Attachment upload completed but the new attachment was not verified under the target item.")
        return ZoteroAttachPdfToItemToolOutput(
            status=status,
            item_key=item_key,
            attachment_title=matched_attachment.title if matched_attachment is not None else attachment_title,
            attachment_count=len(attachments),
            warnings=warnings,
        )

    async def _fetch_item_attachments(
        self,
        client: httpx.AsyncClient,
        item_key: str,
    ) -> list[ZoteroAttachmentRecord]:
        response = await client.get(f"{self.base_url}/api/users/{self.user_id}/items/{item_key}/children")
        response.raise_for_status()
        payload = response.json() if response.content else []
        attachments: list[ZoteroAttachmentRecord] = []
        for child in payload:
            if not isinstance(child, dict):
                continue
            data = self._data_from_entry(child)
            if data.get("itemType") != "attachment":
                continue
            attachments.append(
                ZoteroAttachmentRecord(
                    key=self._coerce_string(data.get("key")),
                    title=self._coerce_string(data.get("title")) or "Attachment",
                    content_type=self._coerce_string(data.get("contentType")),
                    local_path=self._extract_attachment_path(child),
                    url=self._coerce_string(data.get("url")),
                )
            )
        return attachments

    def _parse_collection(self, payload: dict[str, Any]) -> ZoteroCollectionRecord:
        data = self._data_from_entry(payload)
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        return ZoteroCollectionRecord(
            key=self._coerce_string(data.get("key")),
            name=self._coerce_string(data.get("name")) or "Unnamed Collection",
            parent_collection=self._coerce_string(data.get("parentCollection")),
            item_count=meta.get("numItems") if isinstance(meta.get("numItems"), int) else None,
        )

    def _collection_name_matches(self, payload: dict[str, Any], expected_name: str) -> bool:
        collection = self._parse_collection(payload)
        return collection.name.casefold() == expected_name.strip().casefold()

    def _parse_item(
        self,
        payload: dict[str, Any],
        *,
        attachments: Iterable[ZoteroAttachmentRecord] | None = None,
    ) -> ZoteroItemRecord:
        data = self._data_from_entry(payload)
        return ZoteroItemRecord(
            key=self._coerce_string(data.get("key")),
            item_type=self._coerce_string(data.get("itemType")) or "item",
            title=self._coerce_string(data.get("title")) or "Untitled",
            creators=self._creator_names(data.get("creators")),
            abstract=self._coerce_string(data.get("abstractNote")),
            year=self._extract_year(data),
            url=self._coerce_string(data.get("url")),
            doi=self._coerce_string(data.get("DOI")),
            collections=[
                value for value in (data.get("collections") or []) if isinstance(value, str)
            ],
            attachments=list(attachments or []),
        )

    def _data_from_entry(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = payload.get("data")
        return data if isinstance(data, dict) else payload

    def _extract_attachment_path(self, payload: dict[str, Any]) -> str | None:
        links = payload.get("links")
        if isinstance(links, dict):
            enclosure = links.get("enclosure")
            if isinstance(enclosure, dict):
                href = enclosure.get("href")
                if isinstance(href, str) and href.startswith("file:"):
                    return href
        return None

    def _extract_imported_item_key(self, payload: Any, connector_item_id: str) -> str | None:
        if not isinstance(payload, dict):
            return None
        successful = payload.get("successful")
        if not isinstance(successful, dict):
            return None
        for value in successful.values():
            if not isinstance(value, dict):
                continue
            if value.get("id") == connector_item_id:
                return self._coerce_string(value.get("key"))
            data = value.get("data")
            if isinstance(data, dict) and data.get("id") == connector_item_id:
                return self._coerce_string(value.get("key"))
        first_value = next(iter(successful.values()), None)
        return self._coerce_string(first_value.get("key")) if isinstance(first_value, dict) else None

    def _extract_year(self, data: dict[str, Any]) -> str | None:
        date = self._coerce_string(data.get("date"))
        if not date:
            return None
        return date[:4]

    def _creator_names(self, creators: Any) -> list[str]:
        if not isinstance(creators, list):
            return []
        names: list[str] = []
        for creator in creators:
            if not isinstance(creator, dict):
                continue
            full_name = self._coerce_string(creator.get("name"))
            if full_name:
                names.append(full_name)
                continue
            first_name = self._coerce_string(creator.get("firstName")) or ""
            last_name = self._coerce_string(creator.get("lastName")) or ""
            joined = f"{first_name} {last_name}".strip()
            if joined:
                names.append(joined)
        return names

    def _creator_from_name(self, name: str) -> dict[str, str]:
        parts = [part for part in name.split() if part]
        if len(parts) >= 2:
            return {
                "creatorType": "author",
                "firstName": " ".join(parts[:-1]),
                "lastName": parts[-1],
            }
        return {
            "creatorType": "author",
            "name": name.strip(),
        }

    def _coerce_string(self, value: Any) -> str | None:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return None

    def _normalize_match_text(self, value: str) -> str:
        return "".join(char.lower() for char in value if char.isalnum())


def build_zotero_local_mcp_app(
    config: ZoteroLocalServerConfig,
    *,
    server_name: str = "zotero-local",
):
    from core.prompt_resolver import PromptResolver
    from mcp.server.app import MCPServerApp
    from mcp.server.prompt_adapter import MCPPromptAdapter
    from mcp.server.resource_adapter import MCPResourceAdapter
    from mcp.server.tool_adapter import MCPToolAdapter

    gateway = ZoteroLocalGateway(
        base_url=config.base_url,
        user_id=config.user_id,
        timeout_seconds=config.timeout_seconds,
    )
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    registry.register_many(
        [
            ToolSpec(
                name="zotero_ping",
                description="Check whether the local Zotero connector API is reachable.",
                input_schema=EmptyToolInput,
                output_schema=ZoteroStatusToolOutput,
                handler=gateway.ping,
                tags=["zotero", "mcp", "local"],
            ),
            ToolSpec(
                name="zotero_get_selected_collection",
                description="Get the currently selected Zotero collection in the local desktop app.",
                input_schema=EmptyToolInput,
                output_schema=ZoteroSelectedCollectionToolOutput,
                handler=gateway.get_selected_collection,
                tags=["zotero", "mcp", "local"],
            ),
            ToolSpec(
                name="zotero_list_collections",
                description="List collections from the local Zotero library.",
                input_schema=ZoteroCollectionListToolInput,
                output_schema=ZoteroCollectionListToolOutput,
                handler=gateway.list_collections,
                tags=["zotero", "mcp", "local"],
            ),
            ToolSpec(
                name="zotero_search_items",
                description=(
                    "Search items in the local Zotero library, a specific collection, "
                    "or the currently selected collection, and optionally include attachment paths."
                ),
                input_schema=ZoteroSearchItemsToolInput,
                output_schema=ZoteroSearchItemsToolOutput,
                handler=gateway.search_items,
                tags=["zotero", "mcp", "local"],
            ),
            ToolSpec(
                name="zotero_get_item_attachments",
                description="List attachment metadata and local file paths for a Zotero item.",
                input_schema=ZoteroItemChildrenToolInput,
                output_schema=ZoteroItemChildrenToolOutput,
                handler=gateway.get_item_attachments,
                tags=["zotero", "mcp", "local"],
            ),
            ToolSpec(
                name="zotero_import_paper",
                description="Create a Zotero item locally and optionally download/import a PDF into the selected collection.",
                input_schema=ZoteroImportPaperToolInput,
                output_schema=ZoteroImportPaperToolOutput,
                handler=gateway.import_paper,
                tags=["zotero", "mcp", "local", "import"],
            ),
            ToolSpec(
                name="zotero_attach_pdf_to_item",
                description="Download a PDF and attach it to an existing Zotero item, then verify it appears under that item.",
                input_schema=ZoteroAttachPdfToItemToolInput,
                output_schema=ZoteroAttachPdfToItemToolOutput,
                handler=gateway.attach_pdf_to_item,
                tags=["zotero", "mcp", "local", "attachment"],
            ),
        ],
        replace=True,
    )
    resource_adapter = MCPResourceAdapter()
    resource_adapter.set_config_info(
        "zotero_local",
        {
            "base_url": gateway.base_url,
            "user_id": gateway.user_id,
            "server_name": server_name,
        },
    )
    return MCPServerApp(
        server_name=server_name,
        description="Local Zotero desktop MCP server",
        tool_adapter=MCPToolAdapter(registry=registry, executor=executor, server_name=server_name),
        prompt_adapter=MCPPromptAdapter(prompt_resolver=PromptResolver()),
        resource_adapter=resource_adapter,
    )


def build_zotero_local_mcp_client(config: ZoteroLocalServerConfig):
    from mcp.client.base import InProcessMCPClient

    return InProcessMCPClient(build_zotero_local_mcp_app(config))
