from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import re
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from domain.schemas.research_memory import (
    # NOTE: memory.security imported lazily below to avoid circular imports
    LongTermMemoryQuery,
    LongTermMemoryRecord,
    LongTermMemorySearchResult,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _tokenize(text: str) -> set[str]:
    normalized = text.replace("\n", " ").lower()
    tokens = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", normalized))
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", normalized)
    tokens.update(
        "".join(cjk_chars[index : index + 2])
        for index in range(max(len(cjk_chars) - 1, 0))
    )
    tokens.update(
        token.strip().lower()
        for token in normalized.split()
        if token.strip()
    )
    return {token for token in tokens if token}


def deterministic_memory_vector(record_or_text: LongTermMemoryRecord | str, *, size: int = 128) -> list[float]:
    text = (
        " ".join(
            [
                record_or_text.topic,
                record_or_text.content,
                " ".join(record_or_text.keywords),
                " ".join(record_or_text.related_paper_ids),
            ]
        )
        if isinstance(record_or_text, LongTermMemoryRecord)
        else str(record_or_text)
    )
    values = [0.0] * max(1, size)
    for token in _tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % len(values)
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        values[index] += sign
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


class LongTermMemoryStore(Protocol):
    def upsert(self, record: LongTermMemoryRecord) -> LongTermMemoryRecord:
        ...

    def search(self, query: LongTermMemoryQuery) -> LongTermMemorySearchResult:
        ...


class InMemoryLongTermMemoryStore:
    def __init__(self) -> None:
        self._records: dict[str, LongTermMemoryRecord] = {}

    def upsert(self, record: LongTermMemoryRecord) -> LongTermMemoryRecord:
        updated = record.model_copy(update={"updated_at": utc_now()})
        self._records[updated.memory_id] = updated
        return updated

    def search(self, query: LongTermMemoryQuery) -> LongTermMemorySearchResult:
        query_terms = _tokenize(query.query)
        ranked: list[LongTermMemoryRecord] = []
        for record in self._records.values():
            score = _lexical_score(record, query_terms, query.keywords, query.topic)
            if score < query.min_score:
                continue
            ranked.append(record.model_copy(update={"score": score}))
        ranked.sort(key=lambda item: item.score or 0, reverse=True)
        return LongTermMemorySearchResult(query=query, records=ranked[: query.top_k])


class JsonLongTermMemoryStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self._ensure_root_dir()

    def _ensure_root_dir(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, memory_id: str) -> Path:
        digest = hashlib.sha1(memory_id.encode("utf-8")).hexdigest()
        return self.root_dir / f"{digest}.json"

    def _iter_records(self) -> list[LongTermMemoryRecord]:
        self._ensure_root_dir()
        records: list[LongTermMemoryRecord] = []
        for path in sorted(self.root_dir.glob("*.json")):
            try:
                records.append(
                    LongTermMemoryRecord.model_validate_json(path.read_text(encoding="utf-8"))
                )
            except Exception:
                continue
        return records

    def upsert(self, record: LongTermMemoryRecord) -> LongTermMemoryRecord:
        self._ensure_root_dir()
        path = self._path_for(record.memory_id)
        updated = record.model_copy(update={"updated_at": utc_now()})
        path.write_text(updated.model_dump_json(indent=2), encoding="utf-8")
        return updated

    def search(self, query: LongTermMemoryQuery) -> LongTermMemorySearchResult:
        query_terms = _tokenize(query.query)
        ranked: list[LongTermMemoryRecord] = []
        for record in self._iter_records():
            score = _lexical_score(record, query_terms, query.keywords, query.topic)
            if score < query.min_score:
                continue
            ranked.append(record.model_copy(update={"score": score}))
        ranked.sort(
            key=lambda item: (
                float(item.score or 0.0),
                item.updated_at,
            ),
            reverse=True,
        )
        return LongTermMemorySearchResult(query=query, records=ranked[: query.top_k])


class QdrantLongTermMemoryStore:
    def __init__(
        self,
        *,
        path: str,
        collection_name: str,
        vector_size: int = 128,
        max_records: int = 5000,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as rest
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError("qdrant-client is required for QdrantLongTermMemoryStore") from exc

        self._rest = rest
        self.client = QdrantClient(path=path)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.max_records = max(1, max_records)
        collections = {item.name for item in self.client.get_collections().collections}
        if collection_name not in collections:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
            )

    def upsert(self, record: LongTermMemoryRecord) -> LongTermMemoryRecord:
        vector = record.vector or deterministic_memory_vector(record, size=self.vector_size)
        if len(vector) != self.vector_size:
            raise ValueError(
                f"QdrantLongTermMemoryStore vector size mismatch: expected {self.vector_size}, got {len(vector)}"
            )
        updated = record.model_copy(update={"vector": vector, "updated_at": utc_now()})
        point = self._rest.PointStruct(
            id=self._point_id(updated.memory_id),
            vector=vector,
            payload=updated.model_dump(mode="json"),
        )
        self.client.upsert(collection_name=self.collection_name, points=[point])
        self._enforce_record_limit()
        return updated

    def search(self, query: LongTermMemoryQuery) -> LongTermMemorySearchResult:
        query_vector = query.vector or deterministic_memory_vector(query.query, size=self.vector_size)
        if query_vector:
            points = self._query_points(
                query_vector,
                limit=query.top_k,
            )
            records = [
                LongTermMemoryRecord.model_validate(point.payload).model_copy(
                    update={"score": float(point.score or 0)}
                )
                for point in points
                if point.payload
            ]
            return LongTermMemorySearchResult(query=query, records=records)

        scrolled, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=max(query.top_k * 5, 20),
            with_payload=True,
            with_vectors=False,
        )
        query_terms = _tokenize(query.query)
        ranked: list[LongTermMemoryRecord] = []
        for point in scrolled:
            if not point.payload:
                continue
            record = LongTermMemoryRecord.model_validate(point.payload)
            score = _lexical_score(record, query_terms, query.keywords, query.topic)
            if score < query.min_score:
                continue
            ranked.append(record.model_copy(update={"score": score}))
        ranked.sort(key=lambda item: item.score or 0, reverse=True)
        return LongTermMemorySearchResult(query=query, records=ranked[: query.top_k])

    def _enforce_record_limit(self) -> None:
        scrolled, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=self.max_records + 100,
            with_payload=True,
            with_vectors=False,
        )
        overflow = len(scrolled) - self.max_records
        if overflow <= 0:
            return
        sortable: list[tuple[datetime, str]] = []
        for point in scrolled:
            if not point.payload:
                continue
            try:
                record = LongTermMemoryRecord.model_validate(point.payload)
            except Exception:
                continue
            sortable.append((record.updated_at, str(point.id)))
        sortable.sort(key=lambda item: item[0])
        delete_ids = [point_id for _, point_id in sortable[:overflow]]
        if delete_ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=self._rest.PointIdsList(points=delete_ids),
            )

    def _point_id(self, memory_id: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"{self.collection_name}:{memory_id}"))

    def _query_points(self, query_vector: list[float], *, limit: int):
        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
            )
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        return list(getattr(response, "points", []) or [])


class LongTermMemory:
    def __init__(self, store: LongTermMemoryStore | None = None) -> None:
        self.store = store or InMemoryLongTermMemoryStore()

    def upsert(self, record: LongTermMemoryRecord) -> LongTermMemoryRecord:
        from memory.security import scan_memory_content

        threat = scan_memory_content(record.content)
        if threat is not None:
            import logging
            logging.getLogger(__name__).warning(
                "Blocked unsafe memory content (%s): %s", threat, record.memory_id,
            )
            raise ValueError(f"Memory content blocked: {threat}")
        return self.store.upsert(record)

    def search(self, query: LongTermMemoryQuery) -> LongTermMemorySearchResult:
        return self.store.search(query)

    def deduplicate(self, records: Iterable[LongTermMemoryRecord]) -> list[LongTermMemoryRecord]:
        seen: set[tuple[str, str]] = set()
        deduped: list[LongTermMemoryRecord] = []
        for record in records:
            marker = (record.memory_type, record.content.strip().lower())
            if marker in seen:
                continue
            seen.add(marker)
            deduped.append(record)
        return deduped


def _lexical_score(
    record: LongTermMemoryRecord,
    query_terms: set[str],
    query_keywords: list[str],
    topic: str | None,
) -> float:
    haystack = _tokenize(" ".join([record.topic, record.content, *record.keywords]))
    overlap = len(query_terms & haystack)
    keyword_overlap = len(set(keyword.lower() for keyword in query_keywords) & haystack)
    topic_bonus = 1.0 if topic and topic.strip() and topic.strip().lower() == record.topic.strip().lower() else 0.0
    return float(overlap + keyword_overlap + topic_bonus)
