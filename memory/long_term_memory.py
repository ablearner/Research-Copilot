from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import re
from typing import Protocol

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


class SQLiteLongTermMemoryStore:
    """SQLite + FTS5 backed long-term memory store.

    Shares the same kepler.db used by research data and session memory.
    """

    def __init__(self, db_path: str | Path, *, max_records: int = 5000) -> None:
        import sqlite3

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records = max(1, max_records)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS long_term_memory (
                memory_id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL DEFAULT 'topic',
                topic TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                keywords TEXT NOT NULL DEFAULT '',
                related_paper_ids TEXT NOT NULL DEFAULT '',
                source_session_id TEXT,
                context_snapshot TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS long_term_memory_fts USING fts5(
                topic, content, keywords,
                content=long_term_memory,
                content_rowid=rowid,
                tokenize='unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS ltm_ai AFTER INSERT ON long_term_memory BEGIN
                INSERT INTO long_term_memory_fts(rowid, topic, content, keywords)
                VALUES (new.rowid, new.topic, new.content, new.keywords);
            END;

            CREATE TRIGGER IF NOT EXISTS ltm_ad AFTER DELETE ON long_term_memory BEGIN
                INSERT INTO long_term_memory_fts(long_term_memory_fts, rowid, topic, content, keywords)
                VALUES ('delete', old.rowid, old.topic, old.content, old.keywords);
            END;

            CREATE TRIGGER IF NOT EXISTS ltm_au AFTER UPDATE ON long_term_memory BEGIN
                INSERT INTO long_term_memory_fts(long_term_memory_fts, rowid, topic, content, keywords)
                VALUES ('delete', old.rowid, old.topic, old.content, old.keywords);
                INSERT INTO long_term_memory_fts(rowid, topic, content, keywords)
                VALUES (new.rowid, new.topic, new.content, new.keywords);
            END;
            """
        )

    def upsert(self, record: LongTermMemoryRecord) -> LongTermMemoryRecord:
        import json

        updated = record.model_copy(update={"updated_at": utc_now()})
        keywords_str = " ".join(updated.keywords)
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO long_term_memory "
                "(memory_id, memory_type, topic, content, keywords, related_paper_ids, "
                "source_session_id, context_snapshot, created_at, updated_at, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    updated.memory_id,
                    updated.memory_type,
                    updated.topic,
                    updated.content,
                    keywords_str,
                    " ".join(updated.related_paper_ids),
                    updated.source_session_id,
                    json.dumps(updated.context_snapshot, ensure_ascii=False),
                    updated.created_at.isoformat(),
                    updated.updated_at.isoformat(),
                    json.dumps(updated.metadata, ensure_ascii=False),
                ),
            )
        self._enforce_record_limit()
        return updated

    def search(self, query: LongTermMemoryQuery) -> LongTermMemorySearchResult:
        import json

        search_terms = " ".join(query.keywords) + " " + query.query if query.keywords else query.query
        fts_query = " OR ".join(
            f'"{term}"' for term in _tokenize(search_terms) if term
        )
        records: list[LongTermMemoryRecord] = []
        if fts_query:
            rows = self._conn.execute(
                "SELECT m.*, rank FROM long_term_memory m "
                "JOIN long_term_memory_fts fts ON m.rowid = fts.rowid "
                "WHERE long_term_memory_fts MATCH ? "
                "ORDER BY rank "
                "LIMIT ?",
                (fts_query, query.top_k * 3),
            ).fetchall()
            for row in rows:
                record = self._row_to_record(row)
                score = _lexical_score(record, _tokenize(query.query), query.keywords, query.topic)
                if score < query.min_score:
                    continue
                records.append(record.model_copy(update={"score": score}))

        if len(records) < query.top_k:
            all_rows = self._conn.execute(
                "SELECT * FROM long_term_memory ORDER BY updated_at DESC LIMIT ?",
                (query.top_k * 5,),
            ).fetchall()
            seen = {r.memory_id for r in records}
            for row in all_rows:
                record = self._row_to_record(row)
                if record.memory_id in seen:
                    continue
                score = _lexical_score(record, _tokenize(query.query), query.keywords, query.topic)
                if score < query.min_score:
                    continue
                records.append(record.model_copy(update={"score": score}))

        records.sort(key=lambda item: (item.score or 0, item.updated_at), reverse=True)
        return LongTermMemorySearchResult(query=query, records=records[: query.top_k])

    def _enforce_record_limit(self) -> None:
        count = self._conn.execute("SELECT COUNT(*) FROM long_term_memory").fetchone()[0]
        overflow = count - self.max_records
        if overflow <= 0:
            return
        with self._conn:
            self._conn.execute(
                "DELETE FROM long_term_memory WHERE memory_id IN ("
                "  SELECT memory_id FROM long_term_memory ORDER BY updated_at ASC LIMIT ?"
                ")",
                (overflow,),
            )

    def _row_to_record(self, row) -> LongTermMemoryRecord:
        import json

        cols = [desc[0] for desc in self._conn.execute("SELECT * FROM long_term_memory LIMIT 0").description]
        data = dict(zip(cols, row[: len(cols)]))
        return LongTermMemoryRecord(
            memory_id=data["memory_id"],
            memory_type=data["memory_type"],
            topic=data["topic"],
            content=data["content"],
            keywords=[k for k in data["keywords"].split() if k],
            related_paper_ids=[p for p in data["related_paper_ids"].split() if p],
            source_session_id=data["source_session_id"],
            context_snapshot=json.loads(data["context_snapshot"] or "{}"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=json.loads(data["metadata_json"] or "{}"),
        )


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
