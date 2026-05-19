from pathlib import Path

from adapters.storage.sqlite_store import SQLiteStore


def test_sqlite_store_exposes_storage_root_for_file_artifacts(tmp_path: Path) -> None:
    store = SQLiteStore(db_path=tmp_path / "research" / "kepler.db")

    assert store.db_path == tmp_path / "research" / "kepler.db"
    assert store.storage_root == tmp_path / "research"
