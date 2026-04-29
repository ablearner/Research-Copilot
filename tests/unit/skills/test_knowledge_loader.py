import pytest
from pathlib import Path

from core.knowledge_loader import KnowledgeLoader


@pytest.fixture
def knowledge_dir(tmp_path):
    return tmp_path / "knowledge"


@pytest.fixture
def loader(knowledge_dir):
    return KnowledgeLoader(knowledge_dir)


def test_list_empty(loader):
    assert loader.list_entries() == []


def test_save_and_list(loader):
    loader.save_entry("test-skill", "# Hello\n\nSome content", description="A test", tags=["test"])
    entries = loader.list_entries()
    assert len(entries) == 1
    assert entries[0].name == "test-skill"
    assert entries[0].description == "A test"
    assert "test" in entries[0].tags


def test_save_and_load(loader):
    loader.save_entry("quantum", "## Quantum Tips\n\nUse Schrodinger eq.", description="QM", tags=["physics"])
    entry = loader.load_entry("quantum")
    assert entry is not None
    assert entry.meta.name == "quantum"
    assert "Schrodinger" in entry.content
    assert entry.meta.tags == ["physics"]


def test_update_preserves_created_at(loader):
    loader.save_entry("evolve", "v1", description="first")
    s1 = loader.load_entry("evolve")
    assert s1 is not None
    created = s1.meta.created_at

    loader.save_entry("evolve", "v2", description="second")
    s2 = loader.load_entry("evolve")
    assert s2 is not None
    assert s2.meta.created_at == created
    assert s2.meta.description == "second"
    assert "v2" in s2.content


def test_delete_entry(loader):
    loader.save_entry("to-delete", "temp")
    assert loader.load_entry("to-delete") is not None
    deleted = loader.delete_entry("to-delete")
    assert deleted
    assert loader.load_entry("to-delete") is None


def test_delete_nonexistent(loader):
    assert not loader.delete_entry("nonexistent")


def test_load_nonexistent(loader):
    assert loader.load_entry("nonexistent") is None
