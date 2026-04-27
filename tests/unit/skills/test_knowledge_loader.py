import pytest
from pathlib import Path

from skills.knowledge_loader import KnowledgeSkillLoader


@pytest.fixture
def skills_dir(tmp_path):
    return tmp_path / "skills"


@pytest.fixture
def loader(skills_dir):
    return KnowledgeSkillLoader(skills_dir)


def test_list_empty(loader):
    assert loader.list_skills() == []


def test_save_and_list(loader):
    loader.save_skill("test-skill", "# Hello\n\nSome content", description="A test", tags=["test"])
    skills = loader.list_skills()
    assert len(skills) == 1
    assert skills[0].name == "test-skill"
    assert skills[0].description == "A test"
    assert "test" in skills[0].tags


def test_save_and_load(loader):
    loader.save_skill("quantum", "## Quantum Tips\n\nUse Schrodinger eq.", description="QM", tags=["physics"])
    skill = loader.load_skill("quantum")
    assert skill is not None
    assert skill.meta.name == "quantum"
    assert "Schrodinger" in skill.content
    assert skill.meta.tags == ["physics"]


def test_update_preserves_created_at(loader):
    loader.save_skill("evolve", "v1", description="first")
    s1 = loader.load_skill("evolve")
    assert s1 is not None
    created = s1.meta.created_at

    loader.save_skill("evolve", "v2", description="second")
    s2 = loader.load_skill("evolve")
    assert s2 is not None
    assert s2.meta.created_at == created
    assert s2.meta.description == "second"
    assert "v2" in s2.content


def test_delete_skill(loader):
    loader.save_skill("to-delete", "temp")
    assert loader.load_skill("to-delete") is not None
    deleted = loader.delete_skill("to-delete")
    assert deleted
    assert loader.load_skill("to-delete") is None


def test_delete_nonexistent(loader):
    assert not loader.delete_skill("nonexistent")


def test_load_nonexistent(loader):
    assert loader.load_skill("nonexistent") is None
