"""Installed distribution must contain the runtime and its static inputs."""

from __future__ import annotations

import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

from setuptools import find_packages


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_runtime_subpackages_are_discovered() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        config = tomllib.load(file)
    discovery = config["tool"]["setuptools"]["packages"]["find"]
    packages = set(find_packages(where=str(PROJECT_ROOT), include=discovery["include"], exclude=discovery["exclude"]))
    assert {"runtime", "runtime.research", "runtime.research.agent_protocol"} <= packages


def test_wheel_installed_outside_checkout_loads_runtime_resources(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-w", str(wheel_dir), "."],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(wheel_dir.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert "runtime/research/supervisor_graph_runtime_core.py" in names
    resource_paths = [
        *PROJECT_ROOT.joinpath("prompts").rglob("*.txt"),
        *PROJECT_ROOT.joinpath("prompts").rglob("*.md"),
        PROJECT_ROOT / "prompts/skill_prompt_mapping.yaml",
        *PROJECT_ROOT.joinpath("skills/builtin").rglob("SKILL.md"),
    ]
    assert {path.relative_to(PROJECT_ROOT).as_posix() for path in resource_paths} <= names

    install_dir = tmp_path / "installed"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(install_dir), str(wheel)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    code = """
import os
from pathlib import Path
import runtime.research.supervisor_graph_runtime_core
from core.prompt_resolver import PromptResolver
from core.skill_registry import SkillRegistry
from agents.research_supervisor_agent import _DEFAULT_SUPERVISOR_SOUL_PATH
from agents.research_qa_agent import RagReActQAWorker
from adapters.local_runtime import LocalLLMAdapter
from chains.graph_extraction_chain import GraphExtractionChain

assert Path(runtime.research.supervisor_graph_runtime_core.__file__).resolve().is_relative_to(
    Path(os.environ['PYTHONPATH']).resolve()
)
mapping = PromptResolver().load_mapping()
assert mapping['skills'], mapping
prompt = PromptResolver().resolve_prompt_path('answer_prompt_path')
assert prompt.is_file(), prompt
registry = SkillRegistry()
assert any(skill.name == 'research-qa' for skill in registry.scan())
assert _DEFAULT_SUPERVISOR_SOUL_PATH.is_file()
qa_agent = RagReActQAWorker(None, None, None)
assert qa_agent.decision_prompt_path.is_file()
assert qa_agent.synthesis_prompt_path.is_file()
assert GraphExtractionChain(llm=LocalLLMAdapter()).prompt_path.is_file()
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={"PYTHONPATH": str(install_dir)},
        check=True,
        capture_output=True,
        text=True,
    )
