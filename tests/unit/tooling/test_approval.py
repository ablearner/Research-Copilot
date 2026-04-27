import pytest

from tooling.approval import ApprovalGate


def test_safe_commands_pass():
    gate = ApprovalGate()
    assert not gate._is_dangerous({"cmd": "ls -la"})
    assert not gate._is_dangerous({"cmd": "cat README.md"})
    assert not gate._is_dangerous({"query": "SELECT * FROM papers"})


def test_rm_rf_detected():
    gate = ApprovalGate()
    assert gate._is_dangerous({"cmd": "rm -rf /"})


def test_sudo_detected():
    gate = ApprovalGate()
    assert gate._is_dangerous({"cmd": "sudo apt install something"})


def test_drop_table_detected():
    gate = ApprovalGate()
    assert gate._is_dangerous({"query": "DROP TABLE users"})


def test_delete_from_detected():
    gate = ApprovalGate()
    assert gate._is_dangerous({"query": "DELETE FROM papers WHERE id=1"})


@pytest.mark.asyncio
async def test_check_auto_approve():
    gate = ApprovalGate(auto_approve_tools={"safe_tool"})
    assert await gate.check("safe_tool", {"cmd": "rm -rf /"})


@pytest.mark.asyncio
async def test_check_rejects_dangerous_no_callback():
    gate = ApprovalGate()
    assert not await gate.check("shell", {"cmd": "rm -rf /"})


@pytest.mark.asyncio
async def test_check_passes_safe():
    gate = ApprovalGate()
    assert await gate.check("search", {"query": "quantum computing"})
