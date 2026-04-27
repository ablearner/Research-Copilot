from mcp.client.registry import MCPClientRegistry
from mcp.client.stdio_client import StdioMCPClient
from mcp.client.http_client import HttpMCPClient


def test_register_from_config_stdio():
    reg = MCPClientRegistry()
    registered = reg.register_from_config({
        "test-stdio": {
            "transport": "stdio",
            "command": "/usr/bin/echo",
            "args": ["hello"],
            "timeout": 30,
        }
    })
    assert "test-stdio" in registered
    client = reg.get_server("test-stdio")
    assert isinstance(client, StdioMCPClient)


def test_register_from_config_http():
    reg = MCPClientRegistry()
    registered = reg.register_from_config({
        "test-http": {
            "transport": "http",
            "url": "http://localhost:8080",
            "headers": {"Authorization": "Bearer test"},
        }
    })
    assert "test-http" in registered
    client = reg.get_server("test-http")
    assert isinstance(client, HttpMCPClient)


def test_register_from_config_unknown_transport():
    reg = MCPClientRegistry()
    registered = reg.register_from_config({
        "bad": {"transport": "grpc", "command": "foo"},
    })
    assert registered == []


def test_register_from_config_multiple():
    reg = MCPClientRegistry()
    registered = reg.register_from_config({
        "s1": {"transport": "stdio", "command": "/bin/true"},
        "s2": {"transport": "http", "url": "http://localhost:9090"},
    })
    assert len(registered) == 2
    assert set(reg.list_servers()) == {"s1", "s2"}
