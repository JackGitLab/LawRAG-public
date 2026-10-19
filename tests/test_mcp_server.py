from starlette.routing import Mount

from app.main import MCP_OPERATION_IDS, app, mcp_server


def test_operation_ids_are_declared() -> None:
    schema = app.openapi()
    operation_ids = {
        operation["operationId"]
        for path in schema["paths"].values()
        for operation in path.values()
    }

    assert set(MCP_OPERATION_IDS) <= operation_ids
    assert schema["paths"]["/api/chat/stream"]["post"]["operationId"] not in MCP_OPERATION_IDS


def test_mcp_tools_are_exactly_the_whitelist() -> None:
    assert {tool.name for tool in mcp_server.tools} == set(MCP_OPERATION_IDS)


def test_mcp_route_registered_before_frontend_mount() -> None:
    mcp_index = next(
        index for index, route in enumerate(app.routes) if route.path == "/mcp"
    )
    frontend_index = next(
        (
            index
            for index, route in enumerate(app.routes)
            if isinstance(route, Mount) and route.name == "frontend"
        ),
        None,
    )

    if frontend_index is not None:
        assert mcp_index < frontend_index
