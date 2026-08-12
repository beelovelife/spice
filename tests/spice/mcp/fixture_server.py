from mcp.server.fastmcp import FastMCP

server = FastMCP("spice-test")


@server.tool()
def echo(value: str) -> str:
    """Echo a string."""
    return value


if __name__ == "__main__":
    server.run(transport="stdio")
