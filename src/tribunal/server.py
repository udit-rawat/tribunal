"""MCP server. Exposes `verify_claim` so Claude Desktop / Cursor can fact-check inline.

Run: `tribunal-mcp`  (stdio transport)
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .graph import verify_claim as _verify

mcp = FastMCP("Tribunal")


@mcp.tool()
def verify_claim(claim: str) -> dict:
    """Fact-check a factual claim and return a grounded verdict.

    Returns a dict with: verdict (True/False/Misleading/Unverifiable), confidence (0-1),
    summary, reasoning, and citations (quotes + source URLs).
    """
    return _verify(claim)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
