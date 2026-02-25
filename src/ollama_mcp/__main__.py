"""Entry point for `python -m ollama_mcp`."""

from ollama_mcp.server import _init, mcp

_init()
mcp.run(transport="stdio")
