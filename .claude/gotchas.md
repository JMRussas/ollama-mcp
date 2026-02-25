# Gotchas & Pitfalls

## Async cleanup on shutdown

`httpx.AsyncClient.aclose()` is async. The `atexit` handler uses `asyncio.run()` to
await it in a fresh event loop. This works because FastMCP's event loop has already
stopped by the time `atexit` fires. If the server's lifecycle changes (e.g. embedded
in a larger async app), this approach may need revisiting.

## Ollama embed API version

The server uses the newer `/api/embed` endpoint (input/embeddings keys) instead of the
deprecated `/api/embeddings` (prompt/embedding keys). Requires Ollama 0.4.0+. If you
see 404 errors on embed calls, upgrade Ollama.

## MAX_ATTEMPTS vs retries

`MAX_ATTEMPTS` is *total attempts*, not retry count. `MAX_ATTEMPTS=2` means 1 original
request + 1 retry. The `_request()` parameter is also named `attempts` for clarity.

## setup.sh is Windows-only

`setup.sh` uses `cygpath -w` to convert Unix paths to Windows paths for Claude Code
registration. This will fail on Linux/macOS. If porting, replace the `cygpath` call
with `$SERVER` directly.

## Embedding vectors consume context window

`ollama_embed` returns the full float array (768+ dimensions for nomic-embed-text). Each
embed call consumes significant Claude Code context. This is intentional — embeddings are
meant for programmatic consumption by RAG pipelines — but be aware of the cost if making
many embed calls in a single session.

## Tool errors are strings, not exceptions

All tools catch exceptions and return error strings. This prevents MCP from seeing
unhandled errors, but it also means callers can't distinguish errors from valid
responses programmatically. The convention is that error responses start with `"Error:"`.

## Pinned vs flexible dependencies

Dependencies are pinned to exact versions in `pyproject.toml`. When upgrading, test
all four tools and the version check before committing. Key compatibility concern:
the `mcp` library's FastMCP API may change between major versions.

## Deferred initialization

`server.py` has no module-level side effects. All config loading and HTTP client
creation happens in `_init()` (from file) or `_init_from_dict()` (from dict). Tests
use `_init_from_dict(TEST_CONFIG)` for clean setup with no file I/O patching needed.
The `if __name__` block and `__main__.py` both call `_init()` before `mcp.run()`.
