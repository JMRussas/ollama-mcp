# ollama-mcp

MCP server that exposes local Ollama instances as tools for Claude Code.

## Quick Reference

| What | Command |
|------|---------|
| Setup | `bash setup.sh` (creates venv, installs deps, registers MCP) |
| Run server | `python -m ollama_mcp` or `.venv/Scripts/python src/ollama_mcp/server.py` (stdio, launched by Claude Code) |
| Run tests | `pytest tests/ -v` (from venv) |
| Install deps | `pip install .` (runtime) / `pip install .[dev]` (+ tests) |

## Project Structure

```
ollama-mcp/
  pyproject.toml          Project metadata, dependencies, pytest config
  config.json             Machine-specific host config (gitignored)
  config.example.json     Config template
  setup.sh                Creates venv, generates config, registers MCP
  LICENSE                 MIT license
  src/
    ollama_mcp/
      __init__.py
      __main__.py         Entry point for `python -m ollama_mcp`
      server.py           MCP server — 4 tools: generate, chat, embed, list_models
  tests/
    __init__.py
    test_server.py         Unit tests (pytest, mocked httpx)
  .github/
    workflows/ci.yml      GitHub Actions — runs pytest on push/PR
  .claude/
    gotchas.md            Gotchas & pitfalls
```

## How It Works

```
Claude Code → stdio → server.py → httpx → Ollama API (local/remote)
```

The server reads `config.json` at startup to discover Ollama hosts (local + remote machines). Each tool call makes an async HTTP request to the appropriate Ollama instance and returns the formatted result.

## Tools Exposed

| Tool | Purpose | Default Host |
|------|---------|--------------|
| `ollama_generate` | Single-turn prompt completion | server |
| `ollama_chat` | Multi-turn conversation | server |
| `ollama_embed` | Text embedding (returns vector) | local |
| `ollama_list_models` | List available models | all |

## Configuration

`config.json` maps host aliases to Ollama URLs:

```json
{
  "hosts": {
    "local": { "url": "http://localhost:11434", "label": "Local (4090)" },
    "server": { "url": "http://REMOTE_IP:11434", "label": "Server (3090)" }
  },
  "default_model": "qwen2.5-coder:14b",
  "embed_model": "nomic-embed-text",
  "timeout": 120.0,
  "max_attempts": 2,
  "retry_delay": 0.5
}
```

## Conventions

- Config values are never hardcoded — everything comes from `config.json`
- Server communicates via stdio (MCP transport)
- Tool responses include metadata footer (model, token count, speed)
- Errors return user-friendly strings, never raise exceptions to MCP
