# ollama-mcp

An MCP (Model Context Protocol) server that exposes local [Ollama](https://ollama.com) instances as tools for [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

Lets Claude offload code generation, drafts, embeddings, and quick questions to your local GPUs.

## Setup

1. Run the setup script:
   ```bash
   bash setup.sh
   ```
   This creates a venv, installs dependencies, generates a machine-specific `config.json`, and registers the MCP server with Claude Code.

   > **Note:** `setup.sh` uses `cygpath` and targets Windows (Git Bash / MSYS2). On Linux/macOS, replace the `cygpath -w` calls with the paths directly, or register manually:
   > ```bash
   > claude mcp add ollama -s user -- /path/to/.venv/bin/python /path/to/src/ollama_mcp/server.py
   > ```

2. Restart Claude Code.

## Tools

| Tool | Description |
|------|-------------|
| `ollama_generate` | Single-turn prompt → response |
| `ollama_chat` | Multi-turn conversation |
| `ollama_embed` | Generate embedding vectors |
| `ollama_list_models` | List models on your Ollama instances |

## Configuration

Copy `config.example.json` to `config.json` and fill in your machine details, or let `setup.sh` generate it interactively.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) 0.4.0+ running on at least one machine
- Claude Code with MCP support

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## About

Extracted from a private developer infrastructure repo and published as a standalone tool. This server runs daily as part of a multi-project AI development workflow spanning game engines, RAG pipelines, and task orchestration — see [mcp-rag](https://github.com/JMRussas/mcp-rag) and [orchestration-engine](https://github.com/JMRussas/orchestration-engine) for projects that use it.

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `config.json not found` | Setup not run | Run `bash setup.sh` |
| 404 on embed calls | Ollama < 0.4.0 | Upgrade Ollama (`ollama update`) |
| `Cannot connect to...` | Ollama not running on target host | Start Ollama: `ollama serve` or check Docker |
| `Request timed out` | Large model / slow hardware | Increase `timeout` in config.json, or pass `timeout` parameter |
| `OFFLINE` in list_models | Host unreachable | Check network, firewall, Ollama port 11434 |
| `cygpath: command not found` | Running setup.sh on Linux/macOS | See setup note above |
