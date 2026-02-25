#!/usr/bin/env python3
#  ollama-mcp - Ollama MCP Server
#
#  Exposes local Ollama instances as tools for Claude Code.
#  Reads host configuration from config.json (machine-specific, gitignored).
#
#  Depends on: config.json (project root), mcp, httpx
#  Used by:    Claude Code (registered via `claude mcp add`)

import asyncio
import atexit
import json
import logging
import sys
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ollama")
log = logging.getLogger("ollama-mcp")
logging.basicConfig(
    level=logging.INFO,
    format="[ollama-mcp] %(levelname)s: %(message)s",
    stream=sys.stderr,
)

# ---------------------------------------------------------------------------
# State — populated by _init() or _init_from_dict()
# ---------------------------------------------------------------------------

HOSTS: dict = {}
DEFAULT_MODEL: str = "qwen2.5-coder:14b"
EMBED_MODEL: str = "nomic-embed-text"
TIMEOUT: float = 120.0
MAX_ATTEMPTS: int = 2
RETRY_DELAY: float = 0.5
_http_client: httpx.AsyncClient | None = None
_initialized: bool = False
_version_checked: bool = False
_atexit_registered: bool = False


def _apply_config(cfg: dict):
    """Apply a config dict to module globals. Shared by _init() and _init_from_dict()."""
    global HOSTS, DEFAULT_MODEL, EMBED_MODEL, TIMEOUT, MAX_ATTEMPTS, RETRY_DELAY
    global _http_client, _initialized, _atexit_registered

    HOSTS = {
        name: {"url": entry["url"], "label": entry["label"]}
        for name, entry in cfg["hosts"].items()
    }
    DEFAULT_MODEL = cfg.get("default_model", "qwen2.5-coder:14b")
    EMBED_MODEL = cfg.get("embed_model", "nomic-embed-text")
    TIMEOUT = cfg.get("timeout", 120.0)
    MAX_ATTEMPTS = cfg.get("max_attempts", 2)
    RETRY_DELAY = cfg.get("retry_delay", 0.5)

    _http_client = httpx.AsyncClient(timeout=TIMEOUT)
    if not _atexit_registered:
        atexit.register(_cleanup)
        _atexit_registered = True
    _initialized = True
    log.info(f"Loaded config: {len(HOSTS)} host(s), default model={DEFAULT_MODEL}")


def _init(config_path: Path | None = None):
    """Load config from file and initialize the server."""
    if _initialized:
        return

    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.parent / "config.json"

    if not config_path.exists():
        log.error(f"{config_path} not found. Run setup.sh to generate it.")
        sys.exit(1)

    with open(config_path) as f:
        _apply_config(json.load(f))


def _init_from_dict(cfg: dict):
    """Initialize the server from a config dict. Used by tests."""
    global _initialized, _version_checked
    _initialized = False
    _version_checked = False
    _apply_config(cfg)


def _cleanup():
    try:
        if _http_client is not None:
            asyncio.run(_http_client.aclose())
    except Exception:
        pass


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

async def _request(method: str, url: str, *, json: dict = None,
                   timeout: float = None, attempts: int = None) -> httpx.Response:
    """Make an HTTP request with retry + backoff."""
    if attempts is None:
        attempts = MAX_ATTEMPTS
    client = _http_client
    kwargs = {"json": json} if json is not None else {}
    if timeout is not None:
        kwargs["timeout"] = timeout

    for attempt in range(attempts):
        try:
            func = getattr(client, method.lower())
            resp = await func(url, **kwargs)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError:
            raise  # 4xx/5xx — don't retry, won't self-resolve
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            if attempt < attempts - 1:
                log.warning(f"Attempt {attempt + 1} failed ({type(e).__name__}), retrying in {RETRY_DELAY}s...")
                await asyncio.sleep(RETRY_DELAY)
                continue
            raise
    raise RuntimeError("unreachable")


def _tok_per_sec(data: dict) -> str:
    """Format tokens/sec from Ollama response metadata, or '?' if unavailable."""
    eval_count = data.get("eval_count", 0)
    eval_duration = data.get("eval_duration", 0)
    if eval_count and eval_duration:
        return f"{eval_count / (eval_duration / 1e9):.1f}"
    return "?"


# ---------------------------------------------------------------------------
# Lazy version check
# ---------------------------------------------------------------------------

async def _ensure_version_checked():
    """Check Ollama version on all hosts (once, on first tool call)."""
    global _version_checked
    if _version_checked:
        return
    _version_checked = True

    async def _check(name: str, info: dict):
        try:
            resp = await _http_client.get(f"{info['url']}/api/version", timeout=5.0)
            if resp.status_code == 200:
                version = resp.json().get("version", "unknown")
                log.info(f"{info['label']}: Ollama v{version}")
                parts = version.split(".")
                if len(parts) >= 2:
                    major, minor = int(parts[0]), int(parts[1])
                    if (major, minor) < (0, 4):
                        log.warning(
                            f"{info['label']}: Ollama v{version} < 0.4.0 — "
                            f"/api/embed may not work. Consider upgrading."
                        )
        except Exception:
            log.warning(f"{info['label']}: Could not check Ollama version (is it running?)")

    await asyncio.gather(*[_check(n, i) for n, i in HOSTS.items()])


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def ollama_generate(
    prompt: str,
    host: str = "server",
    model: str = "",
    system: str = "",
    timeout: float = 0,
) -> str:
    """Send a prompt to a local Ollama model and return the response.

    Use this for code generation, documentation drafts, quick questions,
    and tasks that don't require frontier-model reasoning.

    Args:
        prompt: The prompt to send to the model.
        host: Which machine to use — "local" (4090) or "server" (3090). Defaults to "server".
        model: Model name. Defaults to qwen2.5-coder:14b.
        system: Optional system prompt.
        timeout: Optional request timeout in seconds. 0 uses the global default.
    """
    await _ensure_version_checked()
    if host not in HOSTS:
        return f"Error: host must be one of {list(HOSTS.keys())}, got '{host}'"
    if not prompt.strip():
        return "Error: prompt cannot be empty"

    if not model:
        model = DEFAULT_MODEL

    url = f"{HOSTS[host]['url']}/api/generate"
    body = {"model": model, "prompt": prompt, "stream": False}
    if system:
        body["system"] = system

    try:
        resp = await _request("POST", url, json=body, timeout=timeout if timeout > 0 else None)
        data = resp.json()
        response = data.get("response", "")
        tok_per_sec = _tok_per_sec(data)
        log.info(f"generate: {HOSTS[host]['label']}/{model} — {data.get('eval_count', 0)} tokens, {tok_per_sec} tok/s")
        return f"{response}\n\n---\n[{HOSTS[host]['label']} | {model} | {data.get('eval_count', 0)} tokens | {tok_per_sec} tok/s]"
    except httpx.HTTPStatusError as e:
        return f"Error: {HOSTS[host]['label']} returned HTTP {e.response.status_code}"
    except httpx.ConnectError:
        return f"Error: Cannot connect to {HOSTS[host]['label']} at {HOSTS[host]['url']}"
    except httpx.TimeoutException:
        return f"Error: Request timed out connecting to {HOSTS[host]['label']}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
async def ollama_chat(
    messages: list[dict],
    host: str = "server",
    model: str = "",
    system: str = "",
    timeout: float = 0,
) -> str:
    """Send a multi-turn conversation to a local Ollama model.

    Use this when you need to have a back-and-forth with the local model,
    or when prior context matters for the response.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
                  Example: [{"role": "user", "content": "explain this code"}]
        host: Which machine — "local" (4090) or "server" (3090). Defaults to "server".
        model: Model name. Defaults to qwen2.5-coder:14b.
        system: Optional system prompt.
        timeout: Optional request timeout in seconds. 0 uses the global default.
    """
    await _ensure_version_checked()
    if host not in HOSTS:
        return f"Error: host must be one of {list(HOSTS.keys())}, got '{host}'"
    if not messages:
        return "Error: messages list cannot be empty"
    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            return "Error: each message must have 'role' and 'content' keys"

    if not model:
        model = DEFAULT_MODEL

    url = f"{HOSTS[host]['url']}/api/chat"
    body = {"model": model, "messages": [dict(m) for m in messages], "stream": False}
    if system:
        body["messages"].insert(0, {"role": "system", "content": system})

    try:
        resp = await _request("POST", url, json=body, timeout=timeout if timeout > 0 else None)
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        tok_per_sec = _tok_per_sec(data)
        log.info(f"chat: {HOSTS[host]['label']}/{model} — {data.get('eval_count', 0)} tokens, {tok_per_sec} tok/s")
        return f"{content}\n\n---\n[{HOSTS[host]['label']} | {model} | {data.get('eval_count', 0)} tokens | {tok_per_sec} tok/s]"
    except httpx.HTTPStatusError as e:
        return f"Error: {HOSTS[host]['label']} returned HTTP {e.response.status_code}"
    except httpx.ConnectError:
        return f"Error: Cannot connect to {HOSTS[host]['label']} at {HOSTS[host]['url']}"
    except httpx.TimeoutException:
        return f"Error: Request timed out connecting to {HOSTS[host]['label']}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
async def ollama_embed(
    text: str,
    host: str = "local",
    model: str = "",
) -> str:
    """Generate an embedding vector for the given text.

    Use this for semantic search, similarity comparisons, or RAG pipelines.
    Returns the embedding as a JSON array of floats.

    Args:
        text: The text to embed.
        host: Which machine — "local" (4090) or "server" (3090). Defaults to "local".
        model: Embedding model name. Defaults to nomic-embed-text.
    """
    await _ensure_version_checked()
    if host not in HOSTS:
        return f"Error: host must be one of {list(HOSTS.keys())}, got '{host}'"
    if not text.strip():
        return "Error: text cannot be empty"

    if not model:
        model = EMBED_MODEL

    url = f"{HOSTS[host]['url']}/api/embed"
    body = {"model": model, "input": text}

    try:
        resp = await _request("POST", url, json=body, timeout=30.0)
        data = resp.json()
        embeddings = data.get("embeddings", [])
        if not embeddings or not embeddings[0]:
            return "Error: Ollama returned empty embeddings. Check that the model supports embedding."
        embedding = embeddings[0]
        log.info(f"embed: {HOSTS[host]['label']}/{model} — {len(embedding)} dimensions")
        return json.dumps({"dimensions": len(embedding), "embedding": embedding})
    except httpx.HTTPStatusError as e:
        return f"Error: {HOSTS[host]['label']} returned HTTP {e.response.status_code}"
    except httpx.ConnectError:
        return f"Error: Cannot connect to {HOSTS[host]['label']} at {HOSTS[host]['url']}"
    except httpx.TimeoutException:
        return f"Error: Request timed out connecting to {HOSTS[host]['label']}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
async def ollama_list_models(host: str = "all") -> str:
    """List all available models on the Ollama instances.

    Args:
        host: "local", "server", or "all" (default).
    """
    await _ensure_version_checked()
    targets = list(HOSTS.keys()) if host == "all" else [host]

    async def _query_host(h: str) -> str:
        if h not in HOSTS:
            return f"{h}: invalid host"
        url = f"{HOSTS[h]['url']}/api/tags"
        try:
            resp = await _request("GET", url, timeout=10.0, attempts=1)
            models = resp.json().get("models", [])
            model_list = ", ".join(
                f"{m.get('name', 'unknown')} ({m.get('size', 0)/1e9:.1f}GB)" for m in models
            )
            return f"{HOSTS[h]['label']}: {model_list}"
        except httpx.HTTPStatusError as e:
            return f"{HOSTS[h]['label']}: HTTP {e.response.status_code}"
        except httpx.ConnectError:
            return f"{HOSTS[h]['label']}: OFFLINE"
        except httpx.TimeoutException:
            return f"{HOSTS[h]['label']}: TIMEOUT"
        except Exception as e:
            return f"{HOSTS[h]['label']}: Error — {e}"

    results = await asyncio.gather(*[_query_host(h) for h in targets])
    log.info(f"list_models: queried {len(targets)} host(s)")
    return "\n".join(results)


if __name__ == "__main__":
    _init()
    mcp.run(transport="stdio")
