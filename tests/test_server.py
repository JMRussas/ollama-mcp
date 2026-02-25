#!/usr/bin/env python3
#  ollama-mcp - Unit Tests
#
#  Tests for src/ollama_mcp/server.py using mocked HTTP responses.
#  Uses _init_from_dict() for clean initialization — no file I/O patching.
#
#  Depends on: ollama_mcp.server, pytest, pytest-asyncio
#  Used by:    Developer validation (`pytest tests/ -v`)

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ollama_mcp import server

# ---------------------------------------------------------------------------
# Test config — applied via _init_from_dict() instead of patching builtins
# ---------------------------------------------------------------------------

TEST_CONFIG = {
    "hosts": {
        "local": {"url": "http://localhost:11434", "label": "Test Local"},
        "server": {"url": "http://test-server:11434", "label": "Test Server"},
    },
    "default_model": "test-model",
    "embed_model": "test-embed",
    "timeout": 10.0,
    "max_attempts": 1,
    "retry_delay": 0.1,
}

server._init_from_dict(TEST_CONFIG)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_version_check():
    """Reset the lazy version check flag before each test."""
    server._version_checked = True  # skip version check in most tests


@pytest.fixture
def mock_request():
    with patch.object(server, "_request", new_callable=AsyncMock) as mock:
        yield mock


def _make_response(data: dict, status_code: int = 200) -> MagicMock:
    """Create a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    return resp


# ---------------------------------------------------------------------------
# _tok_per_sec helper
# ---------------------------------------------------------------------------

class TestTokPerSec:
    def test_normal(self):
        data = {"eval_count": 100, "eval_duration": 2_000_000_000}  # 2 seconds
        assert server._tok_per_sec(data) == "50.0"

    def test_zero_duration(self):
        data = {"eval_count": 100, "eval_duration": 0}
        assert server._tok_per_sec(data) == "?"

    def test_zero_count(self):
        data = {"eval_count": 0, "eval_duration": 1_000_000_000}
        assert server._tok_per_sec(data) == "?"

    def test_missing_keys(self):
        assert server._tok_per_sec({}) == "?"

    def test_both_zero(self):
        data = {"eval_count": 0, "eval_duration": 0}
        assert server._tok_per_sec(data) == "?"


# ---------------------------------------------------------------------------
# ollama_generate
# ---------------------------------------------------------------------------

class TestGenerate:
    @pytest.mark.asyncio
    async def test_success(self, mock_request):
        mock_request.return_value = _make_response({
            "response": "Hello world",
            "eval_count": 10,
            "eval_duration": 1_000_000_000,
        })
        result = await server.ollama_generate("test prompt")
        assert "Hello world" in result
        assert "10.0 tok/s" in result
        assert "Test Server" in result

    @pytest.mark.asyncio
    async def test_invalid_host(self):
        result = await server.ollama_generate("test", host="nonexistent")
        assert result.startswith("Error:")
        assert "nonexistent" in result

    @pytest.mark.asyncio
    async def test_empty_prompt(self):
        result = await server.ollama_generate("   ")
        assert result == "Error: prompt cannot be empty"

    @pytest.mark.asyncio
    async def test_connection_error(self, mock_request):
        mock_request.side_effect = httpx.ConnectError("refused")
        result = await server.ollama_generate("test")
        assert "Cannot connect" in result

    @pytest.mark.asyncio
    async def test_timeout(self, mock_request):
        mock_request.side_effect = httpx.TimeoutException("timeout")
        result = await server.ollama_generate("test")
        assert "timed out" in result

    @pytest.mark.asyncio
    async def test_http_status_error(self, mock_request):
        resp = MagicMock()
        resp.status_code = 404
        mock_request.side_effect = httpx.HTTPStatusError(
            "not found", request=MagicMock(), response=resp
        )
        result = await server.ollama_generate("test")
        assert "HTTP 404" in result

    @pytest.mark.asyncio
    async def test_missing_response_key(self, mock_request):
        mock_request.return_value = _make_response({})
        result = await server.ollama_generate("test")
        assert "? tok/s" in result
        assert "0 tokens" in result

    @pytest.mark.asyncio
    async def test_custom_timeout(self, mock_request):
        mock_request.return_value = _make_response({"response": "ok"})
        await server.ollama_generate("test", timeout=300)
        _, kwargs = mock_request.call_args
        assert kwargs["timeout"] == 300


# ---------------------------------------------------------------------------
# ollama_chat
# ---------------------------------------------------------------------------

class TestChat:
    @pytest.mark.asyncio
    async def test_success(self, mock_request):
        mock_request.return_value = _make_response({
            "message": {"content": "Hi there"},
            "eval_count": 5,
            "eval_duration": 500_000_000,
        })
        result = await server.ollama_chat([{"role": "user", "content": "hello"}])
        assert "Hi there" in result
        assert "10.0 tok/s" in result

    @pytest.mark.asyncio
    async def test_with_system(self, mock_request):
        mock_request.return_value = _make_response({
            "message": {"content": "response"},
        })
        await server.ollama_chat(
            [{"role": "user", "content": "hi"}],
            system="Be helpful",
        )
        call_kwargs = mock_request.call_args[1]
        messages = call_kwargs["json"]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful"

    @pytest.mark.asyncio
    async def test_invalid_host(self):
        result = await server.ollama_chat(
            [{"role": "user", "content": "hi"}],
            host="bad",
        )
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_empty_messages(self):
        result = await server.ollama_chat([])
        assert result == "Error: messages list cannot be empty"

    @pytest.mark.asyncio
    async def test_bad_message_format(self):
        result = await server.ollama_chat([{"role": "user"}])
        assert "role" in result and "content" in result

    @pytest.mark.asyncio
    async def test_non_dict_message(self):
        result = await server.ollama_chat(["not a dict"])
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_http_status_error(self, mock_request):
        resp = MagicMock()
        resp.status_code = 500
        mock_request.side_effect = httpx.HTTPStatusError(
            "server error", request=MagicMock(), response=resp
        )
        result = await server.ollama_chat([{"role": "user", "content": "hi"}])
        assert "HTTP 500" in result

    @pytest.mark.asyncio
    async def test_missing_response_keys(self, mock_request):
        mock_request.return_value = _make_response({})
        result = await server.ollama_chat([{"role": "user", "content": "hi"}])
        assert "? tok/s" in result


# ---------------------------------------------------------------------------
# ollama_embed
# ---------------------------------------------------------------------------

class TestEmbed:
    @pytest.mark.asyncio
    async def test_success(self, mock_request):
        embedding = [0.1, 0.2, 0.3]
        mock_request.return_value = _make_response({
            "embeddings": [embedding],
        })
        result = await server.ollama_embed("test text")
        parsed = json.loads(result)
        assert parsed["dimensions"] == 3
        assert parsed["embedding"] == embedding

    @pytest.mark.asyncio
    async def test_empty_embeddings(self, mock_request):
        mock_request.return_value = _make_response({"embeddings": []})
        result = await server.ollama_embed("test text")
        assert "empty embeddings" in result

    @pytest.mark.asyncio
    async def test_empty_text(self):
        result = await server.ollama_embed("  ")
        assert result == "Error: text cannot be empty"

    @pytest.mark.asyncio
    async def test_default_host_is_local(self, mock_request):
        mock_request.return_value = _make_response({
            "embeddings": [[0.1]],
        })
        result = await server.ollama_embed("test")
        assert "Test Local" not in result  # embed returns JSON, not metadata footer
        call_args = mock_request.call_args
        assert "localhost" in call_args[0][1]  # URL contains localhost


# ---------------------------------------------------------------------------
# ollama_list_models
# ---------------------------------------------------------------------------

class TestListModels:
    @pytest.mark.asyncio
    async def test_all_hosts(self, mock_request):
        mock_request.return_value = _make_response({
            "models": [
                {"name": "qwen2.5-coder:14b", "size": 14_000_000_000},
            ],
        })
        result = await server.ollama_list_models()
        assert "Test Local" in result
        assert "Test Server" in result
        assert "qwen2.5-coder:14b" in result

    @pytest.mark.asyncio
    async def test_single_host(self, mock_request):
        mock_request.return_value = _make_response({
            "models": [{"name": "tiny", "size": 1_000_000_000}],
        })
        result = await server.ollama_list_models(host="local")
        assert "Test Local" in result
        assert "Test Server" not in result

    @pytest.mark.asyncio
    async def test_offline_host(self, mock_request):
        mock_request.side_effect = httpx.ConnectError("refused")
        result = await server.ollama_list_models(host="local")
        assert "OFFLINE" in result

    @pytest.mark.asyncio
    async def test_missing_name_key(self, mock_request):
        mock_request.return_value = _make_response({
            "models": [{"size": 1_000_000_000}],
        })
        result = await server.ollama_list_models(host="local")
        assert "unknown" in result


# ---------------------------------------------------------------------------
# _ensure_version_checked
# ---------------------------------------------------------------------------

class TestVersionCheck:
    @pytest.mark.asyncio
    async def test_new_enough(self):
        server._version_checked = False
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"version": "0.5.1"}

        with patch.object(server._http_client, "get", new_callable=AsyncMock, return_value=mock_resp):
            await server._ensure_version_checked()
        # Should not raise, version is fine

    @pytest.mark.asyncio
    async def test_too_old(self, caplog):
        server._version_checked = False
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"version": "0.3.2"}

        with patch.object(server._http_client, "get", new_callable=AsyncMock, return_value=mock_resp):
            import logging
            with caplog.at_level(logging.WARNING):
                await server._ensure_version_checked()
        assert any("< 0.4.0" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_unreachable(self, caplog):
        server._version_checked = False
        with patch.object(server._http_client, "get", new_callable=AsyncMock, side_effect=Exception("down")):
            import logging
            with caplog.at_level(logging.WARNING):
                await server._ensure_version_checked()
        assert any("Could not check" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_only_runs_once(self):
        server._version_checked = False
        call_count = 0

        async def counting_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"version": "0.5.0"}
            return resp

        with patch.object(server._http_client, "get", side_effect=counting_get):
            await server._ensure_version_checked()
            await server._ensure_version_checked()
        # 2 hosts checked on first call, 0 on second
        assert call_count == 2


# ---------------------------------------------------------------------------
# _init_from_dict
# ---------------------------------------------------------------------------

class TestInitFromDict:
    def test_sets_config_values(self):
        custom = {
            "hosts": {"myhost": {"url": "http://example:11434", "label": "Custom"}},
            "default_model": "custom-model",
            "embed_model": "custom-embed",
            "timeout": 42.0,
            "max_attempts": 5,
            "retry_delay": 2.0,
        }
        server._init_from_dict(custom)
        assert "myhost" in server.HOSTS
        assert server.DEFAULT_MODEL == "custom-model"
        assert server.TIMEOUT == 42.0
        assert server.MAX_ATTEMPTS == 5
        assert server._initialized is True

        # Restore test config
        server._init_from_dict(TEST_CONFIG)

    def test_resets_version_checked(self):
        server._version_checked = True
        server._init_from_dict(TEST_CONFIG)
        assert server._version_checked is False
