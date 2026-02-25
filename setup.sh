#!/usr/bin/env bash
#  ollama-mcp - Setup Script
#
#  Generates machine-specific config.json and registers the MCP server
#  with Claude Code. Run once after cloning.
#
#  Usage: bash setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/config.json"
SERVER="$SCRIPT_DIR/src/ollama_mcp/server.py"
VENV="$SCRIPT_DIR/.venv"

echo "=== Ollama MCP Setup ==="
echo ""

# --- Step 1: Create venv and install deps ---
if [ ! -d "$VENV" ]; then
    echo "[venv] Creating virtual environment..."
    python -m venv "$VENV"
fi
source "$VENV/Scripts/activate"
echo "[venv] Installing dependencies..."
pip install -q "$SCRIPT_DIR"
pip install -q "$SCRIPT_DIR[dev]"

# --- Step 2: Generate config.json ---
if [ -f "$CONFIG" ]; then
    echo "[config] config.json already exists, skipping."
    echo "         Delete it and re-run to regenerate."
else
    echo "Configure Ollama hosts for this machine."
    echo ""
    read -rp "This machine's GPU label (e.g. '4090' or '3090'): " LOCAL_GPU
    read -rp "Remote machine's IP (leave empty for local-only): " REMOTE_IP

    if [ -z "$REMOTE_IP" ]; then
        cat > "$CONFIG" <<EOF
{
  "hosts": {
    "local": {
      "url": "http://localhost:11434",
      "label": "Local ($LOCAL_GPU)"
    }
  },
  "default_model": "qwen2.5-coder:14b",
  "embed_model": "nomic-embed-text",
  "timeout": 120.0,
  "max_attempts": 2,
  "retry_delay": 0.5
}
EOF
    else
        read -rp "Remote machine's GPU label: " REMOTE_GPU
        cat > "$CONFIG" <<EOF
{
  "hosts": {
    "local": {
      "url": "http://localhost:11434",
      "label": "Local ($LOCAL_GPU)"
    },
    "server": {
      "url": "http://$REMOTE_IP:11434",
      "label": "Remote ($REMOTE_GPU)"
    }
  },
  "default_model": "qwen2.5-coder:14b",
  "embed_model": "nomic-embed-text",
  "timeout": 120.0,
  "max_attempts": 2,
  "retry_delay": 0.5
}
EOF
    fi
    echo "[config] Created $CONFIG"
fi

# --- Step 3: Register MCP server with Claude Code ---
echo "[mcp] Registering ollama MCP server..."
claude mcp add ollama -s user -- "$(cygpath -w "$VENV/Scripts/python")" "$(cygpath -w "$SERVER")"
echo "[mcp] Done."

echo ""
echo "=== Setup complete ==="
echo "Restart Claude Code to pick up the new MCP server."
