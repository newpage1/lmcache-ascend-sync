#!/usr/bin/env bash
# Run lmcache-ascend-sync locally
# Usage:
#   ./run_local.sh                    # Full sync check
#   ./run_local.sh --check-only       # Only check for new releases
#   ./run_local.sh --analyze v0.4.4   # Analyze specific version

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check dependencies
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Install deps if needed
if ! python3 -c "import yaml, requests, anthropic" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Check for API key
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "Warning: ANTHROPIC_API_KEY not set. PR generation will fail."
    echo "Export it first: export ANTHROPIC_API_KEY=sk-..."
fi

# Check for GitHub auth
if ! gh auth status &>/dev/null 2>&1; then
    echo "Warning: GitHub CLI not authenticated. PR creation will fail."
    echo "Run: gh auth login"
fi

# Run the monitor
python3 -m src.sync_monitor "$@"
