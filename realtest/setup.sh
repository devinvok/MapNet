#!/usr/bin/env bash
# One-time setup for the real test (macOS / Linux).
#   bash realtest/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

READER="${MAPNET_MODEL:-qwen2.5:3b}"
EMBED="${MAPNET_EMBED_MODEL:-nomic-embed-text}"
PY="${PYTHON:-python3}"

echo "==> Python environment"
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ needed"'
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt pytest

echo "==> Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Install it, then re-run this script:"
  if [ "$(uname)" = "Darwin" ]; then
    echo "  brew install ollama      (or download the app from https://ollama.com/download)"
  else
    echo "  see https://ollama.com/download for the Linux install command"
  fi
  exit 1
fi
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Starting the Ollama server in the background..."
  nohup ollama serve >/dev/null 2>&1 &
  for _ in $(seq 1 20); do curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done
fi
echo "Pulling the reader model ($READER, about 2 GB) and the embedding model ($EMBED, about 300 MB)..."
ollama pull "$READER"
ollama pull "$EMBED"

echo "==> Tests"
python -m pytest -q

echo "==> Kubernetes docs"
python -c "from pathlib import Path; from realtest.corpus import fetch_k8s; fetch_k8s(Path('realtest/data'))"

cat <<EOF

Ready. Run the test with:
  source .venv/bin/activate
  python -m realtest.run
EOF
