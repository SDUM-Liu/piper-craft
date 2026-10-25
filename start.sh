#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
  UV_BIN=${UV_BIN:-$HOME/.local/bin/uv}
  "$UV_BIN" sync --python 3.11
fi
exec .venv/bin/python -m astra_vla.launcher "$@"
