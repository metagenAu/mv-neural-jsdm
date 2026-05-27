#!/usr/bin/env bash
# Smoke test: integration test runs an end-to-end fit of the synthetic dataset.
# Exits 0 on pass, non-zero on fail.
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v uv >/dev/null 2>&1; then
  uv run pytest tests/integration/test_smoke_end_to_end.py -x -v -s
else
  python -m pytest tests/integration/test_smoke_end_to_end.py -x -v -s
fi
