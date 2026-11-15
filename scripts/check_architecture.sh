#!/usr/bin/env bash
# Architecture gates. Offline checks always run; test suites run with --with-tests.
#   scripts/check_architecture.sh               # line budget + online->service ban
#   scripts/check_architecture.sh --with-tests  # + pipeline architecture pytest,
#                                               #   simulator pytest, game npm test
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 scripts/check_architecture.py

if [[ "${1:-}" == "--with-tests" ]]; then
  (cd pipeline && uv run pytest tests/test_architecture.py -q)
  (cd simulator && uv run pytest -q)
  (cd game && npm ci --no-audit --no-fund && npm test)
fi

echo "architecture checks passed"
