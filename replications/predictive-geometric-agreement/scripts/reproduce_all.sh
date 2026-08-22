#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
python3 "$SCRIPT_DIR/paper_harness.py" run-reproduce --project-dir "$PROJECT_DIR" "$@"
