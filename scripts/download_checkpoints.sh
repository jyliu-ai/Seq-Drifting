#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
python -m common.download --repo "${HF_REPO:-jyliuAI/Seq-Drifting}" "$@"

