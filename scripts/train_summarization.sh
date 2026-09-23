#!/usr/bin/env bash
set -euo pipefail
export DATASET=xsum
exec bash "$(dirname "$0")/train_seq2seq.sh" "$@"
