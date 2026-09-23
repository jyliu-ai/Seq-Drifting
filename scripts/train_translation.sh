#!/usr/bin/env bash
set -euo pipefail
export DATASET=wmt14_de_en
exec bash "$(dirname "$0")/train_seq2seq.sh" "$@"
