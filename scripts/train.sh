#!/usr/bin/env bash
# Train and evaluate with the default config, or pass another one: scripts/train.sh my_config.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
python main.py --config "${1:-config.yaml}"
