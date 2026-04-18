#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-train}"
CONFIG="${2:-ClearOcean/options/clearocean.yaml}"

if [[ "${MODE}" == "train" ]]; then
  python ClearOcean/clearocean/train.py -opt "${CONFIG}"
elif [[ "${MODE}" == "infer" ]]; then
  python ClearOcean/clearocean/infer.py -opt "${CONFIG}"
else
  echo "Unknown mode: ${MODE}"
  echo "Usage: ./run.sh [train|infer] [path/to/config.yaml]"
  exit 1
fi
