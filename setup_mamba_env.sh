#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-clearocean}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v mamba >/dev/null 2>&1; then
  echo "mamba was not found. Install Miniforge/Mambaforge first."
  exit 1
fi

echo "Creating environment: ${ENV_NAME}"
mamba create -y -n "${ENV_NAME}" python=3.11 pip

mamba run -n "${ENV_NAME}" python -m pip install --upgrade pip wheel

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "CUDA detected via nvidia-smi. Installing CUDA wheels for PyTorch."
  mamba run -n "${ENV_NAME}" python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
  echo "No NVIDIA runtime detected. Installing default PyTorch wheels (CPU/MPS compatible)."
  mamba run -n "${ENV_NAME}" python -m pip install torch torchvision
fi

mamba run -n "${ENV_NAME}" python -m pip install -r "${ROOT_DIR}/ClearOcean/requirements.txt"

echo "Environment ${ENV_NAME} is ready."
echo "Run training: mamba run -n ${ENV_NAME} python ${ROOT_DIR}/ClearOcean/clearocean/train.py -opt ${ROOT_DIR}/ClearOcean/options/clearocean.yaml"
echo "Run inference: mamba run -n ${ENV_NAME} python ${ROOT_DIR}/ClearOcean/clearocean/infer.py -opt ${ROOT_DIR}/ClearOcean/options/clearocean.yaml"
