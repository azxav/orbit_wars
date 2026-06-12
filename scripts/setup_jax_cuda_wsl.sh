#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv-jax
source .venv-jax/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-jax-cuda.txt
python scripts/check_jax_cuda.py --require-cuda
