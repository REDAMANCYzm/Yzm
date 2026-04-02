#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
#保留多少候选
BEAM_SIZE="${BEAM_SIZE:-5}"
#限制随机测试N个数据
RANDOM_LIMIT="${RANDOM_LIMIT:-20}"

cd "${REPO_ROOT}"

echo "Running main.py with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, beam_size=${BEAM_SIZE}, random_limit=${RANDOM_LIMIT:-full dataset}"

CMD=(python main.py --beam_size "${BEAM_SIZE}")

if [ -n "${RANDOM_LIMIT}" ]; then
  CMD+=(--random_limit "${RANDOM_LIMIT}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" "$@"
