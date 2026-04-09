#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BEAM_SIZE="${BEAM_SIZE:-5}"
SAMPLE_SEED="${SAMPLE_SEED:-}"
RANDOM_LIMIT="${RANDOM_LIMIT:-}"
STRATIFIED_LIMIT="${STRATIFIED_LIMIT:-100}"

cd "${REPO_ROOT}"

HAS_BEAM_SIZE=false
HAS_RANDOM_LIMIT=false
HAS_STRATIFIED_LIMIT=false
HAS_SAMPLE_SEED=false
LOG_FILE=""
FORWARD_ARGS=()
EXPECT_LOG_FILE_VALUE=false

ARGS=("$@")
for arg in "${ARGS[@]}"; do
  if [ "${EXPECT_LOG_FILE_VALUE}" = true ]; then
    LOG_FILE="${arg}"
    EXPECT_LOG_FILE_VALUE=false
    continue
  fi

  case "${arg}" in
    --beam_size|--beam_size=*)
      HAS_BEAM_SIZE=true
      FORWARD_ARGS+=("${arg}")
      ;;
    --random_limit|--random_limit=*)
      HAS_RANDOM_LIMIT=true
      FORWARD_ARGS+=("${arg}")
      ;;
    --stratified_limit|--stratified_limit=*)
      HAS_STRATIFIED_LIMIT=true
      FORWARD_ARGS+=("${arg}")
      ;;
    --sample_seed|--sample_seed=*)
      HAS_SAMPLE_SEED=true
      FORWARD_ARGS+=("${arg}")
      ;;
    --log_file)
      EXPECT_LOG_FILE_VALUE=true
      ;;
    --log_file=*)
      LOG_FILE="${arg#--log_file=}"
      ;;
    *)
      FORWARD_ARGS+=("${arg}")
      ;;
  esac
done

if [ "${EXPECT_LOG_FILE_VALUE}" = true ]; then
  echo "Error: --log_file requires a file path." >&2
  exit 1
fi

if [ "${HAS_RANDOM_LIMIT}" = true ] && [ "${HAS_STRATIFIED_LIMIT}" = true ]; then
  echo "Error: --random_limit and --stratified_limit cannot be used together." >&2
  exit 1
fi

if [ "${HAS_RANDOM_LIMIT}" = true ]; then
  effective_random_limit="command line"
else
  effective_random_limit="${RANDOM_LIMIT:-full dataset}"
fi

if [ "${HAS_STRATIFIED_LIMIT}" = true ]; then
  effective_stratified_limit="command line"
else
  effective_stratified_limit="${STRATIFIED_LIMIT:-unset}"
fi

if [ "${HAS_SAMPLE_SEED}" = true ]; then
  effective_sample_seed="command line"
else
  effective_sample_seed="${SAMPLE_SEED:-unset}"
fi

echo "Running main.py with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, beam_size=${BEAM_SIZE}, random_limit=${effective_random_limit}, stratified_limit=${effective_stratified_limit}, sample_seed=${effective_sample_seed}, log_file=${LOG_FILE:-unset}"

CMD=(python main.py)

if [ "${HAS_BEAM_SIZE}" = false ]; then
  CMD+=(--beam_size "${BEAM_SIZE}")
fi

if [ "${HAS_RANDOM_LIMIT}" = false ] && [ "${HAS_STRATIFIED_LIMIT}" = false ] && [ -n "${RANDOM_LIMIT}" ]; then
  CMD+=(--random_limit "${RANDOM_LIMIT}")
fi

if [ "${HAS_STRATIFIED_LIMIT}" = false ] && [ -n "${STRATIFIED_LIMIT}" ]; then
  CMD+=(--stratified_limit "${STRATIFIED_LIMIT}")
fi

if [ "${HAS_SAMPLE_SEED}" = false ] && [ -n "${SAMPLE_SEED}" ] && { [ "${HAS_RANDOM_LIMIT}" = true ] || [ "${HAS_STRATIFIED_LIMIT}" = true ] || [ -n "${RANDOM_LIMIT}" ] || [ -n "${STRATIFIED_LIMIT}" ]; }; then
  CMD+=(--sample_seed "${SAMPLE_SEED}")
fi

if [ -n "${LOG_FILE}" ]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" "${FORWARD_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${CMD[@]}" "${FORWARD_ARGS[@]}"
fi
