#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "\${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="qwen3-4b-instruct"
SPLIT="test"
RUNS=10
MAX_SAMPLES=0
TEMPERATURE=0.7

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--model qwen3-4b-instruct] [--split train|test] [--runs 10] [--max-samples 100] [--max-model-len 32768]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$SPLIT" != "train" && "$SPLIT" != "test" ]]; then
  echo "ERROR: --split must be train or test." >&2
  exit 2
fi
if [[ "$RUNS" -lt 2 ]]; then
  echo "ERROR: repeated signal collection requires at least two runs." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: a Linux NVIDIA GPU runtime is required." >&2
  exit 3
fi
if ! python -c "import torch, transformers, vllm, datasets" >/dev/null 2>&1; then
  echo "ERROR: install research dependencies first: pip install -r requirements-research.txt" >&2
  exit 4
fi

bash "\${ROOT_DIR}/scripts/bootstrap_when2tool.sh"
UPSTREAM_DIR="\${ROOT_DIR}/.cache/when2tool"
ARTIFACT_DIR="\${ROOT_DIR}/artifacts/when2tool/repeated"
mkdir -p "\${ARTIFACT_DIR}"

if [[ "$SPLIT" == "train" ]]; then
  DATA_PATH="./data/tasks_v1_train.json"
else
  DATA_PATH="./data/tasks_v1_test.json"
fi

NO_TOOL_OUT="\${ARTIFACT_DIR}/\${MODEL}_\${SPLIT}_hard_no_tool_\${RUNS}runs.json"
FORCE_TOOL_OUT="\${ARTIFACT_DIR}/\${MODEL}_\${SPLIT}_force_tool_\${RUNS}runs.json"

python "\${ROOT_DIR}/scripts/run_repeated_no_tool.py" \
  --upstream-dir "\${UPSTREAM_DIR}" \
  --model-path "\${MODEL}" \
  --data-path "\${DATA_PATH}" \
  --output "\${NO_TOOL_OUT}" \
  --runs "\${RUNS}" \
  --max-samples "\${MAX_SAMPLES}" \
  --temperature "\${TEMPERATURE}" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN"

python "\${UPSTREAM_DIR}/src/run_eval.py" \
  --model_path "\${MODEL}" \
  --backend vllm \
  --data_path "\${DATA_PATH}" \
  --output_path "\${FORCE_TOOL_OUT}" \
  --max_samples "\${MAX_SAMPLES}" \
  --n_runs "\${RUNS}" \
  --temperature "\${TEMPERATURE}" \
  --max_rounds 12 \
  --max_new_tokens 2048 \
  --tensor_parallel_size 1 \
  --max_model_len 8192 \
  --record_mode lite \
  --prompt_mode force_tool \
  --reasoning_mode no_reasoning

echo "Repeated-trial artifacts:"
echo "  hard-no-tool: \${NO_TOOL_OUT}"
echo "  force-tool:   \${FORCE_TOOL_OUT}"
echo "Use label-rule any_failure or majority_failure and preserve the rule in the report."
