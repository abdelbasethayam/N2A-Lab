#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "\${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="qwen3-4b-instruct"
MODE="pilot"
PILOT_N=100
STEPS="1-5"
MODEL_REVISION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --pilot)
      MODE="pilot"
      if [[ "\${2:-}" =~ ^[0-9]+$ ]]; then PILOT_N="$2"; shift 2; else shift; fi
      ;;
    --full) MODE="full"; shift ;;
    --steps) STEPS="$2"; shift 2 ;;
    --revision) MODEL_REVISION="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--model qwen3-4b-instruct] [--pilot 100 | --full] [--steps 1-5] [--revision HF_COMMIT_SHA]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: a Linux NVIDIA GPU runtime is required for the official vLLM pipeline." >&2
  echo "Use a cloud GPU runtime (recommended: 24 GB+ VRAM for Qwen3-4B) and install requirements-research.txt." >&2
  exit 3
fi

if ! python -c "import torch, transformers, vllm, datasets" >/dev/null 2>&1; then
  echo "ERROR: research dependencies are missing." >&2
  echo "Run: pip install -r requirements-research.txt" >&2
  exit 4
fi

bash "\${ROOT_DIR}/scripts/bootstrap_when2tool.sh"
UPSTREAM_DIR="\${ROOT_DIR}/.cache/when2tool"
ARTIFACT_DIR="\${ROOT_DIR}/artifacts/when2tool"
mkdir -p "\${ARTIFACT_DIR}/outputs" "\${ARTIFACT_DIR}/probe_data" "\${ARTIFACT_DIR}/figures"

# Keep generated artifacts outside the pinned upstream checkout.
ln -sfn "\${ARTIFACT_DIR}/outputs" "\${UPSTREAM_DIR}/outputs"
ln -sfn "\${ARTIFACT_DIR}/probe_data" "\${UPSTREAM_DIR}/probe_data"
ln -sfn "\${ARTIFACT_DIR}/figures" "\${UPSTREAM_DIR}/figures"

cd "\${UPSTREAM_DIR}"
echo "Model: \${MODEL}"
echo "Pinned source: $(git rev-parse HEAD)"

if [[ "\${MODE}" == "pilot" ]]; then
  OUT="\${ARTIFACT_DIR}/outputs/\${MODEL}_pilot_all_prompts_no_reasoning.json"
  echo "Running a \${PILOT_N}-task smoke/pilot evaluation (not a full reproduction)."
  python src/run_eval.py \
    --model_path "\${MODEL}" \
    --backend vllm \
    --data_path ./data/tasks_v1_test.json \
    --output_path "\${OUT}" \
    --max_samples "\${PILOT_N}" \
    --n_runs 1 \
    --max_rounds 10 \
    --max_new_tokens 1024 \
    --tensor_parallel_size 1 \
    --max_model_len 8192 \
    --record_mode lite \
    --prompt_mode all \
    --reasoning_mode no_reasoning
else
  echo "Running the official five-step single-hop pipeline. This is GPU-intensive."
  bash ./run_pipeline.sh --steps "\${STEPS}" "\${MODEL}"
fi

echo "Finished. Artifacts are in: \${ARTIFACT_DIR}"
