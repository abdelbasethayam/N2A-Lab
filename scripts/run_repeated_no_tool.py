"""Collect repeated hard-no-tool trials with the pinned official When2Tool runner."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", required=True)
    parser.add_argument("--model-path", default="qwen3-4b-instruct")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-rounds", type=int, default=12)
    args = parser.parse_args()

    if args.runs < 2:
        raise ValueError("Repeated-trial collection requires at least two runs.")
    upstream = Path(args.upstream_dir).resolve()
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "src"))

    from model import AgentModel
    from utils import detect_tool_format, evaluate_batched, load_tasks_from_path

    tasks = load_tasks_from_path(args.data_path)
    if args.max_samples > 0:
        tasks = tasks[:args.max_samples]
    if not tasks:
        raise ValueError("No tasks were loaded.")

    model = AgentModel(
        model_path=args.model_path,
        backend="vllm",
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enable_thinking=False,
    )
    tool_format = detect_tool_format(args.model_path)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "benchmark": "When2Tool",
            "mode": "hard_no_tool",
            "model_path": args.model_path,
            "data_path": args.data_path,
            "n_tasks": len(tasks),
            "n_runs_requested": args.runs,
            "temperature": args.temperature,
            "max_rounds": args.max_rounds,
            "max_new_tokens": args.max_new_tokens,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "collected_at_utc": datetime.now(timezone.utc).isoformat(),
            "upstream_runner": "Trustworthy-ML-Lab/when2tool",
            "reproducibility_note": (
                "The pinned upstream AgentModel does not expose a vLLM per-run seed in its "
                "public wrapper. Preserve these raw independent trials and runtime metadata."
            ),
        }
    }

    for run_index in range(args.runs):
        print(f"\nHard-no-tool trial {run_index + 1}/{args.runs} on {len(tasks)} tasks")
        outputs = evaluate_batched(
            tasks,
            model,
            max_rounds=args.max_rounds,
            record_mode="lite",
            prompt_mode="hard_no_tool",
            require_reasoning=False,
            tool_format=tool_format,
        )
        payload[f"run_{run_index}"] = outputs
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output_path)
        print(f"Saved completed trial {run_index + 1} to {output_path}")

    print(f"Completed {args.runs} repeated hard-no-tool trials.")
    print(f"Tasks: {len(tasks)}; output: {output_path}")


if __name__ == "__main__":
    main()
