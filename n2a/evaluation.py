"""Evaluation of model outputs against necessity labels and explicit utility schedules."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from n2a.metrics import classification_metrics, outcome_utility
from n2a.scoring import count_tool_calls, score_output


def _flatten_outputs(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        keys = sorted(
            (key for key in payload if str(key).startswith("run_")),
            key=lambda key: int(str(key).split("_")[-1]) if str(key).split("_")[-1].isdigit() else str(key),
        )
        if keys and all(isinstance(payload[key], list) for key in keys):
            return [item for key in keys for item in payload[key]]
    raise ValueError("Expected a list of output items or a run_0/run_1 dictionary.")


def evaluate_outputs(
    outputs_payload: Any,
    necessity_payload: dict[str, Any],
    *,
    success_value: float = 1.0,
    cost_per_call: float = 0.0,
    risk_per_call: float = 0.0,
) -> dict[str, Any]:
    """Evaluate realized model behavior; necessity labels must come from no-tool runs."""
    labels = {str(row["task_id"]): int(row["tool_necessary"]) for row in necessity_payload["labels"]}
    outputs = _flatten_outputs(outputs_payload)
    y_true, actions, successes, calls, rows = [], [], [], [], []
    for item in outputs:
        task_id = item.get("id")
        key = str(task_id)
        if key not in labels:
            continue
        scored = score_output(item)
        n_calls = count_tool_calls(item)
        necessary = labels[key]
        called = int(n_calls > 0)
        y_true.append(necessary)
        actions.append(called)
        successes.append(scored["correct"])
        calls.append(n_calls)
        rows.append({
            "task_id": task_id,
            "necessity_label": necessary,
            "tool_called": bool(called),
            "tool_calls": n_calls,
            "correct": scored["correct"],
            "prediction": scored["prediction"],
            "reference": scored["reference"],
        })

    if not rows:
        raise ValueError("No output task IDs matched the necessity-label file.")
    return {
        "n_matched_output_records": len(rows),
        "necessity_label_rule": necessity_payload["label_rule"],
        "decision_metrics": classification_metrics(y_true, actions),
        "outcome_utility": outcome_utility(
            successes, calls,
            success_value=success_value,
            cost_per_call=cost_per_call,
            risk_per_call=risk_per_call,
        ),
        "per_task": rows,
        "evidence_status": "REAL_MODEL_OUTPUTS_REQUIRED",
    }
