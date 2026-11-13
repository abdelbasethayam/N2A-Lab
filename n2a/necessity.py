"""Model-relative tool-necessity labels derived from explicit no-tool trials."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from n2a.scoring import score_output


LABEL_RULES = ("single_failure", "any_failure", "majority_failure")


def _runs_from_payload(payload: Any) -> list[list[dict[str, Any]]]:
    if isinstance(payload, list):
        return [payload]
    if isinstance(payload, dict):
        run_keys = sorted(
            (key for key in payload if str(key).startswith("run_")),
            key=lambda key: int(str(key).split("_")[-1]) if str(key).split("_")[-1].isdigit() else str(key),
        )
        if run_keys and all(isinstance(payload[key], list) for key in run_keys):
            return [payload[key] for key in run_keys]
    raise ValueError("Expected a list of outputs or a dict containing run_0, run_1, ...")


def derive_necessity_labels(
    no_tool_payload: Any,
    *,
    label_rule: str = "single_failure",
) -> dict[str, Any]:
    """Create task-level labels and empirical no-tool success rates.

    single_failure: official When2Tool-style binary label for one no-tool run.
    any_failure / majority_failure: explicit repeated-run alternatives; report the
    rule because they are not interchangeable with the original one-run label.
    """
    if label_rule not in LABEL_RULES:
        raise ValueError(f"label_rule must be one of {LABEL_RULES}")
    runs = _runs_from_payload(no_tool_payload)
    by_id: dict[Any, list[bool]] = defaultdict(list)

    for run_index, outputs in enumerate(runs):
        seen = set()
        for item in outputs:
            task_id = item.get("id")
            if task_id is None:
                raise ValueError(f"Output in run {run_index} has no task id.")
            if task_id in seen:
                raise ValueError(f"Duplicate task {task_id} in no-tool run {run_index}.")
            seen.add(task_id)
            by_id[task_id].append(bool(score_output(item)["correct"]))

    labels = []
    for task_id in sorted(by_id, key=lambda x: str(x)):
        outcomes = by_id[task_id]
        if len(outcomes) != len(runs):
            raise ValueError(
                f"Task {task_id} is missing from one or more no-tool runs "
                f"({len(outcomes)}/{len(runs)})."
            )
        successes = sum(outcomes)
        success_rate = successes / len(outcomes)
        if label_rule == "single_failure":
            if len(outcomes) != 1:
                raise ValueError("single_failure requires exactly one no-tool outcome per task.")
            necessary = not outcomes[0]
        elif label_rule == "any_failure":
            necessary = successes < len(outcomes)
        else:
            necessary = success_rate < 0.5
        labels.append({
            "task_id": task_id,
            "tool_necessary": int(necessary),
            "no_tool_successes": successes,
            "no_tool_trials": len(outcomes),
            "no_tool_success_probability": success_rate,
        })

    return {
        "label_source": "model_no_tool_evaluation",
        "label_rule": label_rule,
        "n_runs": len(runs),
        "n_tasks": len(labels),
        "n_necessary": sum(row["tool_necessary"] for row in labels),
        "n_unnecessary": sum(1 - row["tool_necessary"] for row in labels),
        "labels": labels,
        "interpretation_warning": (
            "Necessity is model-, prompt-, decoding-, and evaluator-relative. "
            "Do not transfer these labels to another model or protocol."
        ),
    }
