"""Validation-only threshold selection for score-based necessity policies."""
from __future__ import annotations

from typing import Any

from n2a.metrics import classification_metrics


def _score_for_policy(record: dict[str, Any], policy_id: str) -> float | None:
    if policy_id == "P2":
        value = record.get("no_tool_success_probability")
        return None if value is None else 1.0 - float(value)
    if policy_id == "P3":
        value = record.get("verbalized_necessity_score")
        return None if value is None else float(value)
    if policy_id == "P4":
        value = record.get("surface_necessity_score")
        if value is not None:
            return float(value)
        return {"easy": 0.20, "medium": 0.50, "hard": 0.80}.get(record.get("difficulty"))
    if policy_id == "P5":
        value = record.get("probe_necessity_probability")
        return None if value is None else float(value)
    raise ValueError("Threshold calibration supports P2, P3, P4, and P5 only.")


def calibrate_threshold(
    records: list[dict[str, Any]],
    *,
    policy_id: str,
    metric: str = "f1",
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """Choose a threshold on a designated training/validation table.

    Ties are broken toward the higher threshold (fewer calls), a conservative
    choice when tool use has non-zero cost.
    """
    if policy_id not in ("P2", "P3", "P4", "P5"):
        raise ValueError("policy_id must be one of P2, P3, P4, P5")
    if metric not in ("f1", "balanced_accuracy", "mcc"):
        raise ValueError("metric must be f1, balanced_accuracy, or mcc")
    if not records:
        raise ValueError("At least one record is required.")

    eligible = []
    for record in records:
        if "necessity_label" not in record:
            raise ValueError("Every record must contain necessity_label.")
        score = _score_for_policy(record, policy_id)
        if score is not None:
            label = int(record["necessity_label"])
            if label not in (0, 1) or not 0.0 <= score <= 1.0:
                raise ValueError("Labels must be binary and scores must be in [0, 1].")
            eligible.append((label, score))
    if not eligible:
        raise ValueError(f"No eligible training records for {policy_id}.")

    grid = thresholds or [i / 100 for i in range(101)]
    if any(not 0.0 <= t <= 1.0 for t in grid):
        raise ValueError("All candidate thresholds must be in [0, 1].")

    labels = [label for label, _ in eligible]
    scores = [score for _, score in eligible]
    candidates = []
    for threshold in sorted(set(float(t) for t in grid)):
        actions = [int(score >= threshold) for score in scores]
        result = classification_metrics(labels, actions)
        if metric == "f1":
            value = result["f1"] if result["f1"] is not None else 0.0
        elif metric == "mcc":
            value = result["mcc"] if result["mcc"] is not None else 0.0
        else:
            recall = result["recall"]
            specificity = (
                1.0 - result["over_call_rate"]
                if result["over_call_rate"] is not None else 0.0
            )
            value = ((recall or 0.0) + specificity) / 2.0
        candidates.append({
            "threshold": threshold,
            "objective": value,
            "call_rate": result["call_rate"],
            "action_accuracy": result["action_accuracy"],
        })

    best = max(candidates, key=lambda row: (row["objective"], row["threshold"]))
    return {
        "policy_id": policy_id,
        "selected_threshold": best["threshold"],
        "selection_metric": metric,
        "selection_value": best["objective"],
        "n_eligible_validation_records": len(eligible),
        "tie_break": "higher threshold to reduce tool calls",
        "candidate_results": candidates,
        "warning": (
            "Calibrate on training/validation data only. Apply the frozen threshold to the "
            "locked test split; never select it using test metrics."
        ),
    }
