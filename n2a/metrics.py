"""Metrics for tool-necessity decisions and tool-use outcomes."""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def _safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def classification_metrics(
    necessity_labels: Sequence[int],
    call_decisions: Sequence[int | bool],
    necessity_probabilities: Sequence[float] | None = None,
    *,
    n_bins: int = 10,
) -> dict:
    """Evaluate whether call/no-call decisions match model-relative necessity labels.

    Label convention: 1 = tool necessary; 0 = tool unnecessary.
    Decision convention: 1 = call; 0 = do not call.
    """
    if len(necessity_labels) != len(call_decisions):
        raise ValueError("necessity_labels and call_decisions must have equal length")
    if not necessity_labels:
        raise ValueError("At least one example is required")
    y = [int(v) for v in necessity_labels]
    a = [int(bool(v)) for v in call_decisions]
    if any(v not in (0, 1) for v in y):
        raise ValueError("Necessity labels must be binary (0/1)")
    if necessity_probabilities is not None:
        if len(necessity_probabilities) != len(y):
            raise ValueError("Probability and label lengths must match")
        if any(not 0.0 <= float(p) <= 1.0 for p in necessity_probabilities):
            raise ValueError("Necessity probabilities must be in [0, 1]")

    tp = sum(1 for yi, ai in zip(y, a) if yi == 1 and ai == 1)
    fp = sum(1 for yi, ai in zip(y, a) if yi == 0 and ai == 1)
    fn = sum(1 for yi, ai in zip(y, a) if yi == 1 and ai == 0)
    tn = sum(1 for yi, ai in zip(y, a) if yi == 0 and ai == 0)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = (
        _safe_div(2 * precision * recall, precision + recall)
        if precision is not None and recall is not None else None
    )
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe_div(tp * tn - fp * fn, denominator)

    result = {
        "n": len(y),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "action_accuracy": (tp + tn) / len(y),
        "call_rate": sum(a) / len(a),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "over_call_rate": _safe_div(fp, fp + tn),
        "missed_necessity_rate": _safe_div(fn, fn + tp),
        "unnecessary_calls": fp,
        "missed_necessary_calls": fn,
    }

    if necessity_probabilities is not None:
        probs = [float(p) for p in necessity_probabilities]
        result["brier_score"] = sum((p - yi) ** 2 for p, yi in zip(probs, y)) / len(y)
        bins = []
        ece = 0.0
        for index in range(n_bins):
            lo, hi = index / n_bins, (index + 1) / n_bins
            members = [
                (p, yi) for p, yi in zip(probs, y)
                if lo <= p < hi or (index == n_bins - 1 and p == 1.0)
            ]
            if not members:
                continue
            confidence = sum(p for p, _ in members) / len(members)
            observed = sum(yi for _, yi in members) / len(members)
            weight = len(members) / len(y)
            ece += weight * abs(confidence - observed)
            bins.append({
                "lower": lo, "upper": hi, "n": len(members),
                "mean_probability": confidence, "observed_frequency": observed,
            })
        result["ece"] = ece
        result["calibration_bins"] = bins
    return result


def outcome_utility(
    task_success: Sequence[int | bool],
    tool_calls: Sequence[int],
    *,
    success_value: float = 1.0,
    cost_per_call: float = 0.0,
    risk_per_call: float = 0.0,
    latency_costs: Sequence[float] | None = None,
) -> dict:
    """Compute realized utility with explicit, externally specified cost units."""
    if len(task_success) != len(tool_calls):
        raise ValueError("task_success and tool_calls must have equal length")
    if not task_success:
        raise ValueError("At least one example is required")
    if any(int(c) < 0 for c in tool_calls):
        raise ValueError("tool_calls cannot be negative")
    if min(success_value, cost_per_call, risk_per_call) < 0:
        raise ValueError("Utility values and costs must be non-negative")
    if latency_costs is not None and len(latency_costs) != len(task_success):
        raise ValueError("latency_costs length must match task_success")

    successes = sum(bool(v) for v in task_success)
    calls = sum(int(v) for v in tool_calls)
    total_cost = calls * (cost_per_call + risk_per_call)
    latency_cost = sum(latency_costs or [])
    reward = successes * success_value
    net = reward - total_cost - latency_cost
    return {
        "n_tasks": len(task_success),
        "successes": successes,
        "task_success_rate": successes / len(task_success),
        "total_tool_calls": calls,
        "mean_tool_calls_per_task": calls / len(task_success),
        "gross_success_reward": reward,
        "tool_cost": calls * cost_per_call,
        "risk_cost": calls * risk_per_call,
        "latency_cost": latency_cost,
        "net_utility": net,
        "cost_per_success": (
            (total_cost + latency_cost) / successes if successes else None
        ),
        "utility_units": "user-configured; not directly comparable across studies unless normalized",
    }
