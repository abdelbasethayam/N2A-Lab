"""Run P0-P6 policies on a precomputed, leakage-controlled signal table."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from n2a.metrics import classification_metrics
from n2a.policies import DecisionContext, POLICY_NAMES, PolicyConfig, decide


CONTEXT_FIELDS = set(DecisionContext.__dataclass_fields__)


def evaluate_policy_suite(
    records: list[dict[str, Any]],
    *,
    config: PolicyConfig = PolicyConfig(),
    policies: tuple[str, ...] = tuple(POLICY_NAMES),
) -> dict[str, Any]:
    """Evaluate policies against supplied model-relative labels.

    Each record must contain necessity_label and the signal fields required by
    the selected policies. The caller is responsible for ensuring all learned
    signals and thresholds were produced without test-set leakage.
    """
    if not records:
        raise ValueError("At least one record is required.")
    labels = []
    contexts = []
    for index, record in enumerate(records):
        if "necessity_label" not in record:
            raise ValueError(f"Record {index} is missing necessity_label.")
        label = int(record["necessity_label"])
        if label not in (0, 1):
            raise ValueError(f"Record {index} has a non-binary necessity_label.")
        labels.append(label)
        payload = {key: value for key, value in record.items() if key in CONTEXT_FIELDS}
        if "task_id" not in payload:
            payload["task_id"] = record.get("task_id", record.get("id", index))
        contexts.append(DecisionContext(**payload))

    suite = {}
    for policy_id in policies:
        decisions = [decide(policy_id, context, config) for context in contexts]
        actions = [int(item["tool_called"]) for item in decisions]

        # Only actual probability estimates are passed to proper scoring rules.
        probability_values = None
        if policy_id == "P2":
            probability_values = [
                1.0 - float(context.no_tool_success_probability) for context in contexts
            ]
        elif policy_id == "P5":
            probability_values = [
                float(context.probe_necessity_probability) for context in contexts
            ]

        metrics = classification_metrics(labels, actions, probability_values)
        estimated_utilities = []
        for context, action in zip(contexts, actions):
            if context.expected_no_tool_success is None or context.expected_with_tool_success is None:
                continue
            p0 = float(context.expected_no_tool_success)
            p1 = float(context.expected_with_tool_success)
            expected_success = p1 if action else p0
            expected_cost = (context.tool_cost + context.risk_penalty) if action else 0.0
            estimated_utilities.append(expected_success * context.value_of_success - expected_cost)

        suite[policy_id] = {
            "policy_name": POLICY_NAMES[policy_id],
            "decision_metrics": metrics,
            "mean_estimated_utility": (
                sum(estimated_utilities) / len(estimated_utilities)
                if estimated_utilities else None
            ),
            "n_utility_scored": len(estimated_utilities),
            "decisions": decisions,
        }

    return {
        "n_records": len(records),
        "label_convention": "1=tool necessary; 0=tool unnecessary",
        "policy_config": asdict(config),
        "policies": suite,
        "evidence_status": "MODEL_DERIVED_SIGNALS_REQUIRED",
        "leakage_warning": (
            "P2 thresholds, P3/P5 signals, and any learned P4 surface score must be "
            "fixed using training/validation data or preregistration before test evaluation."
        ),
    }
