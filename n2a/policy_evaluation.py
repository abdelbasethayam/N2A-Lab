"""Run P0-P6 policies on a precomputed, leakage-controlled signal table."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from n2a.metrics import classification_metrics
from n2a.policies import DecisionContext, POLICY_NAMES, PolicyConfig, decide


CONTEXT_FIELDS = set(DecisionContext.__dataclass_fields__)


def _missing_signal_fields(policy_id: str, context: DecisionContext) -> list[str]:
    if policy_id in ("P0", "P1"):
        return []
    if policy_id == "P2":
        return [] if context.no_tool_success_probability is not None else ["no_tool_success_probability"]
    if policy_id == "P3":
        return [] if context.verbalized_necessity_score is not None else ["verbalized_necessity_score"]
    if policy_id == "P4":
        if context.surface_necessity_score is not None:
            return []
        return [] if context.difficulty in ("easy", "medium", "hard") else ["difficulty_or_surface_necessity_score"]
    if policy_id == "P5":
        return [] if context.probe_necessity_probability is not None else ["probe_necessity_probability"]
    if policy_id == "P6":
        missing = []
        if context.expected_no_tool_success is None:
            missing.append("expected_no_tool_success")
        if context.expected_with_tool_success is None:
            missing.append("expected_with_tool_success")
        return missing
    raise ValueError(f"Unknown policy: {policy_id}")


def evaluate_policy_suite(
    records: list[dict[str, Any]],
    *,
    config: PolicyConfig = PolicyConfig(),
    policies: tuple[str, ...] = tuple(POLICY_NAMES),
) -> dict[str, Any]:
    """Evaluate each policy only on records with its required input signals.

    Each record must contain necessity_label. Missing policy-specific signals
    are reported and excluded for that policy, rather than silently imputed.
    """
    if not records:
        raise ValueError("At least one record is required.")
    if not policies:
        raise ValueError("At least one policy must be selected.")
    unknown = set(policies).difference(POLICY_NAMES)
    if unknown:
        raise ValueError(f"Unknown policies: {sorted(unknown)}")

    prepared = []
    for index, record in enumerate(records):
        if "necessity_label" not in record:
            raise ValueError(f"Record {index} is missing necessity_label.")
        label = int(record["necessity_label"])
        if label not in (0, 1):
            raise ValueError(f"Record {index} has a non-binary necessity_label.")
        payload = {key: value for key, value in record.items() if key in CONTEXT_FIELDS}
        if "task_id" not in payload:
            payload["task_id"] = record.get("task_id", record.get("id", index))
        prepared.append((label, DecisionContext(**payload)))

    suite = {}
    for policy_id in policies:
        available = []
        excluded = []
        for label, context in prepared:
            missing = _missing_signal_fields(policy_id, context)
            if missing:
                excluded.append({"task_id": context.task_id, "missing": missing})
            else:
                available.append((label, context))

        if not available:
            suite[policy_id] = {
                "policy_name": POLICY_NAMES[policy_id],
                "status": "NO_ELIGIBLE_RECORDS",
                "n_evaluated": 0,
                "n_excluded_missing_signal": len(excluded),
                "excluded_records": excluded,
                "decision_metrics": None,
                "mean_estimated_utility": None,
                "n_utility_scored": 0,
                "decisions": [],
            }
            continue

        labels = [label for label, _ in available]
        contexts = [context for _, context in available]
        decisions = [decide(policy_id, context, config) for context in contexts]
        actions = [int(item["tool_called"]) for item in decisions]

        # Only probability estimates with a probabilistic interpretation are scored.
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
            "n_evaluated": len(available),
            "n_excluded_missing_signal": len(excluded),
            "excluded_records": excluded,
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
            "P2/P3/P5 signals, P4 surface scores, and P6 outcome estimates must be "
            "fixed using training/validation data or preregistration before test evaluation. "
            "Policies are compared on their eligible records; inspect n_evaluated per policy."
        ),
    }
