"""Explicit P0-P6 decision policies for N2A experiments.

These policies consume estimates supplied by an experiment adapter. They do not
pretend that the public When2Tool task files contain model-specific labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


POLICY_NAMES = {
    "P0": "Never call",
    "P1": "Always call",
    "P2": "No-tool capability threshold",
    "P3": "Verbalized necessity",
    "P4": "Surface/difficulty heuristic",
    "P5": "Internal-state probe",
    "P6": "N2A uncertainty- and utility-aware",
}


@dataclass(frozen=True)
class DecisionContext:
    task_id: str | int
    difficulty: str = "unknown"
    env_name: str = "unknown"
    no_tool_success_probability: Optional[float] = None
    verbalized_necessity_score: Optional[float] = None
    surface_necessity_score: Optional[float] = None
    probe_necessity_probability: Optional[float] = None
    tool_success_probability: Optional[float] = None
    expected_no_tool_success: Optional[float] = None
    expected_with_tool_success: Optional[float] = None
    tool_cost: float = 0.0
    risk_penalty: float = 0.0
    uncertainty: float = 0.0
    value_of_success: float = 1.0


@dataclass(frozen=True)
class PolicyConfig:
    necessity_threshold: float = 0.5
    verbalized_threshold: float = 0.5
    surface_threshold: float = 0.5
    probe_threshold: float = 0.5
    uncertainty_aversion: float = 0.0


def _probability(value: Optional[float], field: str) -> float:
    if value is None:
        raise ValueError(f"{field} is required by this policy")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return value


def decide(policy_id: str, context: DecisionContext, config: PolicyConfig = PolicyConfig()) -> dict:
    """Return a transparent CALL/NO_CALL decision and its rationale."""
    if policy_id not in POLICY_NAMES:
        raise ValueError(f"Unknown policy: {policy_id}")
    call = False
    score = None
    utility = None

    if policy_id == "P0":
        reason = "Fixed no-tool baseline."
    elif policy_id == "P1":
        call, reason = True, "Fixed always-call baseline."
    elif policy_id == "P2":
        p0 = _probability(context.no_tool_success_probability, "no_tool_success_probability")
        call = p0 < config.necessity_threshold
        reason = f"P(no-tool success)={p0:.3f}; threshold={config.necessity_threshold:.3f}."
        score = 1.0 - p0
    elif policy_id == "P3":
        score = _probability(
            context.verbalized_necessity_score, "verbalized_necessity_score"
        )
        call = score >= config.verbalized_threshold
        reason = f"Parsed verbalized-necessity score={score:.3f}; threshold={config.verbalized_threshold:.3f}. This is a categorical self-report score, not a calibrated probability."
    elif policy_id == "P4":
        if context.surface_necessity_score is not None:
            score = _probability(context.surface_necessity_score, "surface_necessity_score")
            source = "provided surface score"
        else:
            # Transparent baseline rule; not learned and not a substitute for validation.
            difficulty_scores = {"easy": 0.20, "medium": 0.50, "hard": 0.80}
            if context.difficulty not in difficulty_scores:
                raise ValueError("P4 requires a surface score or easy/medium/hard difficulty.")
            score = difficulty_scores[context.difficulty]
            source = f"predeclared difficulty mapping ({context.difficulty})"
        call = score >= config.surface_threshold
        reason = f"{source}: score={score:.3f}; threshold={config.surface_threshold:.3f}."
    elif policy_id == "P5":
        score = _probability(
            context.probe_necessity_probability, "probe_necessity_probability"
        )
        call = score >= config.probe_threshold
        reason = f"Hidden-state probe P(necessary)={score:.3f}; threshold={config.probe_threshold:.3f}."
    else:  # P6
        p0 = _probability(context.expected_no_tool_success, "expected_no_tool_success")
        p1 = _probability(context.expected_with_tool_success, "expected_with_tool_success")
        if context.tool_cost < 0 or context.risk_penalty < 0 or context.value_of_success < 0:
            raise ValueError("Costs, risk, and value must be non-negative.")
        if context.uncertainty < 0:
            raise ValueError("uncertainty must be non-negative.")
        expected_gain = p1 - p0
        uncertainty_cost = config.uncertainty_aversion * context.uncertainty
        utility = (
            expected_gain * context.value_of_success
            - context.tool_cost
            - context.risk_penalty
            - uncertainty_cost
        )
        call = utility > 0.0
        reason = (
            f"Expected marginal utility={utility:+.4f} "
            f"(gain={expected_gain:+.4f}, tool_cost={context.tool_cost:.4f}, "
            f"risk={context.risk_penalty:.4f}, uncertainty_cost={uncertainty_cost:.4f})."
        )
        score = expected_gain

    return {
        "policy_id": policy_id,
        "policy_name": POLICY_NAMES[policy_id],
        "task_id": context.task_id,
        "action": "CALL" if call else "NO_CALL",
        "tool_called": call,
        "decision_score": score,
        "expected_net_utility": utility,
        "reason": reason,
        "evidence_status": "MODEL_OR_EXPERIMENT_INPUT_REQUIRED",
    }
