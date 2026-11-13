"""Transparent decision-policy simulator for N2A Lab v0.1.

The engine operates on supplied estimates and synthetic scenarios. It does not
call an LLM or infer hidden-state signals.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from n2a.scenarios import Scenario


POLICIES = {
    "no_tool": "P0 · No tool",
    "always_tool": "P1 · Always tool",
    "necessity_threshold": "P2 · Necessity threshold",
    "self_assessment": "P3 · Self-assessment",
    "n2a_utility": "P6 · N2A utility-aware",
}


@dataclass(frozen=True)
class ExperimentConfig:
    tool_reliability: float = 0.95
    tool_cost: float = 0.10
    risk_penalty: float = 0.05
    value_per_success: float = 1.0
    decision_threshold: float = 0.60
    self_assessment_bias: float = 0.10
    authorized: bool = True
    seed: int = 17

    def validate(self) -> None:
        for name in ("tool_reliability", "decision_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        for name in ("tool_cost", "risk_penalty", "value_per_success"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative.")


def _choose_action(
    policy_id: str,
    scenario: Scenario,
    config: ExperimentConfig,
    estimated_net_utility: float,
) -> tuple[bool, str, float | None]:
    p_no_tool = scenario.p_no_tool_success

    if policy_id == "no_tool":
        return False, "No-tool baseline.", None
    if policy_id == "always_tool":
        return True, "Calls whenever the candidate tool is available.", None
    if policy_id == "necessity_threshold":
        call = p_no_tool < config.decision_threshold
        return call, (
            f"Estimated no-tool success ({p_no_tool:.3f}) "
            f"is {'below' if call else 'at/above'} the threshold "
            f"({config.decision_threshold:.3f})."
        ), None
    if policy_id == "self_assessment":
        signal = min(1.0, max(0.0, 1.0 - p_no_tool + config.self_assessment_bias))
        call = signal >= config.decision_threshold
        return call, (
            f"Self-assessment signal={signal:.3f}. This is illustrative, "
            "not a calibrated probability."
        ), signal
    if policy_id == "n2a_utility":
        call = estimated_net_utility > 0.0
        return call, (
            f"Estimated net utility={estimated_net_utility:+.3f}; "
            "call only when estimated incremental value is positive."
        ), None
    raise ValueError(f"Unknown policy_id: {policy_id}")


def run_policy(
    policy_id: str,
    scenario: Scenario,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Run one policy on one scenario and return an auditable execution trace."""
    if policy_id not in POLICIES:
        raise ValueError(f"Unknown policy_id: {policy_id}")
    config.validate()

    p0 = scenario.p_no_tool_success
    p1 = scenario.p_with_tool_success * config.tool_reliability
    expected_gain = p1 - p0
    estimated_net_utility = (
        expected_gain * config.value_per_success
        - config.tool_cost
        - config.risk_penalty
    )

    call, decision_reason, self_signal = _choose_action(
        policy_id, scenario, config, estimated_net_utility
    )

    trace: list[dict[str, str]] = [
        {"stage": "N · Necessity estimate", "event": f"Estimated P(success | no tool)={p0:.3f}."},
        {
            "stage": "U · Expected utility",
            "event": (
                f"Expected gain={expected_gain:+.3f}; "
                f"estimated net utility={estimated_net_utility:+.3f}."
            ),
        },
    ]

    if call and not config.authorized:
        call = False
        decision_reason += " Authorization gate blocked the call."

    trace.append({
        "stage": "A · Action",
        "event": ("CALL" if call else "NO-CALL") + f" — {decision_reason}",
    })

    tool_succeeded: bool | None = None
    if call:
        # Same seed and scenario => matched tool outcome across compared policies.
        draw = random.Random(f"{config.seed}:{scenario.scenario_id}").random()
        tool_succeeded = draw < p1
        if tool_succeeded:
            answer = scenario.tool_fn()
            trace.append({
                "stage": "Tool execution",
                "event": f"{scenario.tool_name} returned: {answer}",
            })
        else:
            answer = "TOOL_FAILURE"
            trace.append({
                "stage": "Tool execution",
                "event": f"{scenario.tool_name} failed at configured reliability.",
            })
    else:
        answer = scenario.no_tool_answer
        trace.append({"stage": "Tool execution", "event": "No tool was invoked."})

    success = answer.strip().casefold() == scenario.reference_answer.strip().casefold()
    unnecessary_call = call and not scenario.expected_necessity
    missed_necessary_call = (not call) and scenario.expected_necessity
    realized_utility = (
        (config.value_per_success if success else 0.0)
        - ((config.tool_cost + config.risk_penalty) if call else 0.0)
    )

    trace.append({
        "stage": "Outcome",
        "event": f"{'SUCCESS' if success else 'FAILURE'}; answer={answer!r}; reference={scenario.reference_answer!r}.",
    })
    trace.append({
        "stage": "U · Realized utility",
        "event": f"{realized_utility:+.3f} under the configured reward/cost schedule.",
    })

    return {
        "policy_id": policy_id,
        "policy": POLICIES[policy_id],
        "scenario_id": scenario.scenario_id,
        "tool_called": call,
        "tool_succeeded": tool_succeeded,
        "answer": answer,
        "success": success,
        "expected_necessity": scenario.expected_necessity,
        "unnecessary_call": unnecessary_call,
        "missed_necessary_call": missed_necessary_call,
        "p_no_tool_success": p0,
        "p_with_tool_success_estimate": p1,
        "expected_gain": expected_gain,
        "estimated_net_utility": estimated_net_utility,
        "realized_utility": realized_utility,
        "self_assessment_signal": self_signal,
        "decision_reason": decision_reason,
        "trace": trace,
        "evidence_status": "SYNTHETIC_DEMO_NOT_BENCHMARK",
        "config": asdict(config),
    }
