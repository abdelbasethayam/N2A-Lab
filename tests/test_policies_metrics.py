import pytest

from n2a.metrics import classification_metrics, outcome_utility
from n2a.policies import DecisionContext, PolicyConfig, decide


def test_p0_and_p1_are_fixed_baselines():
    context = DecisionContext(task_id="t1")
    assert decide("P0", context)["action"] == "NO_CALL"
    assert decide("P1", context)["action"] == "CALL"


def test_p2_uses_no_tool_success_probability():
    context = DecisionContext(task_id="t1", no_tool_success_probability=0.2)
    result = decide("P2", context, PolicyConfig(necessity_threshold=0.5))
    assert result["action"] == "CALL"


def test_p3_requires_explicit_verbalized_signal():
    context = DecisionContext(task_id="t1", verbalized_necessity_probability=0.8)
    assert decide("P3", context)["action"] == "CALL"
    with pytest.raises(ValueError):
        decide("P3", DecisionContext(task_id="t2"))


def test_p4_is_a_transparent_difficulty_baseline():
    easy = DecisionContext(task_id="e", difficulty="easy")
    hard = DecisionContext(task_id="h", difficulty="hard")
    assert decide("P4", easy)["action"] == "NO_CALL"
    assert decide("P4", hard)["action"] == "CALL"


def test_p5_uses_probe_probability():
    context = DecisionContext(task_id="t1", probe_necessity_probability=0.91)
    assert decide("P5", context)["action"] == "CALL"


def test_p6_subtracts_cost_risk_and_uncertainty():
    context = DecisionContext(
        task_id="t1",
        expected_no_tool_success=0.3,
        expected_with_tool_success=0.95,
        tool_cost=0.1,
        risk_penalty=0.05,
        uncertainty=0.2,
        value_of_success=1.0,
    )
    result = decide("P6", context, PolicyConfig(uncertainty_aversion=0.5))
    assert result["action"] == "CALL"
    assert result["expected_net_utility"] == pytest.approx(0.4)


def test_classification_metrics_and_calibration():
    result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 0], [0.1, 0.8, 0.9, 0.2])
    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert result["over_call_rate"] == 0.5
    assert result["missed_necessity_rate"] == 0.5
    assert "brier_score" in result and "ece" in result


def test_utility_uses_explicit_cost_units():
    result = outcome_utility([1, 0], [2, 1], success_value=1.0, cost_per_call=0.1, risk_per_call=0.05)
    assert result["total_tool_calls"] == 3
    assert result["net_utility"] == pytest.approx(0.55)


def test_policy_suite_compares_all_policies_on_precomputed_signals():
    from n2a.policy_evaluation import evaluate_policy_suite

    records = [
        {
            "task_id": "a",
            "necessity_label": 0,
            "difficulty": "easy",
            "no_tool_success_probability": 0.95,
            "verbalized_necessity_probability": 0.1,
            "probe_necessity_probability": 0.05,
            "expected_no_tool_success": 0.95,
            "expected_with_tool_success": 0.98,
            "tool_cost": 0.1,
            "risk_penalty": 0.02,
            "uncertainty": 0.1,
        },
        {
            "task_id": "b",
            "necessity_label": 1,
            "difficulty": "hard",
            "no_tool_success_probability": 0.10,
            "verbalized_necessity_probability": 0.9,
            "probe_necessity_probability": 0.95,
            "expected_no_tool_success": 0.10,
            "expected_with_tool_success": 0.95,
            "tool_cost": 0.1,
            "risk_penalty": 0.02,
            "uncertainty": 0.1,
        },
    ]
    result = evaluate_policy_suite(records)
    assert set(result["policies"]) == {"P0", "P1", "P2", "P3", "P4", "P5", "P6"}
    assert result["policies"]["P0"]["decision_metrics"]["call_rate"] == 0.0
    assert result["policies"]["P1"]["decision_metrics"]["call_rate"] == 1.0
    assert result["policies"]["P5"]["decision_metrics"]["brier_score"] is not None
