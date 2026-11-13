import pytest

from n2a.engine import ExperimentConfig, run_policy
from n2a.scenarios import get_scenario


def make_config(**overrides):
    values = {
        "tool_reliability": 1.0,
        "tool_cost": 0.05,
        "risk_penalty": 0.0,
        "value_per_success": 1.0,
        "decision_threshold": 0.60,
        "self_assessment_bias": 0.10,
        "authorized": True,
        "seed": 17,
    }
    values.update(overrides)
    return ExperimentConfig(**values)


def test_no_tool_policy_does_not_call_and_solves_arithmetic():
    result = run_policy("no_tool", get_scenario("arithmetic"), make_config())
    assert result["tool_called"] is False
    assert result["success"] is True


def test_always_tool_policy_calls_when_authorized():
    result = run_policy("always_tool", get_scenario("arithmetic"), make_config())
    assert result["tool_called"] is True
    assert result["success"] is True


def test_utility_policy_avoids_low_gain_expensive_call():
    result = run_policy(
        "n2a_utility",
        get_scenario("arithmetic"),
        make_config(tool_cost=0.50, risk_penalty=0.20),
    )
    assert result["tool_called"] is False


def test_utility_policy_calls_for_high_gain_task():
    result = run_policy("n2a_utility", get_scenario("private-records"), make_config())
    assert result["tool_called"] is True
    assert result["success"] is True


def test_authorization_gate_blocks_tool_call():
    result = run_policy(
        "always_tool",
        get_scenario("private-records"),
        make_config(authorized=False),
    )
    assert result["tool_called"] is False
    assert result["missed_necessary_call"] is True


def test_tool_failure_is_reproducible():
    cfg = make_config(tool_reliability=0.0)
    scenario = get_scenario("private-records")
    first = run_policy("always_tool", scenario, cfg)
    second = run_policy("always_tool", scenario, cfg)
    assert first["answer"] == second["answer"] == "TOOL_FAILURE"
    assert first["success"] is False


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        run_policy("n2a_utility", get_scenario("arithmetic"), make_config(tool_reliability=1.5))


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        run_policy("made_up_policy", get_scenario("arithmetic"), make_config())
