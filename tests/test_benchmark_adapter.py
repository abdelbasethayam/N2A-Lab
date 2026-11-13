import json

import pytest

from n2a.benchmarks.when2tool import normalize_when2tool_row
from n2a.necessity import derive_necessity_labels
from n2a.evaluation import evaluate_outputs


def row():
    return {
        "id": 11001,
        "difficulty": "easy",
        "multi_step": False,
        "instruction": "Answer with only the number: 20+20",
        "env_name": "CalculatorEnv",
        "tools": json.dumps(["evaluate_expression", "get_last_result"]),
        "parameters": "{}",
        "answer": "40",
        "steps": "[]",
        "tags": '["calculator", "single_round"]',
    }


def test_hf_row_normalizes_to_official_task_schema():
    task = normalize_when2tool_row(row())
    assert task["id"] == 11001
    assert task["environments"][0]["name"] == "CalculatorEnv"
    assert task["environments"][0]["tools"] == ["evaluate_expression", "get_last_result"]
    assert task["expected"]["answer"] == "40"
    assert "tool_necessary" not in task


def test_bad_json_field_is_rejected():
    item = row()
    item["tools"] = "[not-json]"
    with pytest.raises(ValueError):
        normalize_when2tool_row(item)


def test_single_run_failure_label_is_model_relative():
    payload = [
        {"id": 1, "expected": {"answer": "42"}, "final_response": r"\boxed{42}"},
        {"id": 2, "expected": {"answer": "42"}, "final_response": r"\boxed{41}"},
    ]
    result = derive_necessity_labels(payload, label_rule="single_failure")
    assert result["n_tasks"] == 2
    assert result["n_necessary"] == 1
    assert [x["tool_necessary"] for x in result["labels"]] == [0, 1]


def test_repeated_run_rule_is_explicit():
    payload = {
        "run_0": [{"id": 1, "expected": {"answer": "x"}, "final_response": r"\boxed{x}"}],
        "run_1": [{"id": 1, "expected": {"answer": "x"}, "final_response": r"\boxed{wrong}"}],
    }
    result = derive_necessity_labels(payload, label_rule="any_failure")
    assert result["labels"][0]["no_tool_success_probability"] == 0.5
    assert result["labels"][0]["tool_necessary"] == 1
    with pytest.raises(ValueError):
        derive_necessity_labels(payload, label_rule="single_failure")


def test_behavior_evaluation_reports_over_and_under_calling():
    no_tool = [
        {"id": 1, "expected": {"answer": "42"}, "final_response": r"\boxed{42}"},
        {"id": 2, "expected": {"answer": "42"}, "final_response": r"\boxed{wrong}"},
    ]
    labels = derive_necessity_labels(no_tool)
    outputs = [
        {"id": 1, "expected": {"answer": "42"}, "final_response": r"\boxed{42}", "tool_calls": 1},
        {"id": 2, "expected": {"answer": "42"}, "final_response": r"\boxed{42}", "tool_calls": 0},
    ]
    result = evaluate_outputs(outputs, labels, cost_per_call=0.1)
    metrics = result["decision_metrics"]
    assert metrics["unnecessary_calls"] == 1
    assert metrics["missed_necessary_calls"] == 1
    assert result["outcome_utility"]["task_success_rate"] == 1.0
