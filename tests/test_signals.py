from n2a.policy_evaluation import evaluate_policy_suite
from n2a.signals import build_policy_signal_records, parse_verbalized_necessity


def test_verbalized_necessity_parser_distinguishes_cannot_from_can():
    assert parse_verbalized_necessity("I can solve this directly without using a tool.")["score"] == 0.0
    assert parse_verbalized_necessity("I cannot solve this without a tool.")["score"] == 1.0
    assert parse_verbalized_necessity("Let me think about the problem.")["score"] is None


def test_signal_builder_joins_matched_model_evidence():
    no_tool = [
        {"id": 1, "expected": {"answer": "42"}, "final_response": r"\boxed{42}"},
        {"id": 2, "expected": {"answer": "42"}, "final_response": r"\boxed{wrong}"},
    ]
    force_tool = [
        {"id": 1, "expected": {"answer": "42"}, "final_response": r"\boxed{42}"},
        {"id": 2, "expected": {"answer": "42"}, "final_response": r"\boxed{42}"},
    ]
    reasoning = [
        {"id": 1, "trace": [{"model_raw_output": "I can solve this directly without using a tool."}]},
        {"id": 2, "trace": [{"model_raw_output": "I need to use a tool for this question."}]},
    ]
    probes = {
        "records": [
            {"task_id": 1, "probe_necessity_probability": 0.1, "difficulty": "easy", "env_name": "CalculatorEnv"},
            {"task_id": 2, "probe_necessity_probability": 0.9, "difficulty": "hard", "env_name": "RetrieverEnv"},
        ]
    }
    result = build_policy_signal_records(
        no_tool,
        force_tool_payload=force_tool,
        reasoning_payload=reasoning,
        probe_payload=probes,
        tool_cost=0.1,
        risk_penalty=0.02,
    )
    assert result["n_records"] == 2
    assert result["records"][0]["verbalized_necessity_score"] == 0.0
    assert result["records"][1]["verbalized_necessity_score"] == 1.0
    assert result["records"][1]["expected_no_tool_success"] == 0.0
    assert result["records"][1]["expected_with_tool_success"] == 1.0
    suite = evaluate_policy_suite(result["records"])
    assert set(suite["policies"]) == {"P0", "P1", "P2", "P3", "P4", "P5", "P6"}
    assert all(item["n_evaluated"] == 2 for item in suite["policies"].values())
