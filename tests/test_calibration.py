import pytest

from n2a.calibration import calibrate_threshold


def test_p5_threshold_is_selected_on_validation_records():
    records = [
        {"necessity_label": 0, "probe_necessity_probability": 0.1},
        {"necessity_label": 0, "probe_necessity_probability": 0.2},
        {"necessity_label": 1, "probe_necessity_probability": 0.8},
        {"necessity_label": 1, "probe_necessity_probability": 0.9},
    ]
    result = calibrate_threshold(records, policy_id="P5", metric="f1")
    assert result["selected_threshold"] == 0.8
    assert result["n_eligible_validation_records"] == 4
    assert "locked test split" in result["warning"]


def test_p2_threshold_requires_repeated_probability_signal():
    records = [
        {"necessity_label": 1, "no_tool_success_probability": None},
        {"necessity_label": 0, "no_tool_success_probability": None},
    ]
    with pytest.raises(ValueError):
        calibrate_threshold(records, policy_id="P2")
