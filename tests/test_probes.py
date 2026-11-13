import numpy as np
import pytest

from n2a.probes import predict_probe_probabilities


def test_probe_probability_matches_logistic_regression_formula():
    hidden = np.array([
        [[0.0, 1.0], [1.0, 2.0]],
        [[1.0, 0.0], [2.0, 1.0]],
    ])
    probe = {
        "layer": -1,
        "scaler_mean": np.array([0.0, 0.0]),
        "scaler_scale": np.array([1.0, 1.0]),
        "coef": np.array([1.0, -1.0]),
        "intercept": 0.0,
    }
    probabilities = predict_probe_probabilities(hidden, probe)
    assert probabilities.shape == (2,)
    assert probabilities[0] == pytest.approx(1.0 / (1.0 + np.exp(1.0)))
    assert probabilities[1] == pytest.approx(1.0 / (1.0 + np.exp(-1.0)))


def test_all_layer_probe_flattens_layer_features():
    hidden = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    probe = {
        "layer": "all",
        "scaler_mean": np.zeros(4),
        "scaler_scale": np.ones(4),
        "coef": np.ones(4),
        "intercept": 0.0,
    }
    probability = predict_probe_probabilities(hidden, probe)[0]
    assert probability == pytest.approx(1.0 / (1.0 + np.exp(-10.0)))


def test_probe_rejects_feature_dimension_mismatch():
    hidden = np.zeros((1, 2, 3))
    probe = {
        "layer": -1,
        "scaler_mean": np.zeros(2),
        "scaler_scale": np.ones(2),
        "coef": np.ones(2),
        "intercept": 0.0,
    }
    with pytest.raises(ValueError):
        predict_probe_probabilities(hidden, probe)
