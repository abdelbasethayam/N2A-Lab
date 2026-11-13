"""Load official When2Tool linear-probe artifacts and export per-task necessity scores."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def predict_probe_probabilities(
    hidden_states: Any,
    probe_data: dict[str, Any],
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """Reproduce the official logistic-probe probability calculation."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    hidden = _array(hidden_states)
    layer = probe_data.get("layer", -1)
    if layer == "all":
        features = hidden.reshape(hidden.shape[0], -1)
    elif hidden.ndim == 3:
        layer_index = int(layer)
        if layer_index < 0:
            layer_index += hidden.shape[1]
        if not 0 <= layer_index < hidden.shape[1]:
            raise ValueError(f"Probe layer {layer} is out of range.")
        features = hidden[:, layer_index, :]
    elif hidden.ndim == 2:
        features = hidden
    else:
        raise ValueError(f"Unsupported hidden-state shape: {hidden.shape}")

    mean = _array(probe_data["scaler_mean"]).reshape(-1)
    scale = _array(probe_data["scaler_scale"]).reshape(-1)
    coef = _array(probe_data["coef"]).reshape(-1)
    if np.any(scale == 0):
        raise ValueError("Probe scaler contains a zero scale value.")
    if features.shape[1] != len(mean) or len(mean) != len(scale) or len(mean) != len(coef):
        raise ValueError(
            "Feature dimension mismatch between hidden states, scaler, and probe weights."
        )
    logits = ((features - mean) / scale) @ coef + float(probe_data["intercept"])
    scaled_logits = np.clip(logits / temperature, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-scaled_logits))


def load_probe_records(
    hidden_path: str | Path,
    probe_path: str | Path,
    labels_path: str | Path,
    *,
    temperature: float = 1.0,
) -> list[dict[str, Any]]:
    """Load upstream .pt/.json artifacts and return P5-ready task records."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install research dependencies to read .pt probe artifacts.") from exc

    hidden = torch.load(hidden_path, map_location="cpu", weights_only=True)
    probe = torch.load(probe_path, map_location="cpu", weights_only=True)
    with open(labels_path, "r", encoding="utf-8") as handle:
        label_payload = json.load(handle)

    metadata = label_payload.get("task_meta", [])
    if len(metadata) != len(hidden):
        raise ValueError(
            f"Hidden-state rows ({len(hidden)}) do not match label metadata ({len(metadata)})."
        )
    probabilities = predict_probe_probabilities(hidden, probe, temperature=temperature)

    records = []
    for meta, probability in zip(metadata, probabilities):
        records.append({
            "task_id": meta["id"],
            "necessity_label": int(meta["tool_necessary"]),
            "difficulty": meta.get("difficulty", "unknown"),
            "env_name": meta.get("env", "unknown"),
            "no_tool_correct": int(meta.get("no_tool_correct", 0)),
            "probe_necessity_probability": float(probability),
            "probe_temperature": float(temperature),
            "probe_layer": probe.get("layer", -1),
            "probe_label_source": "official_When2Tool_hard_no_tool_output",
        })
    return records


def export_probe_records(
    hidden_path: str | Path,
    probe_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
    *,
    temperature: float = 1.0,
) -> dict[str, Any]:
    records = load_probe_records(
        hidden_path, probe_path, labels_path, temperature=temperature
    )
    payload = {
        "signal_name": "P5_hidden_state_probe_necessity_probability",
        "n_records": len(records),
        "temperature": temperature,
        "source_artifacts": {
            "hidden_states": str(hidden_path),
            "probe": str(probe_path),
            "labels": str(labels_path),
        },
        "records": records,
        "warning": (
            "Probe scores are model- and protocol-specific. Evaluate calibration on held-out "
            "data; high AUROC alone is not evidence of awareness or causal control."
        ),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
