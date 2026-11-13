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


def train_probe_with_repeated_labels(
    train_hidden_path: str | Path,
    train_metadata_path: str | Path,
    train_no_tool_payload: Any,
    test_hidden_path: str | Path,
    test_metadata_path: str | Path,
    test_no_tool_payload: Any,
    output_dir: str | Path,
    *,
    label_rule: str = "any_failure",
    reg: float = 10000.0,
    layer: str | int = "all",
) -> dict[str, Any]:
    """Train a fresh probe on repeated-run labels aligned to official hidden states.

    This is an N2A extension, not the original one-run When2Tool probe protocol.
    """
    if reg <= 0:
        raise ValueError("reg must be positive")
    try:
        import torch
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, roc_auc_score
    except ImportError as exc:
        raise ImportError("Install requirements-research.txt to train probes.") from exc

    from n2a.necessity import derive_necessity_labels
    from n2a.metrics import classification_metrics

    train_label_payload = derive_necessity_labels(
        train_no_tool_payload, label_rule=label_rule
    )
    test_label_payload = derive_necessity_labels(
        test_no_tool_payload, label_rule=label_rule
    )
    train_labels = {row["task_id"]: row["tool_necessary"] for row in train_label_payload["labels"]}
    test_labels = {row["task_id"]: row["tool_necessary"] for row in test_label_payload["labels"]}

    if set(train_labels).intersection(test_labels):
        raise ValueError("Train/test task IDs overlap; refusing to train with leakage.")

    def load_aligned(hidden_path, metadata_path, label_map, split_name):
        hidden = torch.load(hidden_path, map_location="cpu", weights_only=True)
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle).get("task_meta", [])
        if len(hidden) != len(metadata):
            raise ValueError(f"{split_name}: hidden-state rows and metadata rows differ.")
        indices = []
        ids = []
        meta = []
        for index, item in enumerate(metadata):
            task_id = item["id"]
            if task_id in label_map:
                indices.append(index)
                ids.append(task_id)
                meta.append(item)
        missing = set(label_map).difference(ids)
        if missing:
            raise ValueError(f"{split_name}: no hidden states for task IDs {sorted(missing)[:10]}")
        if not indices:
            raise ValueError(f"{split_name}: no aligned examples.")
        return hidden[indices], [int(label_map[task_id]) for task_id in ids], ids, meta

    H_train, y_train, train_ids, train_meta = load_aligned(
        train_hidden_path, train_metadata_path, train_labels, "train"
    )
    H_test, y_test, test_ids, test_meta = load_aligned(
        test_hidden_path, test_metadata_path, test_labels, "test"
    )
    if len(set(y_train)) < 2:
        raise ValueError("Repeated-run training labels contain only one class.")

    if layer == "all":
        X_train = H_train.reshape(H_train.shape[0], -1).numpy()
        X_test = H_test.reshape(H_test.shape[0], -1).numpy()
        saved_layer = "all"
    else:
        layer_index = int(layer)
        if layer_index < 0:
            layer_index += H_train.shape[1]
        if not 0 <= layer_index < H_train.shape[1]:
            raise ValueError("Requested probe layer is out of range.")
        X_train = H_train[:, layer_index, :].numpy()
        X_test = H_test[:, layer_index, :].numpy()
        saved_layer = layer_index

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    classifier = LogisticRegression(
        C=1.0 / reg, solver="lbfgs", max_iter=2000, random_state=42
    )
    classifier.fit(X_train_scaled, y_train)
    probabilities = classifier.predict_proba(X_test_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = classification_metrics(y_test, predictions.tolist(), probabilities.tolist())
    metrics["train_accuracy"] = float(accuracy_score(y_train, classifier.predict(X_train_scaled)))
    metrics["test_accuracy"] = float(accuracy_score(y_test, predictions))
    metrics["test_auroc"] = (
        float(roc_auc_score(y_test, probabilities)) if len(set(y_test)) > 1 else None
    )

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    probe_path = target / "probe_repeated_labels.pt"
    torch.save({
        "coef": torch.from_numpy(classifier.coef_[0]),
        "intercept": float(classifier.intercept_[0]),
        "scaler_mean": torch.from_numpy(scaler.mean_),
        "scaler_scale": torch.from_numpy(scaler.scale_),
        "C": 1.0 / reg,
        "reg": reg,
        "mode": "no_reasoning",
        "layer": saved_layer,
        "label_rule": label_rule,
        "label_source": "repeated_hard_no_tool_trials",
    }, probe_path)

    signal_records = []
    for task_id, meta, label, probability in zip(test_ids, test_meta, y_test, probabilities):
        signal_records.append({
            "task_id": task_id,
            "necessity_label": int(label),
            "difficulty": meta.get("difficulty", "unknown"),
            "env_name": meta.get("env", "unknown"),
            "probe_necessity_probability": float(probability),
            "probe_label_rule": label_rule,
            "probe_layer": saved_layer,
        })
    signal_payload = {
        "signal_name": "P5_repeated_label_hidden_state_probe",
        "n_records": len(signal_records),
        "label_rule": label_rule,
        "probe_path": str(probe_path),
        "records": signal_records,
        "warning": (
            "This probe is trained on repeated-run necessity labels and is not the original "
            "one-run When2Tool probe. Evaluate only on the disjoint test task IDs."
        ),
    }
    signal_path = target / "test_repeated_probe_signals.json"
    signal_path.write_text(
        json.dumps(signal_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = {
        "experiment": "N2A repeated-label linear probe",
        "label_rule": label_rule,
        "reg": reg,
        "layer": saved_layer,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "n_train_necessary": sum(y_train),
        "n_test_necessary": sum(y_test),
        "metrics": metrics,
        "probe_path": str(probe_path),
        "test_signals_path": str(signal_path),
        "evidence_status": "MODEL_RUNS_REQUIRED",
    }
    (target / "repeated_probe_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
