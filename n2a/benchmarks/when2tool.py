"""Reproducible loader and integrity audit for the official When2Tool dataset.

Important: the public task dataset does not contain model-specific necessity labels.
Those labels must be collected by running the specified model in hard-no-tool mode.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATASET_ID = "cesun/When2Tool"
DATASET_REVISION = "153066c0285887915fc6b530c9af2987a2c859ac"
UPSTREAM_REPO = "https://github.com/Trustworthy-ML-Lab/when2tool"
UPSTREAM_COMMIT = "ef0cf1ea72eb8acdfd3fb1251c4b6a9a0131a23e"

EXPECTED_COUNTS = {
    "single_hop": {"train": 900, "test": 2250},
    "multi_hop": {"train": 180, "test": 450},
}
CONFIG_TO_FILENAME = {
    ("single_hop", "train"): "tasks_v1_train.json",
    ("single_hop", "test"): "tasks_v1_test.json",
    ("multi_hop", "train"): "tasks_v1_multihop_train.json",
    ("multi_hop", "test"): "tasks_v1_multihop_test.json",
}
REQUIRED_COLUMNS = {
    "id", "difficulty", "multi_step", "instruction", "env_name",
    "tools", "parameters", "answer", "steps", "tags",
}


def _parse_json_field(value: Any, *, field: str, task_id: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None or value == "":
        return {} if field == "parameters" else []
    if not isinstance(value, str):
        raise TypeError(f"Task {task_id}: {field} must be JSON text, dict, or list.")
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Task {task_id}: invalid JSON in {field}: {exc}") from exc


def normalize_when2tool_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a Hugging Face row to the task schema used by the official code."""
    missing = REQUIRED_COLUMNS.difference(row)
    if missing:
        raise ValueError(f"Dataset row is missing required columns: {sorted(missing)}")

    task_id = row["id"]
    tools = _parse_json_field(row["tools"], field="tools", task_id=task_id)
    parameters = _parse_json_field(row["parameters"], field="parameters", task_id=task_id)
    steps = _parse_json_field(row["steps"], field="steps", task_id=task_id)
    tags = _parse_json_field(row["tags"], field="tags", task_id=task_id)

    if not isinstance(tools, list):
        raise TypeError(f"Task {task_id}: tools must decode to a list.")
    if not isinstance(parameters, dict):
        raise TypeError(f"Task {task_id}: parameters must decode to an object.")
    if not isinstance(steps, list):
        raise TypeError(f"Task {task_id}: steps must decode to a list.")
    if not isinstance(tags, list):
        raise TypeError(f"Task {task_id}: tags must decode to a list.")

    expected = {"answer": str(row["answer"])}
    if steps:
        expected["steps"] = steps

    task = {
        "id": int(task_id),
        "difficulty": str(row["difficulty"]),
        "multi_step": bool(row["multi_step"]),
        "instruction": str(row["instruction"]),
        "environments": [{
            "name": str(row["env_name"]),
            "tools": tools,
            "parameters": parameters,
        }],
        "expected": expected,
        "tags": tags,
    }
    # Deliberately do not add tool_necessary: it is model/protocol-dependent.
    return task


def load_when2tool_split(
    config: str,
    split: str,
    *,
    revision: str = DATASET_REVISION,
) -> list[dict[str, Any]]:
    """Load one immutable official HF split and normalize it for the upstream runner."""
    if config not in EXPECTED_COUNTS:
        raise ValueError(f"Unsupported When2Tool config: {config}")
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install the benchmark extra with: pip install -r requirements-research.txt"
        ) from exc

    dataset = load_dataset(DATASET_ID, config, split=split, revision=revision)
    tasks = [normalize_when2tool_row(dict(row)) for row in dataset]
    return tasks


def _audit_split(tasks: list[dict[str, Any]], config: str, split: str) -> dict[str, Any]:
    ids = [task["id"] for task in tasks]
    duplicate_ids = sorted(task_id for task_id, count in Counter(ids).items() if count > 1)
    env_counts = Counter(task["environments"][0]["name"] for task in tasks)
    difficulty_counts = Counter(task["difficulty"] for task in tasks)
    expected_count = EXPECTED_COUNTS[config][split]
    return {
        "config": config,
        "split": split,
        "n_tasks": len(tasks),
        "expected_n_tasks": expected_count,
        "count_matches_release_card": len(tasks) == expected_count,
        "unique_ids": len(set(ids)) == len(ids),
        "duplicate_ids": duplicate_ids,
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "environment_counts": dict(sorted(env_counts.items())),
        "n_environments": len(env_counts),
        "multi_step_count": sum(bool(task["multi_step"]) for task in tasks),
        "contains_model_necessity_labels": any(
            "tool_necessary" in task or "necessity_label" in task for task in tasks
        ),
    }


def audit_when2tool(*, revision: str = DATASET_REVISION) -> dict[str, Any]:
    """Load and audit all four official splits; fail on duplicates or split leakage."""
    reports: dict[str, Any] = {}
    id_sets: dict[tuple[str, str], set[int]] = {}
    for config in EXPECTED_COUNTS:
        for split in ("train", "test"):
            tasks = load_when2tool_split(config, split, revision=revision)
            reports[f"{config}/{split}"] = _audit_split(tasks, config, split)
            id_sets[(config, split)] = {task["id"] for task in tasks}
            if reports[f"{config}/{split}"]["duplicate_ids"]:
                raise ValueError(f"Duplicate task IDs found in {config}/{split}.")
            if not reports[f"{config}/{split}"]["count_matches_release_card"]:
                raise ValueError(
                    f"Unexpected row count for {config}/{split}: "
                    f"{len(tasks)} != {EXPECTED_COUNTS[config][split]}"
                )
        overlap = id_sets[(config, "train")].intersection(id_sets[(config, "test")])
        if overlap:
            raise ValueError(f"Train/test ID leakage in {config}: {sorted(overlap)[:10]}")

    return {
        "benchmark": "When2Tool",
        "dataset_id": DATASET_ID,
        "dataset_revision": revision,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "splits": reports,
        "total_tasks": sum(item["n_tasks"] for item in reports.values()),
        "label_warning": (
            "The task dataset contains reference answers and environment metadata, "
            "not model-specific tool-necessity labels. Labels require model evaluation."
        ),
        "status": "PASS",
    }


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def prepare_when2tool(
    output_dir: str | Path,
    *,
    revision: str = DATASET_REVISION,
) -> dict[str, Any]:
    """Export four official splits in the upstream JSON format and write a manifest."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "benchmark": "When2Tool",
        "dataset_id": DATASET_ID,
        "dataset_revision": revision,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {},
        "warning": "No model-specific necessity labels are included in these task files.",
    }
    for (config, split), filename in CONFIG_TO_FILENAME.items():
        tasks = load_when2tool_split(config, split, revision=revision)
        report = _audit_split(tasks, config, split)
        if report["duplicate_ids"] or not report["count_matches_release_card"]:
            raise ValueError(f"Integrity audit failed for {config}/{split}: {report}")
        digest = _write_json(out / filename, tasks)
        manifest["files"][filename] = {
            "config": config,
            "split": split,
            "rows": len(tasks),
            "sha256": digest,
        }
    manifest["total_tasks"] = sum(x["rows"] for x in manifest["files"].values())
    manifest_path = out / "when2tool_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest
