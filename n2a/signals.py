"""Join no-tool, forced-tool, reasoning, and probe artifacts into policy signal rows."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

from n2a.necessity import _runs_from_payload, derive_necessity_labels
from n2a.scoring import score_output


def parse_verbalized_necessity(text: str) -> dict[str, Any]:
    """Parse an explicit verbalized tool-necessity claim; ambiguous text stays missing."""
    value = str(text or "").strip().lower()
    value = re.split(r"<tool_call>|\\boxed\\{", value, maxsplit=1)[0]
    value = re.sub(r"\\s+", " ", value)

    no_tool_patterns = [
        r"\bcan solve\b.*\bdirectly\b",
        r"\bcan answer\b.*\bdirectly\b",
        r"\bsolve directly\b",
        r"\banswer directly\b",
        r"\b(?:do not|don't|does not|doesn't) need\b.*\btool\b",
        r"\bno tool\b.*\b(?:needed|necessary|required)\b",
        r"\bwithout (?:using )?a tool\b",
    ]
    need_tool_patterns = [
        r"\bneed(?:s)?\b.*\btool\b",
        r"\bmust\b.*\btool\b",
        r"\brequire(?:s|d)?\b.*\btool\b",
        r"\b(?:cannot|can't|unable to)\b.*\bwithout\b.*\btool\b",
        r"\btool\b.*\b(?:necessary|required)\b",
    ]

    if any(re.search(pattern, value) for pattern in no_tool_patterns):
        return {"score": 0.0, "status": "explicit_no_tool_claim"}
    if any(re.search(pattern, value) for pattern in need_tool_patterns):
        return {"score": 1.0, "status": "explicit_tool_needed_claim"}
    return {"score": None, "status": "ambiguous_or_missing"}


def _outcome_rates(payload: Any) -> dict[Any, dict[str, float | int]]:
    runs = _runs_from_payload(payload)
    by_id: dict[Any, list[bool]] = defaultdict(list)
    for run_index, outputs in enumerate(runs):
        seen = set()
        for item in outputs:
            task_id = item.get("id")
            if task_id is None:
                raise ValueError(f"Output in run {run_index} is missing task id.")
            if task_id in seen:
                raise ValueError(f"Duplicate task {task_id} in run {run_index}.")
            seen.add(task_id)
            by_id[task_id].append(bool(score_output(item)["correct"]))
    rates = {}
    for task_id, outcomes in by_id.items():
        if len(outcomes) != len(runs):
            raise ValueError(f"Task {task_id} is missing from one or more outcome runs.")
        n = len(outcomes)
        successes = sum(outcomes)
        rates[task_id] = {
            "success_probability": successes / n,
            "successes": successes,
            "trials": n,
        }
    return rates


def _first_reasoning_text(item: dict[str, Any]) -> str:
    trace = item.get("trace", [])
    if isinstance(trace, list):
        for entry in trace:
            raw = (entry or {}).get("model_raw_output", "")
            if raw:
                return str(raw)
    return str(item.get("final_response", "") or "")


def _reasoning_scores(payload: Any) -> dict[Any, dict[str, Any]]:
    scores = {}
    for run_index, outputs in enumerate(_runs_from_payload(payload)):
        for item in outputs:
            task_id = item.get("id")
            if task_id is None:
                continue
            parsed = parse_verbalized_necessity(_first_reasoning_text(item))
            # Keep the first run's explicit self-report; do not average categorical parses.
            scores.setdefault(task_id, parsed)
    return scores


def build_policy_signal_records(
    no_tool_payload: Any,
    *,
    label_rule: str = "single_failure",
    force_tool_payload: Any = None,
    reasoning_payload: Any = None,
    probe_payload: Any = None,
    tool_cost: float = 0.0,
    risk_penalty: float = 0.0,
    value_of_success: float = 1.0,
) -> dict[str, Any]:
    """Join empirical outcomes and internal signals without imputing absent evidence."""
    if min(tool_cost, risk_penalty, value_of_success) < 0:
        raise ValueError("Tool cost, risk penalty, and value must be non-negative.")

    necessity = derive_necessity_labels(no_tool_payload, label_rule=label_rule)
    p0_by_id = {
        row["task_id"]: row for row in necessity["labels"]
    }
    p1_by_id = _outcome_rates(force_tool_payload) if force_tool_payload is not None else {}
    verbalized_by_id = _reasoning_scores(reasoning_payload) if reasoning_payload is not None else {}

    probe_by_id = {}
    if probe_payload is not None:
        probe_records = probe_payload.get("records", probe_payload) if isinstance(probe_payload, dict) else probe_payload
        for row in probe_records:
            probe_by_id[row["task_id"]] = row

    records = []
    for task_id, no_tool in p0_by_id.items():
        p0 = float(no_tool["no_tool_success_probability"])
        p1_record = p1_by_id.get(task_id)
        p1 = float(p1_record["success_probability"]) if p1_record else None
        n1 = int(p1_record["trials"]) if p1_record else 0
        uncertainty = None
        if p1 is not None:
            n0 = int(no_tool["no_tool_trials"])
            variance = p0 * (1.0 - p0) / max(n0, 1) + p1 * (1.0 - p1) / max(n1, 1)
            uncertainty = math.sqrt(max(variance, 0.0))

        verbalized = verbalized_by_id.get(task_id, {"score": None, "status": "not_provided"})
        probe = probe_by_id.get(task_id, {})
        record = {
            "task_id": task_id,
            "necessity_label": int(no_tool["tool_necessary"]),
            "no_tool_success_probability": p0,
            "no_tool_trials": int(no_tool["no_tool_trials"]),
            "verbalized_necessity_score": verbalized.get("score"),
            "verbalized_parse_status": verbalized.get("status"),
            "probe_necessity_probability": probe.get("probe_necessity_probability"),
            "expected_no_tool_success": p0,
            "expected_with_tool_success": p1,
            "uncertainty": uncertainty,
            "tool_cost": float(tool_cost),
            "risk_penalty": float(risk_penalty),
            "value_of_success": float(value_of_success),
            "difficulty": probe.get("difficulty", "unknown"),
            "env_name": probe.get("env_name", "unknown"),
            "force_tool_trials": n1,
        }
        records.append(record)

    return {
        "signal_table": "N2A P0-P6 policy inputs",
        "n_records": len(records),
        "label_rule": label_rule,
        "source_run_counts": {
            "no_tool_runs": necessity["n_runs"],
            "force_tool_runs": max((v["trials"] for v in p1_by_id.values()), default=0),
            "reasoning_records": len(verbalized_by_id),
            "probe_records": len(probe_by_id),
        },
        "records": records,
        "signal_semantics": {
            "P2": "Empirical no-tool success frequency; a single trial yields only a 0/1 observation.",
            "P3": "Categorical parsed self-report score; ambiguous/missing statements remain null and are excluded.",
            "P5": "Official linear-probe probability; evaluate on held-out tasks.",
            "P6": "Expected success gain estimated from matched no-tool and forced-tool runs; uncertainty is the standard error of the difference.",
        },
        "warning": (
            "Expected success rates are empirical estimates, not causal effects. Forced-tool and no-tool "
            "runs must be matched by task/model/decoding settings. Cost and risk are user-specified. "
            "Thresholds and utility weights must be fixed without test-set tuning."
        ),
    }
