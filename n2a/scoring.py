"""Answer and tool-call scoring compatible with the official When2Tool output schema."""
from __future__ import annotations

import ast
import re
from typing import Any


def extract_boxed(text: str) -> str:
    text = str(text or "").strip()
    marker = r"\boxed{"
    start = text.find(marker)
    if start < 0:
        return ""
    index = start + len(marker)
    depth = 1
    result = []
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
            result.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
            result.append(char)
        else:
            result.append(char)
        index += 1
    return "".join(result).strip() if result else text


def _last_assistant_text(item: dict[str, Any]) -> str:
    final = item.get("final_response")
    if isinstance(final, str) and final:
        return final
    output = item.get("output", [])
    if isinstance(output, list):
        for message in reversed(output):
            if isinstance(message, dict) and message.get("role") == "assistant":
                if isinstance(message.get("content"), str):
                    return message["content"]
    return ""


def _normalize_structured(value: Any) -> Any:
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [_normalize_structured(x) for x in value]
    if isinstance(value, dict):
        return {k: _normalize_structured(v) for k, v in value.items()}
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"[-+]?\d+", stripped):
            try:
                return int(stripped)
            except ValueError:
                return stripped
        if re.fullmatch(r"[-+]?\d*\.\d+", stripped):
            try:
                return float(stripped)
            except ValueError:
                return stripped
    return value


def _parse_structured(text: str) -> Any:
    try:
        return _normalize_structured(ast.literal_eval(str(text or "").strip()))
    except (ValueError, SyntaxError, TypeError):
        return None


def _normalize_scalar(text: Any) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    value = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", value)
    value = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", value)
    value = value.replace("{", "").replace("}", "").replace("\\", "")
    return re.sub(r"\s+", " ", value).strip().lower()


def compare_answers(prediction: Any, reference: Any) -> bool:
    pred_text, ref_text = str(prediction or "").strip(), str(reference or "").strip()
    pred_struct, ref_struct = _parse_structured(pred_text), _parse_structured(ref_text)
    if pred_struct is not None and ref_struct is not None:
        return pred_struct == ref_struct
    normalized_ref = _normalize_scalar(ref_text)
    if not normalized_ref:
        return False
    return _normalize_scalar(pred_text) == normalized_ref


def score_output(item: dict[str, Any]) -> dict[str, Any]:
    expected = item.get("expected", {}) or {}
    reference = expected.get("answer", "")
    raw = _last_assistant_text(item)
    boxed = extract_boxed(raw)
    return {
        "task_id": item.get("id"),
        "prediction_raw": raw,
        "prediction": boxed,
        "reference": reference,
        "correct": compare_answers(boxed, reference),
    }


def count_tool_calls(item: dict[str, Any]) -> int:
    if "tool_calls" in item:
        try:
            return max(0, int(item.get("tool_calls", 0)))
        except (TypeError, ValueError):
            return 0
    trace = item.get("trace", [])
    if not isinstance(trace, list):
        return 0
    return sum(
        1 for entry in trace
        if (entry or {}).get("parsed_output", {}).get("type") == "tool"
    )
