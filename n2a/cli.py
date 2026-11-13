"""Command-line interface for reproducible N2A benchmark preparation and analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from n2a.benchmarks.when2tool import audit_when2tool, prepare_when2tool
from n2a.evaluation import evaluate_outputs
from n2a.necessity import derive_necessity_labels
from n2a.policy_evaluation import evaluate_policy_suite
from n2a.policies import POLICY_NAMES, PolicyConfig
from n2a.probes import export_probe_records


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(prog="n2a", description="N2A Lab research CLI")
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit-when2tool", help="Download, audit, and export the four official dataset splits.")
    audit.add_argument("--output-dir", default="data/when2tool")
    audit.add_argument("--revision", default=None, help="Optional immutable Hugging Face dataset revision.")

    labels = commands.add_parser("derive-labels", help="Derive model-relative necessity labels from hard-no-tool outputs.")
    labels.add_argument("--no-tool-outputs", required=True, help="JSON list or run_0/run_1 JSON object.")
    labels.add_argument("--label-rule", choices=["single_failure", "any_failure", "majority_failure"], default="single_failure")
    labels.add_argument("--output", required=True)

    evaluate = commands.add_parser("evaluate-outputs", help="Evaluate tool-using outputs against model-relative necessity labels.")
    evaluate.add_argument("--outputs", required=True)
    evaluate.add_argument("--labels", required=True)
    evaluate.add_argument("--success-value", type=float, default=1.0)
    evaluate.add_argument("--cost-per-call", type=float, default=0.0)
    evaluate.add_argument("--risk-per-call", type=float, default=0.0)
    evaluate.add_argument("--output", required=True)

    suite = commands.add_parser("policy-suite", help="Compare P0-P6 on a table of model-derived signals.")
    suite.add_argument("--records", required=True, help="JSON list with necessity_label and policy signal fields.")
    suite.add_argument("--output", required=True)
    suite.add_argument("--necessity-threshold", type=float, default=0.5)
    suite.add_argument("--verbalized-threshold", type=float, default=0.5)
    suite.add_argument("--surface-threshold", type=float, default=0.5)
    suite.add_argument("--probe-threshold", type=float, default=0.5)
    suite.add_argument("--uncertainty-aversion", type=float, default=0.0)
    suite.add_argument("--policies", nargs="+", choices=list(POLICY_NAMES), default=list(POLICY_NAMES))

    probe = commands.add_parser("probe-signals", help="Export per-task P5 probabilities from official probe artifacts.")
    probe.add_argument("--hidden", required=True)
    probe.add_argument("--probe", required=True)
    probe.add_argument("--labels", required=True)
    probe.add_argument("--temperature", type=float, default=1.0)
    probe.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.command == "audit-when2tool":
        kwargs = {"output_dir": args.output_dir}
        if args.revision:
            kwargs["revision"] = args.revision
        audit_result = audit_when2tool(revision=args.revision) if args.revision else audit_when2tool()
        manifest = prepare_when2tool(**kwargs)
        _write_json(str(Path(args.output_dir) / "when2tool_audit.json"), audit_result)
        print(json.dumps({
            "status": "PASS",
            "total_tasks": manifest["total_tasks"],
            "output_dir": str(Path(args.output_dir).resolve()),
            "manifest": str(Path(args.output_dir) / "when2tool_manifest.json"),
        }, indent=2))
    elif args.command == "derive-labels":
        payload = _read_json(args.no_tool_outputs)
        result = derive_necessity_labels(payload, label_rule=args.label_rule)
        _write_json(args.output, result)
        print(json.dumps({k: result[k] for k in ("label_rule", "n_runs", "n_tasks", "n_necessary", "n_unnecessary")}, indent=2))
    elif args.command == "evaluate-outputs":
        outputs = _read_json(args.outputs)
        labels_payload = _read_json(args.labels)
        result = evaluate_outputs(
            outputs,
            labels_payload,
            success_value=args.success_value,
            cost_per_call=args.cost_per_call,
            risk_per_call=args.risk_per_call,
        )
        _write_json(args.output, result)
        print(json.dumps({
            "n_matched_output_records": result["n_matched_output_records"],
            "decision_metrics": result["decision_metrics"],
            "outcome_utility": result["outcome_utility"],
            "output": str(Path(args.output).resolve()),
        }, indent=2))
    elif args.command == "policy-suite":
        records_payload = _read_json(args.records)
        records = records_payload.get("records", records_payload) if isinstance(records_payload, dict) else records_payload
        config = PolicyConfig(
            necessity_threshold=args.necessity_threshold,
            verbalized_threshold=args.verbalized_threshold,
            surface_threshold=args.surface_threshold,
            probe_threshold=args.probe_threshold,
            uncertainty_aversion=args.uncertainty_aversion,
        )
        result = evaluate_policy_suite(records, config=config, policies=tuple(args.policies))
        _write_json(args.output, result)
        print(json.dumps({
            "n_records": result["n_records"],
            "policies": {
                key: value["decision_metrics"]
                for key, value in result["policies"].items()
            },
            "output": str(Path(args.output).resolve()),
        }, indent=2))
    elif args.command == "probe-signals":
        result = export_probe_records(
            args.hidden, args.probe, args.labels, args.output,
            temperature=args.temperature,
        )
        print(json.dumps({
            "n_records": result["n_records"],
            "signal_name": result["signal_name"],
            "output": str(Path(args.output).resolve()),
        }, indent=2))


if __name__ == "__main__":
    main()
