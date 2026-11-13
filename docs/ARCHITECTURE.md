# N2A Lab — Architecture

## Product boundary

N2A Lab is a research workbench, not a general-purpose agent framework. Its central question is:

> Given a task, a model, a candidate tool, and a cost/risk context, when should the agent call the tool, and does that action improve task success and net utility?

The project separates benchmark ingestion, model execution, signal extraction, decision policies, outcome evaluation, and user interface. Synthetic demonstrations never share result tables with real model runs.

## Current components

| Component | Responsibility |
|---|---|
| app.py | Synthetic policy sandbox and top-level app |
| pages/1_When2Tool_Benchmark.py | Pinned public benchmark explorer |
| n2a/benchmarks/when2tool.py | Hugging Face loading, official task-schema normalization, split audit, immutable manifest |
| n2a/scenarios.py | Hand-authored synthetic fixtures only |
| n2a/engine.py | Synthetic policy simulator only |
| n2a/scoring.py | Answer and tool-call scoring compatible with official output schema |
| n2a/necessity.py | Model-relative labels from explicit hard-no-tool trials |
| n2a/probes.py | Official probe inference and repeated-label probe training |
| n2a/signals.py | Matched no-tool, forced-tool, reasoning, and probe signal joins |
| n2a/policies.py | P0–P6 policy definitions |
| n2a/policy_evaluation.py | Policy comparison with per-policy signal eligibility |
| n2a/metrics.py | Decision, calibration, and utility metrics |
| n2a/evaluation.py | Evaluation of observed model outputs |
| n2a/cli.py | Reproducible data, label, signal, probe, and evaluation commands |
| scripts/ | Pinned upstream bootstrap and cloud GPU runners |
| tests/ | Unit and integration-contract tests |

## Evidence and construct separation

The system keeps these constructs separate:

1. **Capability / necessity (N):** empirical success without a tool under a specified model, prompt, decoding, and evaluator protocol.
2. **Internal signal (Z):** a representation or observable feature that predicts the operational necessity label.
3. **Belief / self-assessment (B):** the model's verbalized or otherwise explicit estimate of necessity.
4. **Action (A):** call, no-call, abstain, clarify, verify, or stop.
5. **Outcome (O):** answer correctness and task completion.
6. **Utility (U):** outcome value minus tool cost, latency, privacy, and risk.

A probe's predictive performance does not establish awareness, causal control, or good policy behavior. Probe&Prefill is an intervention and must be evaluated separately from probe decodability.

## P0–P6 policies

- **P0:** never call.
- **P1:** always call.
- **P2:** call when estimated no-tool success probability falls below a fixed threshold.
- **P3:** act on an explicit verbalized necessity score; ambiguous statements are excluded rather than imputed.
- **P4:** transparent surface/difficulty heuristic.
- **P5:** act on a held-out hidden-state probe probability.
- **P6:** act when expected marginal success value exceeds tool cost, risk, and uncertainty penalty.

The policies consume precomputed signals. They do not generate missing signals or silently impute them. The evaluator reports eligible sample counts separately for each policy.

## Decision lifecycle

1. Estimate no-tool capability from repeated no-tool trials.
2. Estimate tool-enabled capability from matched forced-tool trials.
3. Extract self-report and hidden-state signals.
4. Apply a fixed P0–P6 decision rule.
5. Execute or ingest the corresponding model action.
6. Score final correctness, call count, latency, and utility.

Expected probabilities and thresholds must be calibrated or selected on training/validation data. Locked test data must not be used for tuning.

## Model execution

The When2Tool adapter pins the public Hugging Face dataset revision and the official GitHub runner commit. The cloud runner bootstraps that exact upstream revision instead of silently copying or modifying its code.

- The official five-step pipeline reproduces the upstream baseline, hard-no-tool labels, hidden-state extraction, linear probe, Probe&Prefill, and figures.
- The repeated-trial runner is an N2A extension. It collects multiple hard-no-tool and forced-tool attempts at a declared temperature.
- When the label definition changes from one-run failure to repeated-run failure, P5 is retrained against the repeated-run labels. The original probe is not evaluated against a different target.

## Evidence modes

- SYNTHETIC_DEMO: hand-authored examples; teaching only.
- P31_OFFICIAL_REPRODUCTION: protocol-aligned execution of the pinned When2Tool pipeline.
- N2A_REPEATED_TRIAL_EXTENSION: repeated no-tool/forced-tool runs with a declared label rule.
- BLACK_BOX_RUN: observable API behavior only; no hidden-state claims.
- WHITE_BOX_RUN: open-weight model with documented internal activation access.

A successful import, passing tests, or a pilot is not a reproduction. Every empirical run must preserve immutable source pins, model revision, tokenizer/chat template, prompt, decoding settings, trial count, task IDs, raw outputs, tool traces, evaluator version, and cost schedule.
