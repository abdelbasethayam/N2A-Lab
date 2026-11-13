# N2A Lab — Architecture

## Product boundary

N2A Lab is a research workbench, not a general-purpose agent framework. Its central question is:

> Given a task, a candidate tool, a model, and a cost/risk context, when should the agent call the tool, and did that action improve realized utility?

The project keeps experimental decision logic separate from the user interface.

## Current v0.1 components

- app.py: Streamlit UI, experiment controls, comparison table, traces, and exports.
- n2a/scenarios.py: explicit synthetic fixtures and local read-only tool functions.
- n2a/engine.py: deterministic policy execution, expected utility calculation, authorization gate, and trace generation.
- tests/: unit tests for policy behavior and governance gates.

## Decision lifecycle

1. **N — Necessity estimate:** estimate success without the tool.
2. **G — Expected gain:** estimate the incremental success probability from tool access.
3. **U — Expected net utility:** expected gain × value of success − call cost − risk penalty.
4. **A — Action:** call or do not call, subject to authorization.
5. **O — Outcome:** evaluate the returned answer against an independent reference.
6. **R — Realized utility:** observed reward less realized costs and risk penalties.

The labels and probability values in v0.1 are hand-authored teaching fixtures. They are not inferred from a language model.

## Planned adapters

### Benchmark adapter
Load official dataset splits without rewriting their labels. Preserve source item IDs, versions, licenses, and split definitions.

### Black-box model adapter
Call a model API, expose a constrained tool registry, and log observable behavior: prompts, outputs, calls, latency, token usage, and tool results. Never claim hidden-state access in this mode.

### White-box model adapter
Run a pinned open-weight model locally or on an authorized GPU host. Capture the exact hidden-state locations required by the target paper. Store model revision, tokenizer, chat template, layer, token position, dtype, and extraction code.

### Evaluation layer
Compute task success, unnecessary calls, missed necessary calls, calibration metrics, cost per successful task, latency, and utility regret. Use task-cluster bootstrap intervals for aggregate results.

## Evidence modes

- SYNTHETIC_DEMO: hand-authored examples; teaching only.
- BLACK_BOX_RUN: real model API behavior; no hidden-state claims.
- WHITE_BOX_RUN: open-weight model with documented internal activation access.
- PAPER_REPRODUCTION: protocol-matched reproduction with pinned artifacts and raw logs.

Each run must carry exactly one evidence mode. Never mix synthetic results with benchmark results in the same aggregate table.
