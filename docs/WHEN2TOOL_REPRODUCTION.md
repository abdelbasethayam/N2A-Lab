# When2Tool Integration and Reproduction Protocol

## Immutable source pins

N2A Lab pins two separate artifacts:

- Official runner: \`Trustworthy-ML-Lab/when2tool\` at commit \`ef0cf1ea72eb8acdfd3fb1251c4b6a9a0131a23e\`.
- Dataset: \`cesun/When2Tool\` at revision \`153066c0285887915fc6b530c9af2987a2c859ac\`.

Do not silently update either pin. A pin update requires a new audit, a changelog entry, and regenerated manifests. Before each GPU run, N2A resolves the model alias to an immutable Hugging Face commit, downloads that snapshot, and writes `artifacts/when2tool/model_manifest.json`. You may pass `--revision HF_COMMIT_SHA` to either runner to select a specific model revision.

## Dataset structure

| Configuration | Train | Test | Total |
|---|---:|---:|---:|
| single_hop | 900 | 2,250 | 3,150 |
| multi_hop | 180 | 450 | 630 |
| **Total** | **1,080** | **2,700** | **3,780** |

The public task rows contain task ID, difficulty, instruction, environment, available tools, environment parameters, reference answer, steps, and tags. They do **not** contain universal model-independent necessity labels.

The adapter in \`n2a/benchmarks/when2tool.py\` converts Hugging Face rows to the task schema expected by the official runner. It does not synthesize necessity labels.

## Prepare and audit the dataset

Install the lightweight benchmark dependencies:

\`\`\`bash
pip install -r requirements.txt
python -m n2a.cli audit-when2tool --output-dir data/when2tool
\`\`\`

This exports the four upstream-compatible JSON files and a manifest containing source revisions, row counts, and SHA-256 hashes. The audit checks expected split sizes, duplicate IDs, and train/test ID separation.

## GPU pilot

The official model pipeline uses vLLM and Transformers. Run it in a Linux x86_64 NVIDIA cloud GPU runtime with Python 3.11 and a compatible CUDA 12.4 environment, not on a CPU-only Streamlit instance or a standard Mac environment.

\`\`\`bash
pip install -r requirements-research.txt
bash scripts/run_when2tool.sh --model qwen3-4b-instruct --pilot 100
\`\`\`

The pilot runs the official five prompt modes on a 100-task slice, without reasoning, using one run. It is a runtime smoke test, not a paper reproduction and not suitable for final claims.

## Full single-hop reproduction

\`\`\`bash
bash scripts/run_when2tool.sh --model qwen3-4b-instruct --full
\`\`\`

The pinned upstream pipeline runs:
1. Prompt baselines across the official prompt and reasoning settings.
2. Model-specific hard-no-tool evaluation and hidden-state extraction.
3. Linear-probe training on the training split.
4. Probe&Prefill evaluation with threshold sweeps.
5. Accuracy/tool-call trade-off figures.

The default upstream settings are computationally expensive. Start with one model and retain the exact model revision, hardware, package versions, decoding settings, seeds, output files, and run logs.

## Necessity-label semantics

The official When2Tool feature-extraction code derives a binary label from its hard-no-tool model evaluation: \`tool_necessary = 1\` when the model's no-tool answer is incorrect; otherwise \`0\`.

N2A's \`derive_necessity_labels\` makes the label rule explicit:
- \`single_failure\`: exactly one no-tool outcome per task; reproduces the one-run binary rule.
- \`any_failure\`: for repeated attempts, mark necessary if at least one no-tool attempt fails.
- \`majority_failure\`: for repeated attempts, mark necessary if fewer than half of no-tool attempts succeed.

These rules are not interchangeable. Always record the chosen rule and number of trials. For P32-style repeated-attempt analyses, keep those labels separate from the P31/When2Tool one-run labels.

## N2A P0–P6 policy layer

| Policy | Definition | Required signal |
|---|---|---|
| P0 | Never call | None |
| P1 | Always call | None |
| P2 | No-tool capability threshold | Empirical no-tool success probability |
| P3 | Verbalized necessity | Explicit model self-report score |
| P4 | Surface/difficulty heuristic | Difficulty or predeclared surface score |
| P5 | Internal-state probe | Held-out probe probability |
| P6 | N2A uncertainty-/utility-aware | Expected success gain, tool cost, risk, uncertainty |

Thresholds and costs must be selected on training/validation data or preregistered. Do not tune them on the locked test split.

## Build per-task N2A signals

After the official pipeline has produced its model outputs and probe artifacts, export P5 probabilities:

```bash
python -m n2a.cli probe-signals \
  --hidden artifacts/when2tool/probe_data/qwen3-4b-instruct/test_hidden_no_reasoning.pt \
  --probe artifacts/when2tool/probe_data/qwen3-4b-instruct/probe_no_reasoning.pt \
  --labels artifacts/when2tool/probe_data/qwen3-4b-instruct/test_labels_no_reasoning.json \
  --output artifacts/when2tool/p5_probe_signals_test.json
```

Join matched no-tool, forced-tool, reasoning, and probe evidence:

```bash
python -m n2a.cli build-signals \
  --no-tool-outputs artifacts/when2tool/probe_data/qwen3-4b-instruct/test_no_tool_outputs.json \
  --force-tool-outputs artifacts/when2tool/outputs/qwen3-4b-instruct/force_tool__no_reasoning.json \
  --reasoning-outputs artifacts/when2tool/outputs/qwen3-4b-instruct/current__reasoning.json \
  --probe-records artifacts/when2tool/p5_probe_signals_test.json \
  --tool-cost 0.01 --risk-penalty 0.00 \
  --output artifacts/when2tool/n2a_policy_signals_test.json
```

Then compare the policies:

```bash
python -m n2a.cli policy-suite \
  --records artifacts/when2tool/n2a_policy_signals_test.json \
  --output artifacts/when2tool/p0_p6_comparison_test.json
```

The signal join deliberately does not expose a one-run 0/1 no-tool outcome as a probability. In the original P31 track, P2 and P6 are therefore excluded; compare P0/P1/P3/P4/P5 against the original one-run labels.

## Repeated-trial N2A extension

To estimate model-relative no-tool success probabilities and expected tool benefit, collect matched repeated trials on both train and test splits:

```bash
bash scripts/run_repeated_signals.sh --model qwen3-4b-instruct --split train --runs 10
bash scripts/run_repeated_signals.sh --model qwen3-4b-instruct --split test --runs 10
```

This uses hard-no-tool evaluation for no-tool trials and the official force-tool prompt for the tool-enabled condition. The temperature defaults to 0.7. It is an N2A repeated-trial extension, not the original P31 one-run protocol.

Train a new P5 probe on repeated-run labels so the probe target matches the repeated-run policy labels:

```bash
python -m n2a.cli train-repeated-probe \
  --train-hidden artifacts/when2tool/probe_data/qwen3-4b-instruct/train_hidden_no_reasoning.pt \
  --train-metadata artifacts/when2tool/probe_data/qwen3-4b-instruct/train_labels_no_reasoning.json \
  --train-no-tool-outputs artifacts/when2tool/repeated/qwen3-4b-instruct_train_hard_no_tool_10runs.json \
  --test-hidden artifacts/when2tool/probe_data/qwen3-4b-instruct/test_hidden_no_reasoning.pt \
  --test-metadata artifacts/when2tool/probe_data/qwen3-4b-instruct/test_labels_no_reasoning.json \
  --test-no-tool-outputs artifacts/when2tool/repeated/qwen3-4b-instruct_test_hard_no_tool_10runs.json \
  --label-rule any_failure \
  --output-dir artifacts/when2tool/probe_data/qwen3-4b-instruct_repeated
```

Build the matched test signal table from the repeated outcomes, the repeated-label probe, and the reasoning-mode self-report:

```bash
python -m n2a.cli build-signals \
  --no-tool-outputs artifacts/when2tool/repeated/qwen3-4b-instruct_test_hard_no_tool_10runs.json \
  --label-rule any_failure \
  --force-tool-outputs artifacts/when2tool/repeated/qwen3-4b-instruct_test_force_tool_10runs.json \
  --reasoning-outputs artifacts/when2tool/outputs/qwen3-4b-instruct/current__reasoning.json \
  --probe-records artifacts/when2tool/probe_data/qwen3-4b-instruct_repeated/test_repeated_probe_signals.json \
  --tool-cost 0.01 --risk-penalty 0.00 \
  --output artifacts/when2tool/n2a_repeated_policy_signals_test.json
```

Finally, compare all seven policies using predeclared thresholds and utility weights:

```bash
python -m n2a.cli policy-suite \
  --records artifacts/when2tool/n2a_repeated_policy_signals_test.json \
  --necessity-threshold 0.5 \
  --probe-threshold 0.5 \
  --uncertainty-aversion 0.0 \
  --output artifacts/when2tool/p0_p6_repeated_comparison_test.json
```

The repeated-label probe is intentionally retrained using the same repeated-trial label rule; the original one-run probe must not be evaluated against a different target definition. The suite reports eligible sample counts separately for each policy. P3 is a categorical parse of explicit verbalized necessity, not a calibrated probability. The example cost value is illustrative and must be replaced or preregistered for a real utility study.

## Metrics

The implementation reports action accuracy, precision, recall, F1, MCC, over-calling and missed-necessity rates, Brier score and ECE when probabilities are available, task success, tool calls, tool cost, risk cost, latency cost, net utility, and cost per success.

Utility units are user-specified. Cross-paper utility comparisons are invalid unless reward and cost schedules are aligned.

## What counts as a valid reproduction?

A successful import, passing unit tests, or a pilot run is not a reproduction. A reproduction requires matching the paper's pinned model, data splits, prompt templates, decoding settings, evaluation loop, label construction, probe extraction location, training procedure, and metrics. Keep raw per-task outputs and a complete run manifest.
