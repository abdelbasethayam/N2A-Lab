# Research Protocol and Reproducibility Contract

## 1. Keep the constructs separate

- **Actual/model-relative necessity:** whether a specified model meets the task success criterion without a tool under a specified decoding protocol.
- **Perceived necessity:** the model's explicit or implicit estimate that a tool is needed.
- **Decodability:** whether a probe can predict a label from a specified internal representation on held-out data.
- **Action:** whether the model calls, declines, asks, abstains, verifies, or stops.
- **Utility:** incremental outcome value after tool cost, latency, reliability, privacy, and risk.
- **Answer confidence:** confidence that the final answer is correct; this is not automatically necessity confidence.

## 2. Repeated-run necessity labels

Do not treat one model response as ground truth. For each model/task/prompt/decoding configuration:

1. Run no-tool trials with documented seeds and temperature.
2. Preserve every output, not only aggregate success.
3. Report the number of trials and exact label rule.
4. Run sensitivity analysis over trial count and thresholds.
5. Keep labels from different papers separate until harmonized.

Strict rules such as “any failure across ten trials means necessary” are not interchangeable with lenient rules such as “at least one success across five trials means answerable.”

## 3. Matched comparisons

Compare policies on identical task IDs, model revisions, prompts, decoding settings, tool implementations, and cost schedules. Report task success, unnecessary-call rate, missed-necessary-call rate, call rate, expected and realized net utility, cost per successful task, latency, token usage, tool failures/retries, calibration metrics only when meaningful probabilities exist, and probe AUROC only for held-out leakage-controlled splits.

## 4. Minimum run manifest

Every non-synthetic run must record:
- paper/dataset version and immutable revision;
- task IDs and split;
- model name and immutable revision;
- tokenizer and chat-template version;
- prompt and system-instruction hashes;
- decoding parameters and seeds;
- tool schema, implementation revision, reliability, and cost schedule;
- policy name and version;
- raw model outputs and tool traces;
- evaluator version and metric definitions;
- hardware/runtime and package lockfile.

## 5. Semantic alignment audit

Before claiming reproduction, translate the source paper into atomic checks: numerical settings, models and revisions, datasets and splits, prompts and templates, algorithms and stopping rules, layer/token/activation extraction locations, procedure ordering, metric formulas, and statistical tests.

A passing unit test or successful program execution is not evidence that the implementation matches the paper.

## 6. Claim language

Use “synthetic demonstration” for v0.1, “behavioral evaluation” for black-box model runs, “representation probe” for white-box decodability experiments, and “reproduction” only after protocol alignment and artifact comparison.

Do not claim that a model “knows” in the human sense based only on probe accuracy. Prefer “the representation contains information predictive of the operational label.”
