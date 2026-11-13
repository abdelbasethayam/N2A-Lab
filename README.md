# N2A Lab

**When-to-Tool Research Laboratory**

N2A Lab studies when an LLM agent should call a tool, when it should answer directly, and whether tool use improves task success after accounting for latency, cost, privacy, and risk.

## Current release: v0.2 — Benchmark adapter and policy research layer

### Implemented

- Pinned When2Tool dataset adapter and official task-schema normalization.
- Live integrity audit for all four official splits (3,780 tasks).
- Streamlit pages for benchmark exploration and interactive P0–P6 policy comparison.
- Explicit model-relative necessity-label construction from hard-no-tool outputs.
- P0–P6 decision policies, including self-report, surface heuristics, hidden-state probes, and uncertainty-/utility-aware decisions.
- Matched no-tool / forced-tool / reasoning / probe signal joins.
- Original linear-probe inference and a repeated-label probe-training extension.
- Decision, calibration, and utility metrics; CSV/JSON research exports.
- Pinned upstream bootstrap, cloud-GPU pilot, and repeated-trial runners.
- Unit tests and GitHub Actions validation.

### Scientific boundary

The original v0.1 synthetic sandbox remains available for teaching and UI testing only. It is not empirical evidence. The live benchmark audit verifies dataset ingestion and split integrity; it does not execute a language model. A GPU pilot is a runtime smoke test, not a paper reproduction. No claim that N2A outperforms existing methods is made at this stage.

## Run locally

Python 3.10+ is recommended.

\`\`\`bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
\`\`\`

The app includes:
- **When2Tool Benchmark Explorer:** inspect official splits, difficulty, and environment distributions.
- **N2A Policy Lab:** upload a model-derived signal table and compare eligible P0–P6 policies.

## Pinned benchmark

- Dataset: [cesun/When2Tool](https://huggingface.co/datasets/cesun/When2Tool)
- Dataset revision: \`153066c0285887915fc6b530c9af2987a2c859ac\`
- Official runner: [Trustworthy-ML-Lab/when2tool](https://github.com/Trustworthy-ML-Lab/when2tool)
- Runner commit: \`ef0cf1ea72eb8acdfd3fb1251c4b6a9a0131a23e\`

| Configuration | Train | Test | Total |
|---|---:|---:|---:|
| Single-hop | 900 | 2,250 | 3,150 |
| Multi-hop | 180 | 450 | 630 |
| **Total** | **1,080** | **2,700** | **3,780** |

The public task dataset does not contain universal tool-necessity labels. Necessity is model-, prompt-, decoding-, and evaluator-relative; it must be derived from no-tool model runs.

## Prepare and audit the data

\`\`\`bash
pip install -r requirements.txt
python -m n2a.cli audit-when2tool --output-dir data/when2tool
\`\`\`

The command exports the four official splits in upstream-compatible JSON format and writes a manifest with pinned revisions and SHA-256 hashes. GitHub Actions also runs this live audit against the pinned Hugging Face dataset.

## Run a cloud GPU pilot

Use a Linux NVIDIA GPU runtime. The pinned upstream runner uses vLLM and Transformers; a standard Mac or CPU-only Streamlit instance is not the target runtime.

\`\`\`bash
pip install -r requirements-research.txt
bash scripts/run_when2tool.sh --model qwen3-4b-instruct --pilot 100
\`\`\`

Run the official five-step single-hop pipeline with:

\`\`\`bash
bash scripts/run_when2tool.sh --model qwen3-4b-instruct --full
\`\`\`

The pilot runs a 100-task slice across the official prompt modes. The full pipeline performs baseline evaluation, hard-no-tool labels and hidden-state extraction, linear-probe training, Probe&Prefill evaluation, and figure generation. Preserve all raw outputs and run metadata.

## Repeated-trial N2A extension

The original P31/When2Tool one-run labels must remain separate from repeated-trial labels. To estimate no-tool and forced-tool success rates, collect matched repeated trials:

\`\`\`bash
bash scripts/run_repeated_signals.sh --model qwen3-4b-instruct --split train --runs 10
bash scripts/run_repeated_signals.sh --model qwen3-4b-instruct --split test --runs 10
\`\`\`

The repeated-trial protocol retrains P5 against the same repeated-run label definition before comparing P0–P6. It does not reuse a one-run probe against a different target. Full commands, signal semantics, and interpretation limits are documented in [WHEN2TOOL_REPRODUCTION.md](docs/WHEN2TOOL_REPRODUCTION.md).

## CLI research workflow

- \`audit-when2tool\`: download, audit, and export pinned benchmark splits.
- \`derive-labels\`: derive necessity labels from hard-no-tool model outputs.
- \`probe-signals\`: export per-task P5 probabilities from official probe artifacts.
- \`train-repeated-probe\`: train P5 using repeated-run labels.
- \`build-signals\`: join no-tool, forced-tool, reasoning, and probe evidence.
- \`policy-suite\`: compare selected P0–P6 policies.
- \`evaluate-outputs\`: evaluate observed tool-using outputs and realized utility.

## Repository map

\`\`\`text
.
├── app.py
├── n2a/
│   ├── benchmarks/when2tool.py
│   ├── cli.py
│   ├── engine.py
│   ├── evaluation.py
│   ├── metrics.py
│   ├── necessity.py
│   ├── policies.py
│   ├── policy_evaluation.py
│   ├── probes.py
│   ├── scenarios.py
│   ├── scoring.py
│   └── signals.py
├── pages/
│   ├── 1_When2Tool_Benchmark.py
│   └── 2_N2A_Policy_Lab.py
├── scripts/
├── tests/
├── docs/
└── requirements*.txt
\`\`\`

## Research roadmap

1. **v0.1:** synthetic sandbox for interface and policy-logic testing.
2. **v0.2:** pinned benchmark adapter, data audit, P0–P6 policies, metrics, and cloud runner.
3. **v0.3:** protocol-matched When2Tool reproduction and independent result validation.
4. **v0.4:** Knowing–Doing Gap experiments with separate belief, signal, action, and outcome measures.
5. **v0.5:** cost-sensitive N2A policy evaluation with validation-only calibration and uncertainty analysis.
6. **v1.0:** fully reproducible research lab with pinned artifacts, statistical reports, and publication-ready figures.

For development tests:

\`\`\`bash
pip install -r requirements-dev.txt
python -m pytest -q
\`\`\`

See [Architecture](docs/ARCHITECTURE.md) and [Research Protocol](docs/RESEARCH_PROTOCOL.md) before interpreting results.
