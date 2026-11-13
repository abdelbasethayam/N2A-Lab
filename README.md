# N2A Lab

**When-to-Tool Research Laboratory**

N2A Lab is an open research sandbox for studying when an LLM agent should call a tool, when it should answer directly, and whether the expected benefit justifies tool cost and risk.

## Current release: v0.2 — Benchmark adapter + policy research layer

This first release provides a browser-based Streamlit interface, five decision policies, matched synthetic scenarios, tool reliability/cost/risk controls, side-by-side metrics, decision traces, CSV/JSON exports, unit tests, and GitHub Actions CI.

> **Scientific boundary:** v0.1 uses hand-authored synthetic scenarios and illustrative probability estimates. It is not an LLM benchmark reproduction, does not access hidden states, and does not demonstrate that N2A outperforms existing methods. Results must be labelled SYNTHETIC.

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Prepare the When2Tool benchmark

```bash
pip install -r requirements.txt
python -m n2a.cli audit-when2tool --output-dir data/when2tool
```

The adapter pins the Hugging Face dataset revision and the official upstream runner commit. Public task rows do not contain model-specific necessity labels; collect those by evaluating a model in hard-no-tool mode. See [When2Tool reproduction protocol](docs/WHEN2TOOL_REPRODUCTION.md).

After collecting official model outputs, derive labels and evaluate decisions:

```bash
python -m n2a.cli derive-labels --no-tool-outputs artifacts/when2tool/probe_data/qwen3-4b-instruct/test_no_tool_outputs.json --label-rule single_failure --output artifacts/when2tool/test_necessity_labels.json
python -m n2a.cli evaluate-outputs --outputs artifacts/when2tool/outputs/qwen3-4b-instruct/current__no_reasoning.json --labels artifacts/when2tool/test_necessity_labels.json --cost-per-call 0.01 --risk-per-call 0.00 --output artifacts/when2tool/current_behavior_metrics.json
```

For a P0–P6 comparison, prepare one row per task with the model-derived necessity label and the required signals (no-tool success probability, verbalized necessity, probe probability, expected success with/without tool, tool cost, risk, and uncertainty), then run:

```bash
python -m n2a.cli policy-suite --records artifacts/when2tool/policy_signals_test.json --output artifacts/when2tool/p0_p6_comparison.json
```

## Run a cloud GPU pilot

On a Linux NVIDIA GPU runtime:

```bash
pip install -r requirements-research.txt
bash scripts/run_when2tool.sh --model qwen3-4b-instruct --pilot 100
```

Use `--full` for the official five-step single-hop pipeline. A pilot is a runtime smoke test, not a paper reproduction.

## Deploy free

Compatible with [Streamlit Community Cloud](https://share.streamlit.io/).

- Repository: `abdelbasethayam/N2A-Lab`
- Branch: `main`
- Entrypoint: `app.py`

## Repository map

```text
.
├── app.py
├── n2a/
│   ├── benchmarks/when2tool.py
│   ├── engine.py
│   ├── evaluation.py
│   ├── metrics.py
│   ├── necessity.py
│   ├── policies.py
│   ├── scenarios.py
│   └── scoring.py
├── pages/1_When2Tool_Benchmark.py
├── scripts/
├── tests/
├── docs/ARCHITECTURE.md
├── docs/WHEN2TOOL_REPRODUCTION.md
├── docs/RESEARCH_PROTOCOL.md
├── .github/workflows/tests.yml
└── requirements.txt
```

## Research roadmap

1. v0.1 — Synthetic sandbox and policy logic.
2. v0.2 — Import a pinned official When2Tool release and reproduce published baselines.
3. v0.3 — Add black-box LLM agents and observable traces.
4. v0.4 — Add open-weight white-box experiments for hidden-state extraction and probe training.
5. v1.0 — Reproducible research lab with pinned artifacts, statistical analysis, and public reports.

For local tests, install development dependencies with `pip install -r requirements-dev.txt`, then run `pytest -q`.

Read [Architecture](docs/ARCHITECTURE.md) and [Research protocol](docs/RESEARCH_PROTOCOL.md) before interpreting results.
