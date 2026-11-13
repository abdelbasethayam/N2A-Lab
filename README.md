# N2A Lab

**When-to-Tool Research Laboratory**

N2A Lab is an open research sandbox for studying when an LLM agent should call a tool, when it should answer directly, and whether the expected benefit justifies tool cost and risk.

## Current release: v0.1 — Synthetic policy sandbox

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
│   ├── __init__.py
│   ├── engine.py
│   └── scenarios.py
├── tests/test_engine.py
├── docs/ARCHITECTURE.md
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
