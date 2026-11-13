"""Streamlit interface for N2A Lab v0.1."""

import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from n2a.engine import ExperimentConfig, POLICIES, run_policy
from n2a.scenarios import SCENARIOS, get_scenario

st.set_page_config(page_title="N2A Lab", page_icon="🧪", layout="wide")

st.markdown(
    "<span style='color:#16a34a;font-weight:700;letter-spacing:.16em'>PAPER RESEARCH PROJECT</span>",
    unsafe_allow_html=True,
)
st.title("🧪 N2A Lab")
st.subheader("Synthetic policy sandbox")
st.caption("Illustrative scenarios for interface and policy-logic testing — not benchmark evidence.")

st.warning(
    "Research status — v0.1 is a synthetic policy simulator with hand-authored scenarios. "
    "It does not call a real LLM, extract hidden states, reproduce a published benchmark, "
    "or establish that one policy is scientifically superior."
)

with st.sidebar:
    st.header("Experiment configuration")
    scenario_id = st.selectbox(
        "Scenario",
        options=list(SCENARIOS),
        format_func=lambda key: SCENARIOS[key].title,
    )
    st.divider()
    st.subheader("Tool properties")
    reliability = st.slider("Tool reliability", 0, 100, 95, 5) / 100
    tool_cost = st.slider("Tool-call cost", 0.0, 1.0, 0.10, 0.05)
    risk_penalty = st.slider("Risk / privacy penalty", 0.0, 1.0, 0.05, 0.05)
    st.divider()
    st.subheader("Decision settings")
    value = st.slider("Value of a correct answer", 0.1, 2.0, 1.0, 0.1)
    threshold = st.slider("Necessity threshold", 0.0, 1.0, 0.60, 0.05)
    self_bias = st.slider("Self-assessment bias", -0.5, 0.5, 0.10, 0.05)
    authorized = st.checkbox("Tool use is authorized", value=True)
    seed = st.number_input("Reproducibility seed", min_value=0, max_value=1000000, value=17)

scenario = get_scenario(scenario_id)
left, right = st.columns([1.6, 1])
with left:
    st.markdown("### Task")
    st.info(scenario.question)
    st.markdown(f"**Candidate tool:** {scenario.tool_name}")
    st.caption(scenario.tool_description)
    st.markdown("**Why this scenario is included**")
    st.write(scenario.rationale)
with right:
    st.markdown("### Scenario reference")
    st.metric("Reference answer", scenario.reference_answer)
    st.metric("Expected necessity label", "Necessary" if scenario.expected_necessity else "Not necessary")
    st.caption("This label is manually assigned for the synthetic teaching example.")

st.divider()
st.markdown("### Compare decision policies")
policy_ids = list(POLICIES)
selected = st.multiselect(
    "Select policies to run",
    options=policy_ids,
    default=policy_ids,
    format_func=lambda key: POLICIES[key],
)
config = ExperimentConfig(
    tool_reliability=reliability,
    tool_cost=tool_cost,
    risk_penalty=risk_penalty,
    value_per_success=value,
    decision_threshold=threshold,
    self_assessment_bias=self_bias,
    authorized=authorized,
    seed=int(seed),
)

if st.button("▶ Run experiment", type="primary", use_container_width=True, disabled=not selected):
    results = [run_policy(pid, scenario, config) for pid in selected]
    st.session_state["results"] = results
    st.session_state["scenario_id"] = scenario_id
    st.session_state["config"] = config.__dict__

if "results" in st.session_state:
    results = st.session_state["results"]
    df = pd.DataFrame([
        {
            "Policy": r["policy"],
            "Tool called": r["tool_called"],
            "Task success": r["success"],
            "Unnecessary call": r["unnecessary_call"],
            "Missed necessary call": r["missed_necessary_call"],
            "Expected gain": round(r["expected_gain"], 3),
            "Estimated net utility": round(r["estimated_net_utility"], 3),
            "Realized utility": round(r["realized_utility"], 3),
        }
        for r in results
    ])

    st.markdown("### Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Successful runs", f"{sum(r['success'] for r in results)}/{len(results)}")
    c2.metric("Tool calls", sum(r["tool_called"] for r in results))
    c3.metric("Unnecessary calls", sum(r["unnecessary_call"] for r in results))
    c4.metric("Missed necessary calls", sum(r["missed_necessary_call"] for r in results))
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("#### Utility comparison")
    st.bar_chart(df.set_index("Policy")[["Estimated net utility", "Realized utility"]])

    st.markdown("#### Decision traces")
    for result in results:
        label = "CALL" if result["tool_called"] else "NO-CALL"
        outcome = "SUCCESS" if result["success"] else "FAILURE"
        with st.expander(f"{result['policy']} — {label} — {outcome}"):
            st.write(f"**Answer:** {result['answer']}")
            st.write(f"**Decision:** {result['decision_reason']}")
            for item in result["trace"]:
                st.markdown(f"- **{item['stage']}:** {item['event']}")

    export = {
        "application": "N2A Lab",
        "version": "0.1.0",
        "evidence_status": "SYNTHETIC_DEMO_NOT_BENCHMARK",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_id": st.session_state["scenario_id"],
        "config": st.session_state["config"],
        "results": results,
    }
    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇ Download CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name="n2a_experiment_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
    d2.download_button(
        "⬇ Download JSON traces",
        json.dumps(export, ensure_ascii=False, indent=2, default=str),
        file_name="n2a_experiment_traces.json",
        mime="application/json",
        use_container_width=True,
    )

with st.expander("How to interpret the metrics"):
    st.markdown(
        """
        - **Necessity:** whether the task's success criterion can be met without the tool by the evaluated model. In v0.1, the label is manually assigned to a synthetic scenario.
        - **Expected gain:** estimated change in success probability when the tool is available.
        - **Estimated net utility:** expected gain × value of success − tool cost − risk penalty.
        - **Over-calling:** calling in a scenario labelled not necessary.
        - **Under-calling:** not calling in a scenario labelled necessary.
        - **Self-assessment:** an illustrative decision signal, not a calibrated probability.
        - **Realized utility:** observed simulator reward minus configured call cost and risk penalty.
        """
    )

with st.expander("Research roadmap"):
    st.markdown(
        """
        1. Import a pinned official When2Tool release and reproduce its baselines.
        2. Add a black-box LLM adapter with explicit secret management and rate limits.
        3. Add an open-weight white-box runner for hidden-state extraction and probe training.
        4. Add statistical evaluation, confidence intervals, calibration metrics, and full run manifests.
        """
    )

st.caption("N2A Lab · v0.2.0 · Synthetic sandbox + benchmark research adapter")
