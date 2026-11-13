"""Interactive P0-P6 comparison using an uploaded, model-derived signal table."""
import json

import pandas as pd
import streamlit as st

from n2a.policies import POLICY_NAMES, PolicyConfig
from n2a.policy_evaluation import evaluate_policy_suite

st.set_page_config(page_title="N2A Policy Lab", page_icon="⚖️", layout="wide")
st.title("N2A Policy Lab")
st.caption("Compare when-to-call policies using model-derived, task-matched signals.")

st.warning(
    "Upload a signal table produced by the N2A CLI. This page does not invent missing "
    "necessity labels, probabilities, or tool outcomes. Policies with insufficient signals "
    "are shown as unavailable rather than silently imputed."
)

uploaded = st.file_uploader("N2A signal table (JSON)", type=["json"])
with st.sidebar:
    st.header("Decision thresholds")
    necessity_threshold = st.slider("P2 no-tool threshold", 0.0, 1.0, 0.5, 0.05)
    verbalized_threshold = st.slider("P3 self-report threshold", 0.0, 1.0, 0.5, 0.05)
    surface_threshold = st.slider("P4 surface threshold", 0.0, 1.0, 0.5, 0.05)
    probe_threshold = st.slider("P5 probe threshold", 0.0, 1.0, 0.5, 0.05)
    uncertainty_aversion = st.slider("P6 uncertainty aversion", 0.0, 2.0, 0.0, 0.05)
    selected_policies = st.multiselect(
        "Policies",
        options=list(POLICY_NAMES),
        default=list(POLICY_NAMES),
        format_func=lambda key: f"{key} — {POLICY_NAMES[key]}",
    )

if uploaded is not None:
    try:
        payload = json.load(uploaded)
        records = payload.get("records", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError("The uploaded JSON must contain a list named 'records'.")
        st.success(f"Loaded {len(records):,} task records.")
        if isinstance(payload, dict):
            with st.expander("Signal table provenance"):
                st.json({
                    key: value for key, value in payload.items()
                    if key != "records"
                })
        if st.button("Evaluate selected policies", type="primary", disabled=not selected_policies):
            config = PolicyConfig(
                necessity_threshold=necessity_threshold,
                verbalized_threshold=verbalized_threshold,
                surface_threshold=surface_threshold,
                probe_threshold=probe_threshold,
                uncertainty_aversion=uncertainty_aversion,
            )
            result = evaluate_policy_suite(
                records, config=config, policies=tuple(selected_policies)
            )
            st.session_state["policy_suite_result"] = result
    except Exception as exc:
        st.error(f"Could not read or evaluate this signal table: {exc}")

if "policy_suite_result" in st.session_state:
    result = st.session_state["policy_suite_result"]
    st.markdown("### Policy comparison")
    rows = []
    for policy_id, item in result["policies"].items():
        metrics = item.get("decision_metrics") or {}
        rows.append({
            "Policy": f"{policy_id} — {item['policy_name']}",
            "Status": item.get("status", "EVALUATED"),
            "N evaluated": item.get("n_evaluated", 0),
            "Action accuracy": metrics.get("action_accuracy"),
            "Over-call rate": metrics.get("over_call_rate"),
            "Missed-necessity rate": metrics.get("missed_necessity_rate"),
            "Brier": metrics.get("brier_score"),
            "ECE": metrics.get("ece"),
            "Mean estimated utility": item.get("mean_estimated_utility"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Brier/ECE appear only for signals treated as probabilities. Estimated utility is "
        "not realized utility; use executed outcomes for final utility claims."
    )
    for policy_id, item in result["policies"].items():
        with st.expander(f"{policy_id} — {item['policy_name']}"):
            if item.get("status") == "NO_ELIGIBLE_RECORDS":
                st.warning("No records contain the required signal for this policy.")
            else:
                st.json({
                    "decision_metrics": item["decision_metrics"],
                    "mean_estimated_utility": item["mean_estimated_utility"],
                    "n_utility_scored": item["n_utility_scored"],
                    "n_excluded_missing_signal": item["n_excluded_missing_signal"],
                })
    st.download_button(
        "Download full policy comparison",
        json.dumps(result, ensure_ascii=False, indent=2),
        file_name="n2a_policy_comparison.json",
        mime="application/json",
    )
