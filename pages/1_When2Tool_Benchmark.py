"""Explore the official When2Tool task benchmark from Streamlit."""
import json

import pandas as pd
import streamlit as st

from n2a.benchmarks.when2tool import (
    DATASET_ID,
    DATASET_REVISION,
    UPSTREAM_COMMIT,
    EXPECTED_COUNTS,
    audit_when2tool,
    load_when2tool_split,
)

st.set_page_config(page_title="When2Tool Benchmark", page_icon="🧪", layout="wide")
st.title("When2Tool Benchmark Explorer")
st.caption("Pinned public benchmark data · Model-specific necessity labels are collected separately.")

st.info(
    "The public dataset contains task instructions, reference answers, environment/tool schemas, "
    "and split metadata. It does not contain universal tool-necessity labels: those depend on "
    "the evaluated model and no-tool evaluation protocol."
)

with st.expander("Pinned source versions", expanded=False):
    st.code(f"Hugging Face dataset: {DATASET_ID}\nDataset revision: {DATASET_REVISION}\nOfficial runner commit: {UPSTREAM_COMMIT}\nRepository: https://github.com/Trustworthy-ML-Lab/when2tool")

st.markdown("### Official split sizes")
summary = []
for config, splits in EXPECTED_COUNTS.items():
    summary.append({
        "Configuration": config,
        "Train": splits["train"],
        "Test": splits["test"],
        "Total": sum(splits.values()),
    })
st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

@st.cache_data(show_spinner=False)
def cached_load(config, split):
    return load_when2tool_split(config, split)

left, right = st.columns(2)
with left:
    config = st.selectbox("Dataset configuration", list(EXPECTED_COUNTS), index=0)
with right:
    split = st.selectbox("Split", ["train", "test"], index=1)

if st.button("Load selected split", type="primary"):
    with st.spinner("Downloading the pinned dataset split from Hugging Face..."):
        tasks = cached_load(config, split)
    st.session_state["when2tool_tasks"] = tasks
    st.session_state["when2tool_selection"] = f"{config}/{split}"

if "when2tool_tasks" in st.session_state:
    tasks = st.session_state["when2tool_tasks"]
    st.markdown(f"### Loaded: {st.session_state['when2tool_selection']}")
    env_counts = pd.Series([t["environments"][0]["name"] for t in tasks]).value_counts()
    diff_counts = pd.Series([t["difficulty"] for t in tasks]).value_counts()
    a, b, c = st.columns(3)
    a.metric("Tasks loaded", len(tasks))
    b.metric("Environments", len(env_counts))
    c.metric("Unique task IDs", len({t["id"] for t in tasks}))
    st.markdown("**Difficulty distribution**")
    st.dataframe(diff_counts.rename_axis("Difficulty").rename("Tasks").reset_index(), use_container_width=True, hide_index=True)
    st.markdown("**Environment distribution**")
    st.dataframe(env_counts.rename_axis("Environment").rename("Tasks").reset_index(), use_container_width=True, hide_index=True)
    st.markdown("**Sample tasks**")
    sample = [{
        "id": t["id"],
        "difficulty": t["difficulty"],
        "environment": t["environments"][0]["name"],
        "instruction": t["instruction"],
        "reference_answer": t["expected"]["answer"],
        "tools": ", ".join(t["environments"][0]["tools"]),
    } for t in tasks[:30]]
    st.dataframe(pd.DataFrame(sample), use_container_width=True, hide_index=True)
    st.download_button(
        "Download normalized JSON",
        json.dumps(tasks, ensure_ascii=False, indent=2),
        file_name=f"when2tool_{config}_{split}.json",
        mime="application/json",
    )

st.divider()
if st.button("Audit all four official splits"):
    with st.spinner("Checking row counts, duplicate IDs, and train/test separation..."):
        report = audit_when2tool()
    st.session_state["when2tool_audit"] = report

if "when2tool_audit" in st.session_state:
    report = st.session_state["when2tool_audit"]
    st.success(f"Dataset audit completed: {report['total_tasks']:,} tasks inspected.")
    st.json(report)
