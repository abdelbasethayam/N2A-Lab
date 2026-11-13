"""Transparent synthetic scenarios for the N2A teaching sandbox.

These fixtures are NOT a research benchmark. Replace them with official benchmark
items and independently collected model rollouts before making empirical claims.
"""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    title: str
    question: str
    reference_answer: str
    no_tool_answer: str
    tool_name: str
    tool_description: str
    p_no_tool_success: float
    p_with_tool_success: float
    expected_necessity: bool
    rationale: str
    tool_fn: Callable[[], str]


def _calculator() -> str:
    return str(17 * 19)


def _revenue_lookup() -> str:
    revenue_2024, revenue_2025 = 8.0, 10.0
    growth = (revenue_2025 - revenue_2024) / revenue_2024 * 100
    return f"{growth:.1f}%"


def _sales_lookup_and_calculate() -> str:
    q1_units, q2_units = 1200, 1500
    increase = (q2_units - q1_units) / q1_units * 100
    return f"{increase:.1f}%"


def _mock_search() -> str:
    local_index = {"capital_of_france": "Paris"}
    return local_index["capital_of_france"]


SCENARIOS = {
    "arithmetic": Scenario(
        scenario_id="arithmetic",
        title="Simple arithmetic",
        question="Calculate 17 × 19.",
        reference_answer="323",
        no_tool_answer="323",
        tool_name="calculator",
        tool_description="A local calculator tool.",
        p_no_tool_success=0.98,
        p_with_tool_success=0.995,
        expected_necessity=False,
        rationale="The task is readily solvable from the prompt. A calculator can verify the answer but adds little expected value.",
        tool_fn=_calculator,
    ),
    "private-records": Scenario(
        scenario_id="private-records",
        title="Private record lookup",
        question="Revenue was $8.0M in 2024 and $10.0M in 2025. What was year-over-year growth?",
        reference_answer="25.0%",
        no_tool_answer="I cannot verify the figures without consulting the records.",
        tool_name="company_facts_lookup",
        tool_description="A read-only lookup over a fixed local example record.",
        p_no_tool_success=0.12,
        p_with_tool_success=0.98,
        expected_necessity=True,
        rationale="This task-specific lookup requires retrieving the supplied record before calculating growth.",
        tool_fn=_revenue_lookup,
    ),
    "multi-step": Scenario(
        scenario_id="multi-step",
        title="Multi-step retrieval and calculation",
        question="The dataset says 1,200 units were sold in Q1 and 1,500 in Q2. What was the percentage increase?",
        reference_answer="25.0%",
        no_tool_answer="I do not have the dataset values available in the task context.",
        tool_name="sales_lookup_and_calculator",
        tool_description="A local lookup followed by a deterministic calculation.",
        p_no_tool_success=0.10,
        p_with_tool_success=0.96,
        expected_necessity=True,
        rationale="The example requires retrieving task-specific values and then calculating the change.",
        tool_fn=_sales_lookup_and_calculate,
    ),
    "stable-knowledge": Scenario(
        scenario_id="stable-knowledge",
        title="Stable general knowledge",
        question="What is the capital of France?",
        reference_answer="Paris",
        no_tool_answer="Paris",
        tool_name="mock_search",
        tool_description="A read-only local mock search index.",
        p_no_tool_success=0.99,
        p_with_tool_success=0.995,
        expected_necessity=False,
        rationale="This is stable, common knowledge. Search can confirm it, but the expected incremental gain is small.",
        tool_fn=_mock_search,
    ),
}


def get_scenario(scenario_id: str) -> Scenario:
    """Return a scenario by stable ID."""
    try:
        return SCENARIOS[scenario_id]
    except KeyError as exc:
        raise ValueError(f"Unknown scenario_id: {scenario_id}") from exc
