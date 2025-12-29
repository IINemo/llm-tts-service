"""
LangGraph Uncertainty-Aware Routing Example.

Demonstrates how to use ChatTTS with LangGraph for uncertainty-based routing:
- If uncertainty is low -> accept answer
- If uncertainty is high -> retry with higher budget or escalate

Requirements:
    pip install langgraph langchain-core

    # Start the TTS service
    cd service_app && docker compose up -d

Usage:
    python examples/langgraph_uncertainty_routing.py
"""

from typing import Annotated, Literal, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

from llm_tts.integrations import ChatTTS


# =============================================================================
# State Definition
# =============================================================================

class AgentState(TypedDict):
    """State for uncertainty-aware routing."""
    question: str
    answer: str
    content: str
    confidence: float
    uncertainty: float
    budget: int
    attempts: int
    status: Literal["pending", "accepted", "escalated"]


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "service_url": "http://localhost:8001/v1",
    "model": "openai/gpt-4o-mini",
    "initial_budget": 3,
    "max_budget": 12,
    "uncertainty_threshold": 0.3,
    "max_attempts": 3,
}


# =============================================================================
# Graph Nodes
# =============================================================================

def call_tts_node(state: AgentState) -> AgentState:
    """Call ChatTTS and extract uncertainty metrics."""

    llm = ChatTTS(
        base_url=CONFIG["service_url"],
        model=CONFIG["model"],
        tts_strategy="self_consistency",
        tts_budget=state["budget"],
        temperature=0.7,
        max_tokens=512,
    )

    messages = [
        SystemMessage(content="Solve step by step. Put your final answer in \\boxed{}."),
        HumanMessage(content=state["question"]),
    ]

    response = llm.invoke(messages)

    # Extract TTS metadata
    meta = response.response_metadata.get("tts_metadata", {})

    return {
        **state,
        "answer": meta.get("selected_answer", "no_answer"),
        "content": response.content,
        "confidence": meta.get("consensus_score", 0.0),
        "uncertainty": meta.get("uncertainty_score", 1.0),
        "attempts": state["attempts"] + 1,
    }


def increase_budget_node(state: AgentState) -> AgentState:
    """Double the budget for retry."""
    new_budget = min(state["budget"] * 2, CONFIG["max_budget"])
    return {**state, "budget": new_budget}


def accept_node(state: AgentState) -> AgentState:
    """Mark answer as accepted."""
    return {**state, "status": "accepted"}


def escalate_node(state: AgentState) -> AgentState:
    """Mark for human review."""
    return {**state, "status": "escalated"}


# =============================================================================
# Routing Logic
# =============================================================================

def route_by_uncertainty(state: AgentState) -> Literal["accept", "retry", "escalate"]:
    """Route based on uncertainty and attempt count."""

    uncertainty = state["uncertainty"]
    attempts = state["attempts"]
    budget = state["budget"]

    # Low uncertainty -> accept
    if uncertainty <= CONFIG["uncertainty_threshold"]:
        return "accept"

    # Can we retry with higher budget?
    if attempts < CONFIG["max_attempts"] and budget < CONFIG["max_budget"]:
        return "retry"

    # Too many attempts or max budget reached -> escalate
    return "escalate"


# =============================================================================
# Build Graph
# =============================================================================

def build_uncertainty_graph() -> StateGraph:
    """Build LangGraph with uncertainty-based routing."""

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("call_tts", call_tts_node)
    graph.add_node("increase_budget", increase_budget_node)
    graph.add_node("accept", accept_node)
    graph.add_node("escalate", escalate_node)

    # Entry point
    graph.set_entry_point("call_tts")

    # Conditional routing after TTS call
    graph.add_conditional_edges(
        "call_tts",
        route_by_uncertainty,
        {
            "accept": "accept",
            "retry": "increase_budget",
            "escalate": "escalate",
        }
    )

    # Retry loops back to TTS
    graph.add_edge("increase_budget", "call_tts")

    # Terminal nodes
    graph.add_edge("accept", END)
    graph.add_edge("escalate", END)

    return graph.compile()


# =============================================================================
# Main
# =============================================================================

def solve_with_uncertainty_routing(question: str) -> AgentState:
    """Solve a question with uncertainty-aware routing."""

    graph = build_uncertainty_graph()

    initial_state: AgentState = {
        "question": question,
        "answer": "",
        "content": "",
        "confidence": 0.0,
        "uncertainty": 1.0,
        "budget": CONFIG["initial_budget"],
        "attempts": 0,
        "status": "pending",
    }

    return graph.invoke(initial_state)


def main():
    """Demo the uncertainty-aware routing."""

    print("=" * 60)
    print("LangGraph Uncertainty-Aware Routing Demo")
    print("=" * 60)

    questions = [
        "What is 7 * 8?",
        "What is 15% of 240?",
        "A train travels at 60 mph for 2.5 hours. How far does it travel?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 40)

        result = solve_with_uncertainty_routing(question)

        print(f"Status: {result['status'].upper()}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Uncertainty: {result['uncertainty']:.2f}")
        print(f"Attempts: {result['attempts']}")
        print(f"Final budget: {result['budget']}")


if __name__ == "__main__":
    main()
