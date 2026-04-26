"""Wire the supervisor + workers into a LangGraph state machine.

Topology: hub-and-spoke. Every worker returns to the supervisor, which then
makes the next decision. This guarantees the supervisor sees every state
transition and keeps loops bounded.
"""
from langgraph.graph import END, START, StateGraph

from agents import (
    critic_node,
    general_qa_node,
    math_solver_node,
    planner_node,
    policy_rag_node,
    product_rag_node,
    writer_node,
)
from state import AgentState
from supervisor import supervisor_node

WORKERS = [
    "planner", "policy_rag", "product_rag",
    "general_qa", "math_solver", "writer", "critic",
]


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("planner", planner_node)
    g.add_node("policy_rag", policy_rag_node)
    g.add_node("product_rag", product_rag_node)
    g.add_node("general_qa", general_qa_node)
    g.add_node("math_solver", math_solver_node)
    g.add_node("writer", writer_node)
    g.add_node("critic", critic_node)

    g.add_edge(START, "supervisor")
    for w in WORKERS:
        g.add_edge(w, "supervisor")

    g.add_conditional_edges(
        "supervisor",
        lambda s: s["next"],
        {**{w: w for w in WORKERS}, "FINISH": END},
    )
    return g.compile()
