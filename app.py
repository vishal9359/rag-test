"""Entry point: stream the supervisor's decisions and worker outputs.

Run:
    1. ollama serve
    2. ollama pull llama3.1:8b && ollama pull nomic-embed-text
    3. pip install -r requirements.txt
    4. python ingest.py     # one-time, builds ./chroma_db
    5. python app.py
"""
import _silence_chroma  # noqa: F401  — MUST be the first import (silences telemetry)

from langchain_core.messages import HumanMessage

from graph import build_graph


def initial_state(question: str, max_steps: int = 20) -> dict:
    return {
        "messages": [HumanMessage(content=question)],
        "plan": [],
        "completed_steps": [],
        "current_step": 0,
        "next": "",
        "route_reason": "",
        "draft": "",
        "grounded": None,
        "critique": "",
        "revisions": 0,
        "step_count": 0,
        "max_steps": max_steps,
    }


def _truncate(s: str, n: int = 240) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


def run(question: str, verbose: bool = True) -> str:
    graph = build_graph()
    state = initial_state(question)
    last_state: dict = {}

    for chunk in graph.stream(state, config={"recursion_limit": 60}):
        for node, payload in chunk.items():
            last_state = payload
            if not verbose:
                continue
            if node == "supervisor":
                print(f"  → supervisor: next={payload.get('next')} "
                      f"({payload.get('route_reason','')})")
            else:
                msgs = payload.get("messages") or []
                if msgs:
                    print(f"  • {node}: {_truncate(msgs[-1].content)}")

    return last_state.get("draft") or "(no draft produced)"


if __name__ == "__main__":
    examples = [
        # Multi-hop: HR retrieval + math
        "I joined ACME 8 months ago and have already taken 5 days off. "
        "How many days of annual leave do I have left this year?",

        # Product RAG + general knowledge fusion
        "What's the suction power of the RoboVac X9, and how does 4000 Pa "
        "compare to typical robot vacuums in 2024?",

        # Pure product troubleshooting
        "My RoboVac shows error E03. What should I do?",

        # Pure general — should NOT touch the KB
        "In one sentence, what is retrieval-augmented generation?",
    ]
    for q in examples:
        print(f"\n=== Q: {q}")
        final = run(q)
        print(f"\n--- FINAL ---\n{final}\n")
