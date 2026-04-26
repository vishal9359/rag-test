"""The Supervisor.

Mostly DETERMINISTIC. Real production supervisors lean on rules for plumbing
(routing by domain, advancing the plan, enforcing budget) and only spend LLM
calls on judgment-heavy steps. The judgment-heavy steps in this graph are
delegated to dedicated agents (planner, grader, critic) — so the supervisor
itself can stay rule-based, fast, and predictable.

Decisions handled here:
  - Decomposition: route to planner if no plan yet
  - Routing:       map next sub-question's domain -> specialist worker
  - Composition:   route to writer once the plan is fully executed
  - Quality:       route to critic after writer; loop back if not grounded
  - Recovery:      bounded revision count to prevent oscillation
  - Termination:   FINISH on grounded draft, FAIL-soft on budget exhaustion
"""
from state import AgentState

DOMAIN_TO_WORKER = {
    "hr": "policy_rag",
    "product": "product_rag",
    "general": "general_qa",
    "math": "math_solver",
}

MAX_REVISIONS = 1   # writer may be re-invoked once after a critic rejection


def supervisor_node(state: AgentState) -> dict:
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 20)

    # 6. Termination — budget guard. Always wins.
    if step_count >= max_steps:
        return {"next": "FINISH", "route_reason": f"budget exhausted ({step_count}/{max_steps})"}

    # 1. Decomposition — no plan yet?
    plan = state.get("plan", [])
    if not plan:
        return {"next": "planner", "route_reason": "no plan yet — decompose query"}

    # 2. Routing — plan still in progress?
    current_step = state.get("current_step", 0)
    if current_step < len(plan):
        sub = plan[current_step]
        worker = DOMAIN_TO_WORKER.get(sub["domain"], "general_qa")
        return {
            "next": worker,
            "route_reason": f"step {current_step+1}/{len(plan)} domain={sub['domain']}",
        }

    # 3. Composition — all sub-questions answered, no draft yet
    if not state.get("draft"):
        return {"next": "writer", "route_reason": "plan complete — compose draft"}

    # 4. Quality — draft exists but hasn't been critiqued
    grounded = state.get("grounded")
    if grounded is None:
        return {"next": "critic", "route_reason": "draft needs grounding check"}

    # 5. Recovery — critic rejected; allow ONE revision pass
    revisions = state.get("revisions", 0)
    if grounded is False and revisions <= MAX_REVISIONS:
        return {"next": "writer", "route_reason": f"critic rejected — revision {revisions}"}

    # Done — either grounded, or we exhausted the revision budget
    return {
        "next": "FINISH",
        "route_reason": "grounded" if grounded else "revision budget exhausted",
    }
