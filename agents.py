"""Worker agents. Each one is a node fn: AgentState -> partial state update.

LLM + embeddings come from `providers.py` so the same agents work with either
local Ollama or any OpenAI-compatible endpoint — choose by editing `config.py`.
"""
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from pydantic import BaseModel, Field

from providers import CHROMA_SETTINGS, PERSIST_DIR, make_embeddings, make_llm
from state import AgentState, StepResult, SubQuestion

llm = make_llm(temperature=0)
embeddings = make_embeddings()


def _retriever(collection: str, k: int = 4):
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
        client_settings=CHROMA_SETTINGS,
    ).as_retriever(search_kwargs={"k": k})


def _user_question(state: AgentState) -> str:
    for m in state.get("messages", []):
        if isinstance(m, HumanMessage):
            return m.content
    return ""


# ======================================================================
# 1. Planner — LLM decomposes the user's query into typed sub-questions
# ======================================================================
class PlannedSubQ(BaseModel):
    question: str
    domain: Literal["hr", "product", "general", "math"]


class Plan(BaseModel):
    """Ordered list of self-contained sub-questions."""
    sub_questions: list[PlannedSubQ] = Field(
        description="Smallest sequence needed to answer. Use a single item if simple."
    )


PLANNER_SYS = (
    "You are a Query Planner. Decompose the user's question into the smallest "
    "ordered list of self-contained sub-questions. Tag each with a domain so a "
    "specialist can handle it:\n"
    "  hr      -> ACME HR / leave / benefits / remote work policy\n"
    "  product -> ACME RoboVac product specs and troubleshooting\n"
    "  general -> general world knowledge (no internal docs)\n"
    "  math    -> arithmetic computed from prior sub-answers\n"
    "Rules:\n"
    "- Math sub-questions MUST come AFTER the sub-questions that supply the numbers.\n"
    "- If the user's question is simple, return one sub-question."
)


def planner_node(state: AgentState) -> dict:
    user_q = _user_question(state)
    try:
        plan: Plan = llm.with_structured_output(Plan).invoke([
            SystemMessage(content=PLANNER_SYS),
            HumanMessage(content=user_q),
        ])
        sub_qs: list[SubQuestion] = [
            {"question": s.question, "domain": s.domain} for s in plan.sub_questions
        ]
    except Exception:
        sub_qs = [{"question": user_q, "domain": "general"}]

    return {
        "plan": sub_qs,
        "current_step": 0,
        "completed_steps": [],
        "step_count": state.get("step_count", 0) + 1,
        "messages": [AIMessage(content=f"[planner] {sub_qs}", name="planner")],
    }


# ======================================================================
# 2. RAG agents — domain-specific retrieval + per-call relevance grading
# ======================================================================
class RelevanceGrade(BaseModel):
    grade: Literal["relevant", "partial", "irrelevant"]
    reason: str


GRADER_SYS = (
    "Decide whether the context is sufficient to answer the question. "
    "Return 'relevant' if it fully covers it, 'partial' if it covers some of it, "
    "'irrelevant' if it does not address the question at all."
)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are the {domain} specialist. Answer ONLY from the context. "
     "If the context is insufficient, say so plainly. "
     "Cite source filenames in square brackets when you use them.\n\n"
     "<context>\n{context}\n</context>"),
    ("human", "{question}"),
])


def _make_rag_node(domain_label: str, collection: str, agent_name: str):
    retriever = _retriever(collection)

    def node(state: AgentState) -> dict:
        idx = state["current_step"]
        sub = state["plan"][idx]
        question = sub["question"]

        docs = retriever.invoke(question)
        context = "\n\n---\n\n".join(
            f"[source: {Path(d.metadata.get('source', 'unknown')).name}]\n{d.page_content}"
            for d in docs
        ) or "(no documents retrieved)"

        # Grade BEFORE answering. If irrelevant, skip the generation cost.
        try:
            grade: RelevanceGrade = llm.with_structured_output(RelevanceGrade).invoke([
                SystemMessage(content=GRADER_SYS),
                HumanMessage(content=f"Question: {question}\n\nContext:\n{context}"),
            ])
            grade_label = grade.grade
        except Exception:
            grade_label = "partial"

        if grade_label == "irrelevant":
            answer = f"No relevant {domain_label} documents found for: {question}"
            citations: list[str] = []
        else:
            resp = (RAG_PROMPT | llm).invoke({
                "domain": domain_label, "context": context, "question": question,
            })
            answer = resp.content
            citations = sorted({
                Path(d.metadata.get("source", "")).name for d in docs if d.metadata.get("source")
            })

        step: StepResult = {
            "question": question, "answer": answer,
            "source_agent": agent_name, "citations": citations,
            "grade": grade_label,
        }
        return {
            "completed_steps": state.get("completed_steps", []) + [step],
            "current_step": idx + 1,
            "step_count": state.get("step_count", 0) + 1,
            "messages": [AIMessage(
                content=f"[{agent_name}] grade={grade_label} :: {answer}",
                name=agent_name,
            )],
        }

    return node


policy_rag_node = _make_rag_node("HR policy", "hr_kb", "policy_rag")
product_rag_node = _make_rag_node("Product", "product_kb", "product_rag")


# ======================================================================
# 3. General-knowledge agent (no retrieval)
# ======================================================================
def general_qa_node(state: AgentState) -> dict:
    idx = state["current_step"]
    sub = state["plan"][idx]
    question = sub["question"]
    out = llm.invoke([
        SystemMessage(content=(
            "You are a general-knowledge agent. Answer concisely from world knowledge. "
            "Never invent ACME-internal facts."
        )),
        HumanMessage(content=question),
    ])
    step: StepResult = {
        "question": question, "answer": out.content,
        "source_agent": "general_qa", "citations": [], "grade": "n/a",
    }
    return {
        "completed_steps": state.get("completed_steps", []) + [step],
        "current_step": idx + 1,
        "step_count": state.get("step_count", 0) + 1,
        "messages": [AIMessage(content=f"[general_qa] {out.content}", name="general_qa")],
    }


# ======================================================================
# 4. Math solver — extracts a pure arithmetic expression from prior facts
#    and evaluates it in a sandboxed eval (whitelisted character set)
# ======================================================================
class MathPlan(BaseModel):
    expression: str = Field(
        description="A pure Python arithmetic expression. "
                    "Use only digits, + - * / ( ) and . — no variables, no functions."
    )
    explanation: str


_SAFE_CHARS = set("0123456789+-*/(). ")


def math_solver_node(state: AgentState) -> dict:
    idx = state["current_step"]
    sub = state["plan"][idx]
    question = sub["question"]
    prior_facts = "\n".join(
        f"- Q: {s['question']}\n  A: {s['answer']}"
        for s in state.get("completed_steps", [])
    ) or "(no prior facts)"

    try:
        plan: MathPlan = llm.with_structured_output(MathPlan).invoke([
            SystemMessage(content=(
                "You are a math agent. Read prior facts, extract the relevant numbers, "
                "and produce ONE pure arithmetic expression that answers the question."
            )),
            HumanMessage(content=f"Question: {question}\n\nPrior facts:\n{prior_facts}"),
        ])
        expr = "".join(c for c in plan.expression if c in _SAFE_CHARS)
        value = eval(expr, {"__builtins__": {}}, {})  # safe: chars whitelisted above
        answer = f"{plan.explanation} => {expr} = {value}"
    except Exception as e:
        answer = f"Math step failed: {e}"

    step: StepResult = {
        "question": question, "answer": answer,
        "source_agent": "math_solver", "citations": [], "grade": "n/a",
    }
    return {
        "completed_steps": state.get("completed_steps", []) + [step],
        "current_step": idx + 1,
        "step_count": state.get("step_count", 0) + 1,
        "messages": [AIMessage(content=f"[math_solver] {answer}", name="math_solver")],
    }


# ======================================================================
# 5. Writer — composes the final user-facing answer from completed_steps
# ======================================================================
def writer_node(state: AgentState) -> dict:
    user_q = _user_question(state)
    facts = "\n".join(
        f"- ({s['source_agent']}, grade={s['grade']}) Q: {s['question']}\n"
        f"  A: {s['answer']}\n  Citations: {', '.join(s['citations']) or '—'}"
        for s in state.get("completed_steps", [])
    ) or "(none)"
    critique = state.get("critique", "")
    revision_note = (
        f"\n\nThe previous draft was rejected by the critic for: {critique}. "
        "Rewrite to fix those issues."
        if critique and critique != "ok" else ""
    )

    out = llm.invoke([
        SystemMessage(content=(
            "You are the Writer. Compose a concise, well-structured final answer. "
            "Cite source filenames in square brackets when you rely on retrieved facts. "
            "If a sub-step has grade=irrelevant, acknowledge the gap rather than inventing."
        )),
        HumanMessage(content=(
            f"User question: {user_q}\n\nGathered facts:\n{facts}{revision_note}"
        )),
    ])
    return {
        "draft": out.content,
        "grounded": None,  # reset so the critic re-evaluates the new draft
        "revisions": state.get("revisions", 0) + 1,
        "step_count": state.get("step_count", 0) + 1,
        "messages": [AIMessage(content=out.content, name="writer")],
    }


# ======================================================================
# 6. Critic — checks the draft for grounding against gathered facts
# ======================================================================
class Critique(BaseModel):
    grounded: bool = Field(
        description="True iff every claim in the draft is supported by the gathered facts."
    )
    issues: list[str]


def critic_node(state: AgentState) -> dict:
    facts = "\n".join(
        f"- {s['answer']}  [citations: {', '.join(s['citations']) or '—'}]"
        for s in state.get("completed_steps", [])
    )
    try:
        crit: Critique = llm.with_structured_output(Critique).invoke([
            SystemMessage(content=(
                "You are the Critic. Mark grounded=False if the draft introduces "
                "information not present in the facts, fails to cite where it should, "
                "or contradicts a fact. Otherwise grounded=True."
            )),
            HumanMessage(content=(
                f"Draft:\n{state.get('draft','')}\n\nGathered facts:\n{facts}"
            )),
        ])
        grounded = crit.grounded
        issues = crit.issues
    except Exception:
        grounded, issues = True, []   # fail-open so a flaky critic can't block forever

    return {
        "grounded": grounded,
        "critique": "; ".join(issues) if issues else "ok",
        "step_count": state.get("step_count", 0) + 1,
        "messages": [AIMessage(
            content=f"[critic] grounded={grounded} issues={issues}",
            name="critic",
        )],
    }
