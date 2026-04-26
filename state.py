"""Shared state passed between supervisor and worker nodes.

Every field here is a deliberate Supervisor decision surface:
  plan / current_step  -> decomposition + sequencing
  completed_steps      -> per-step grading & provenance for the writer
  draft / grounded     -> quality / hallucination control
  revisions            -> bounded recovery loop
  step_count/max_steps -> resource budget
"""
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

Domain = Literal["hr", "product", "general", "math"]


class SubQuestion(TypedDict):
    question: str
    domain: Domain


class StepResult(TypedDict):
    question: str
    answer: str
    source_agent: str
    citations: list[str]
    grade: str   # 'relevant' | 'partial' | 'irrelevant' | 'n/a'


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]

    # Plan + execution
    plan: list[SubQuestion]
    completed_steps: list[StepResult]
    current_step: int

    # Routing decision recorded by the supervisor each turn
    next: str
    route_reason: str

    # Final composition + critique loop
    draft: str
    grounded: Optional[bool]
    critique: str
    revisions: int

    # Budget guard
    step_count: int
    max_steps: int
