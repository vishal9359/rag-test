# multiagent_rag

A teachable, production-shaped **multi-agent RAG system with a Supervisor**, built on **LangGraph** and **local Ollama models** (no cloud calls). Designed for running on a GPU server.

The goal is to make every *type* of supervisor decision visible in code — routing, decomposition, retrieval grading, draft critique, bounded revision, and budget enforcement.

---

## Architecture

```
                         +-------------+
   user query ---------->|  Supervisor |   deterministic controller
                         +------+------+
                                |
       +----------+--------+----+----+--------+----------+----------+
       |          |        |        |         |          |          |
       v          v        v        v         v          v          v
   +-------+  +--------+ +--------+ +-------+ +------+ +--------+ +------+
   |planner|  |policy_ | |product_| |general| | math_| | writer | |critic|
   |       |  |  rag   | |  rag   | |  qa   | |solver| |        | |      |
   +-------+  +--------+ +--------+ +-------+ +------+ +--------+ +------+
                  |          |
                  +-> per-call relevance grading (relevant|partial|irrelevant)

   Every worker returns to the supervisor (hub-and-spoke).
   The supervisor decides: route / compose / critique / revise / FINISH.
```

### Topology in one sentence

The supervisor is a **rule-based controller**; the LLM-driven judgment lives in dedicated agents (`planner`, the per-RAG `grader`, `critic`). This is how real production supervisors are split — mechanical decisions stay cheap and predictable, judgment lives where it's testable.

---

## Worker roles

| Agent | Role | LLM-driven? |
|---|---|---|
| `planner` | Decomposes the user query into a typed list of `SubQuestion(question, domain)` | ✓ structured output |
| `policy_rag` | Retrieves from the **HR** Chroma collection, grades, then answers | ✓ retrieval grader + answer |
| `product_rag` | Retrieves from the **Product** Chroma collection, grades, then answers | ✓ retrieval grader + answer |
| `general_qa` | Answers from the LLM's general knowledge — never touches the KB | ✓ |
| `math_solver` | Extracts a pure arithmetic expression from prior facts and evaluates it in a sandboxed `eval` (whitelisted character set) | ✓ structured output → safe eval |
| `writer` | Composes the final user-facing answer using `completed_steps` and any critic feedback | ✓ |
| `critic` | Grades the writer's draft for grounding (claims supported by the gathered facts) | ✓ structured output |

---

## Supervisor decisions

| Decision | Implementation |
|---|---|
| **Decomposition** — does the query need a plan? | `if not plan → planner` |
| **Routing** — which specialist? | `DOMAIN_TO_WORKER[sub.domain]` |
| **Composition** — when to write the final answer? | All sub-questions answered & no draft → `writer` |
| **Quality** — is the draft grounded? | Draft exists & `grounded is None` → `critic` |
| **Recovery** — what if the critic rejects? | `grounded is False` and `revisions ≤ MAX_REVISIONS` → back to `writer` |
| **Termination** — when to stop? | `FINISH` on grounded draft, or budget exhaustion |
| **Budget** — guard against runaway loops | `step_count >= max_steps` always wins |

The supervisor itself has **zero LLM calls** — see [supervisor.py](supervisor.py).

---

## File layout

```
multiagent_rag/
├── requirements.txt
├── ingest.py              # one-time: builds two Chroma collections
├── state.py               # AgentState (plan, draft, grounded, budget...)
├── agents.py              # all 7 worker nodes
├── supervisor.py          # deterministic controller
├── graph.py               # hub-and-spoke wiring
├── app.py                 # streaming entrypoint w/ example queries
└── sample_docs/
    ├── hr/
    │   ├── company_policy.txt
    │   └── leave_calculation_rules.txt
    └── product/
        ├── product_specs.txt
        └── troubleshooting.txt
```

---

## Local model setup (Ollama)

Every model is local. Default choices:

| Purpose | Model | Pull |
|---|---|---|
| Chat / structured output | `llama3.1:8b` | `ollama pull llama3.1:8b` |
| Embeddings | `nomic-embed-text` | `ollama pull nomic-embed-text` |

Both are configured at the top of [agents.py](agents.py) — change `LLM_MODEL` to `qwen2.5:14b` or `llama3.1:70b` if your GPU has the headroom. If Ollama runs on a different host than Python, change `OLLAMA_BASE_URL`.

> Structured output (`with_structured_output`) needs a tool-calling model. `llama3.1`, `qwen2.5`, and `mistral` all support it.

---

## Quickstart

```bash
# 1. Start Ollama and pull models
ollama serve
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 2. Install Python deps (Python 3.10+)
pip install -r requirements.txt

# 3. Build the vector store (one-time)
python ingest.py

# 4. Run the example queries
python app.py
```

Output streams every supervisor decision and every worker's reply, then prints the FINAL answer.

---

## Example queries baked into `app.py`

| Query | What it exercises |
|---|---|
| *"I joined ACME 8 months ago and have already taken 5 days off. How many days of annual leave do I have left this year?"* | planner → 2× `policy_rag` → `math_solver` → writer → critic |
| *"Suction power of the RoboVac X9, and how does 4000 Pa compare to typical robot vacuums in 2024?"* | planner → `product_rag` + `general_qa` (mixed domain fusion) |
| *"My RoboVac shows error E03. What should I do?"* | single-step `product_rag` |
| *"In one sentence, what is retrieval-augmented generation?"* | single-step `general_qa` (KB never queried) |

---

## How a multi-hop query flows

```
Q: "I joined 8 months ago, took 5 days off — leave remaining?"

supervisor: next=planner            (no plan yet)
planner:    [{q: "annual leave entitlement",   domain: hr},
             {q: "pro-rating rule for partial year", domain: hr},
             {q: "(8/12)*24 - 5", domain: math}]

supervisor: next=policy_rag         (step 1/3, hr)
policy_rag: grade=relevant -> "24 days/year [company_policy.txt]"

supervisor: next=policy_rag         (step 2/3, hr)
policy_rag: grade=relevant -> "(months/12)*24 rounded down [leave_calculation_rules.txt]"

supervisor: next=math_solver        (step 3/3, math)
math_solver: (8/12)*24 - 5 = 11.0

supervisor: next=writer             (plan complete)
writer:     draft with citations

supervisor: next=critic             (grounded is None)
critic:     grounded=True

supervisor: next=FINISH
```

---

## Extending it

Real-world supervisors tend to grow these capabilities — the current structure is set up to absorb each one cleanly:

| Extension | Where it plugs in |
|---|---|
| **Parallel fan-out** for independent sub-questions | Use LangGraph's `Send` API in the supervisor's conditional edges |
| **Clarification loop** when planner can't pick a domain | New `clarify_node`; supervisor routes when `domain == "unknown"` |
| **Cost-aware routing** (cheap model first, escalate) | Two `ChatOllama` instances; supervisor picks based on `grade=partial` |
| **Long-term memory** | Add another Chroma collection; supervisor decides when to query it |
| **Human-in-the-loop escalation** | Terminal `escalate` node when `step_count >= max_steps` and `grounded is False` |
| **Observability** | LangSmith trace, or just persist `state["completed_steps"]` to disk per run |

---

## Why split the supervisor from the agents?

A common beginner shape is a single LLM that "decides what to do next" on every step. It works for demos but in production it:

- Burns tokens on routing decisions that are essentially `if/elif`
- Oscillates between workers when the prompt isn't perfect
- Is hard to unit-test (you can't assert on rule outcomes)

The split here gives you:

- **Predictable, testable plumbing** in the supervisor
- **LLM intelligence concentrated** in narrow, well-prompted workers (planner, grader, critic)
- **Clean budget & recovery semantics** because state transitions are rule-driven

---

## Configuration knobs

| Knob | File | Default |
|---|---|---|
| LLM model | [agents.py](agents.py) | `llama3.1:8b` |
| Embedding model | [agents.py](agents.py), [ingest.py](ingest.py) | `nomic-embed-text` |
| Ollama URL | [agents.py](agents.py) | `http://localhost:11434` |
| Chunk size / overlap | [ingest.py](ingest.py) | `600 / 80` |
| Top-k retrieval | [agents.py `_retriever`](agents.py) | `4` |
| Max revisions after critic | [supervisor.py](supervisor.py) | `1` |
| Step budget | [app.py `initial_state`](app.py) | `20` |
| LangGraph recursion limit | [app.py `run`](app.py) | `60` |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `with_structured_output` returns gibberish or raises | Model doesn't support tool calling | Switch to `llama3.1:8b` / `qwen2.5` / `mistral` |
| Empty retrieval results | `python ingest.py` not run, or wrong collection name | Rebuild: `rm -rf chroma_db && python ingest.py` |
| Hits recursion limit | Loop between writer↔critic | Increase `MAX_REVISIONS` cautiously, or lower `max_steps` to fail fast |
| `Connection refused` to 11434 | Ollama not running | `ollama serve` |
| Slow first call | Ollama loading the model into VRAM | Subsequent calls are fast; pre-warm with `ollama run llama3.1:8b ""` |

---

## Requirements

See [requirements.txt](requirements.txt). Core deps: `langchain`, `langchain-ollama`, `langchain-chroma`, `langgraph`, `chromadb`, `pydantic`.
