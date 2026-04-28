# multiagent_rag

A reference implementation of a **multi-agent Retrieval-Augmented Generation (RAG) system with a Supervisor pattern**, built on **LangGraph** with a pluggable model backend — works with **local Ollama** (default) or any **OpenAI-compatible API** (real OpenAI, Azure OpenAI, vLLM, internal gateway). Structured so the architecture maps 1:1 to real-world enterprise AI assistants like Glean, Intercom Fin, Harvey, and customer-support copilots.

---

## What this project does (in 30 seconds)

Given a user's natural-language question, the system:

1. **Plans** — decomposes the question into a typed list of sub-questions (HR / Product / General / Math)
2. **Routes** — sends each sub-question to the right specialist agent
3. **Retrieves** — pulls from domain-specific Chroma knowledge bases (one per domain)
4. **Grades** — checks each retrieval for relevance *before* generating an answer (skips generation if irrelevant)
5. **Composes** — writes the final answer with citations, using all the gathered facts
6. **Critiques** — checks the draft for grounding (no hallucinated claims)
7. **Revises once** if the critic rejects, then **finishes**

A **rule-based Supervisor** orchestrates all of this. It makes routing, budget, and termination decisions without spending LLM tokens — the LLM intelligence lives in dedicated workers (planner, grader, critic).

> 📖 The goal is to make every *type* of supervisor decision visible in code — routing, decomposition, retrieval grading, draft critique, bounded revision, and budget enforcement — so you can lift the pattern into your own product.

---

## What you can do with it (out of the box)

The shipped sample knowledge base contains fictional **ACME Corp** data:

- **HR policies** — leave entitlements, remote work, reimbursements, pro-rating rules
- **Product specs & troubleshooting** — for the fictional ACME RoboVac X9

You can ask things like:

| Question | What the system does |
|---|---|
| *"How many days of paid leave do I get? If I joined 8 months ago and took 5 days, how many remain?"* | planner → 2× HR-RAG → math solver → writer → critic |
| *"Suction power of the RoboVac X9, and how does 4000 Pa compare to typical robot vacuums?"* | planner → product-RAG + general-knowledge fusion → writer |
| *"My RoboVac shows error E03. What should I do?"* | product-RAG (single hop) → writer |
| *"Explain RAG in one sentence."* | general-knowledge only — the KB is **not** touched (that's an automatic supervisor decision) |

Run `python app.py` and you'll see every supervisor decision and every worker's output streamed live.

---

## What real-world problems this pattern solves

The exact "Supervisor + domain-specialist RAG agents + critic" topology is in production at scale across:

| Domain | Real product example | Maps to our agents |
|---|---|---|
| Enterprise IT/HR helpdesk | Glean, Moveworks, MS Copilot for M365 | `policy_rag` + `product_rag` + math/lookup |
| Customer support automation | Intercom Fin, Zendesk AI, Salesforce Agentforce | Specialist RAG per product line + critic |
| Field service technical support | ServiceNow Now Assist, Aquant | Manuals + troubleshooting + warranty (the RoboVac shape) |
| Legal & compliance research | Harvey, Hebbia, Robin AI | Internal-contracts RAG + statute RAG + math |
| Healthcare clinical decision support | Abridge, Hippocratic AI | Patient-record RAG + drug-interaction DB + guidelines + critic |
| Financial advisory | Bloomberg GPT, Morgan Stanley's internal assistant | Account RAG + market data + regulation RAG + math |
| Software engineering assistants | Cursor, Sourcegraph Cody, Cognition Devin | Codebase RAG + docs RAG + logs + planner |

**Why this pattern keeps appearing**: real organizations have multiple, separately-governed knowledge bases. A single mega-RAG over all of them retrieves worse than domain-specialized retrievers behind a router. That's the supervisor's job.

---

## How to make it solve YOUR problem

Swap the corpus and the tools — keep the architecture.

1. **Drop your real docs** into `sample_docs/<your_domain>/` (PDF, txt, etc.)
2. **Add a domain** in [ingest.py](ingest.py) (`DOMAINS` dict) and [agents.py](agents.py) (one new `_make_rag_node` call)
3. **Update the planner prompt** so it knows about your new domain
4. **Plug in real APIs** as additional agents — CRM lookup, ticket fetch, SQL query, calculator
5. **(Optional) Add a `safety` agent** for PII / refusal / out-of-scope detection

The Supervisor, planner, critic, and bounded-revision logic are domain-agnostic — they work unchanged.

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
├── config.py              # ★ EDIT THIS — pick provider, models, URLs, API key
├── providers.py           # LLM + embeddings factory (reads config.py)
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

## Model backend (Ollama or OpenAI) — one config file

Everything model-related is in [config.py](config.py). No environment variables. Open the file, change one line, save.

| Provider | When to use | LLM default | Embeddings default |
|---|---|---|---|
| `ollama` *(default)* | GPU server with local models, no API cost, full data privacy | `llama3.1:8b` | `nomic-embed-text` |
| `openai` | GPU unavailable, or any OpenAI-compatible endpoint (real OpenAI, Azure OpenAI, vLLM, internal company gateway) | `gpt-4o-mini` | `text-embedding-3-small` |

> **Important — embedding dimensions differ between providers** (Ollama `nomic-embed-text` = 768d, OpenAI `text-embedding-3-small` = 1536d). The Chroma persist directory is namespaced per provider (`chroma_db_ollama/`, `chroma_db_openai/`) so they coexist without collision. **Run `python ingest.py` once after switching providers.**

### Option A — Ollama (local, default)

In `config.py`:

```python
PROVIDER = "ollama"
OLLAMA_BASE_URL    = "http://localhost:11434"   # or your remote GPU host
OLLAMA_LLM_MODEL   = "llama3.1:8b"              # or qwen2.5:14b, etc.
OLLAMA_EMBED_MODEL = "nomic-embed-text"
```

Then in your shell:

```bash
ollama serve
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

> Structured output (`with_structured_output`) needs a tool-calling model. `llama3.1`, `qwen2.5`, and `mistral` all support it.

### Option B — OpenAI / OpenAI-compatible API

In `config.py`:

```python
PROVIDER = "openai"

OPENAI_BASE_URL        = "http://localhost:11434"        # or your internal gateway
OPENAI_LLM_MODEL       = "qwen2.5-coder:14b"             # or gpt-4o-mini, etc.
OPENAI_EMBED_MODEL     = "nomic-embed-text"              # whatever your gateway exposes
OPENAI_TIMEOUT_SECONDS = 600

# Either set OPENAI_API_KEY = "sk-..." for real OpenAI, OR leave it empty
# and put your auth in OPENAI_CUSTOM_HEADERS (typical for internal gateways).
OPENAI_API_KEY = ""

OPENAI_CUSTOM_HEADERS = {
    "x-dep-ticket":     "credential:<paste-your-credential>",
    "User-Type":        "AD_ID",
    "User-Id":          "<your-AD-id>",
    "Send-System-Name": "rvc-api-module",
}
```

How the OpenAI options are forwarded:

| `config.py` field | Forwarded to |
|---|---|
| `OPENAI_BASE_URL` | `ChatOpenAI(base_url=...)` and `OpenAIEmbeddings(base_url=...)` |
| `OPENAI_API_KEY` | `api_key=...` (placeholder used when empty so the SDK accepts it) |
| `OPENAI_TIMEOUT_SECONDS` | `timeout=...` per request |
| `OPENAI_CUSTOM_HEADERS` | `default_headers=...` — sent on **every** request, for header-based auth like `x-dep-ticket` |

> ⚠️  **Do not commit real API keys or credentials.** Edit `config.py` locally only — or `git update-index --skip-worktree config.py` to make git ignore your local edits to it.

---

## Quickstart

```bash
# 1. Install Python deps (Python 3.10+)
pip install -r requirements.txt

# 2. Choose a provider in config.py (default is Ollama)

# 3. Build the vector store (one-time per provider)
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

| Knob | Where | Default |
|---|---|---|
| Provider switch | [config.py](config.py) `PROVIDER` | `ollama` |
| Ollama LLM | [config.py](config.py) `OLLAMA_LLM_MODEL` | `llama3.1:8b` |
| Ollama embeddings | [config.py](config.py) `OLLAMA_EMBED_MODEL` | `nomic-embed-text` |
| Ollama URL | [config.py](config.py) `OLLAMA_BASE_URL` | `http://localhost:11434` |
| OpenAI LLM | [config.py](config.py) `OPENAI_LLM_MODEL` | `gpt-4o-mini` |
| OpenAI embeddings | [config.py](config.py) `OPENAI_EMBED_MODEL` | `text-embedding-3-small` |
| OpenAI base URL (Azure / internal) | [config.py](config.py) `OPENAI_BASE_URL` | `None` (real OpenAI) |
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
