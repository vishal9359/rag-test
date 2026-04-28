"""Single configuration file for the project.

EDIT THIS FILE to switch between local Ollama and an OpenAI-compatible API.
No environment variables, no shell exports — just change PROVIDER below.

WARNING: do NOT commit real API keys or credentials. Replace placeholder values
with your real values locally and keep them out of version control.
"""

# =====================================================================
# Choose the model backend: "ollama"  or  "openai"
# =====================================================================
PROVIDER = "ollama"


# =====================================================================
# Ollama (local GPU)  — used when PROVIDER == "ollama"
# =====================================================================
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3.1:8b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"


# =====================================================================
# OpenAI / OpenAI-compatible API  — used when PROVIDER == "openai"
# ---------------------------------------------------------------------
# Works with: real OpenAI, Azure OpenAI, vLLM, internal company gateways,
# or any endpoint that speaks the OpenAI Chat Completions / Embeddings API.
# =====================================================================
OPENAI_BASE_URL = "http://localhost:11434"
OPENAI_LLM_MODEL = "qwen2.5-coder:14b"

# NOTE: if your OPENAI_BASE_URL points at an internal Ollama-backed gateway,
# `text-embedding-3-small` may not exist there — use whatever embedding model
# your gateway actually exposes (e.g. "nomic-embed-text").
OPENAI_EMBED_MODEL = "text-embedding-3-small"

# Per-request timeout in seconds. Generous default for slow internal gateways.
OPENAI_TIMEOUT_SECONDS = 600

# API key. If your gateway authenticates via custom headers instead, leave
# this empty — providers.py will pass a placeholder so the SDK accepts it.
OPENAI_API_KEY = ""

# Headers attached to every request. Use this for internal gateways that
# authenticate via headers (e.g. "x-dep-ticket"). Fill in user-specific
# values before running.
OPENAI_CUSTOM_HEADERS = {
    "x-dep-ticket": "credential:",
    "User-Type": "AD_ID",
    "User-Id": "",
    "Send-System-Name": "rvc-api-module",
}
