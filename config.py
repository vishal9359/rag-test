"""Single configuration file for the project.

EDIT THIS FILE to switch between local Ollama and an OpenAI-compatible API.
No environment variables, no shell exports — just change PROVIDER below.

WARNING: do NOT commit real API keys. Replace the placeholder OPENAI_API_KEY
with your value locally and keep it out of version control.
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
# OPENAI_BASE_URL:
#   - None              -> public api.openai.com
#   - any other URL     -> Azure OpenAI, vLLM, an internal company gateway,
#                          or any OpenAI-compatible endpoint.
# Examples:
#   OPENAI_BASE_URL = "https://your-internal-gateway.example.com/v1"
#   OPENAI_BASE_URL = "https://<resource>.openai.azure.com/openai/v1"
# =====================================================================
OPENAI_API_KEY = "sk-replace-me"
OPENAI_BASE_URL = None
OPENAI_LLM_MODEL = "gpt-4o-mini"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
