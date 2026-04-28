"""LLM + embeddings provider factory.

All settings come from `config.py`. Change `PROVIDER` there to switch between
local Ollama and any OpenAI-compatible endpoint (real OpenAI, Azure OpenAI,
vLLM, internal company gateway, etc.). No environment variables involved.

The Chroma persist directory is namespaced per provider so embeddings of
different dimensions don't collide. Re-run `python ingest.py` once after
switching providers.
"""
from pathlib import Path

import config

PROVIDER = config.PROVIDER.lower()

PERSIST_DIR = str(Path(__file__).parent / f"chroma_db_{PROVIDER}")


def _openai_common_kwargs() -> dict:
    """Build the kwargs shared by ChatOpenAI and OpenAIEmbeddings.

    Handles three things real internal gateways tend to need:
      - empty api_key (auth lives in custom headers) -> use placeholder
      - custom default_headers (e.g. x-dep-ticket) on every request
      - long per-request timeout for slow gateways
    """
    kwargs: dict = {
        # The openai SDK rejects empty/None api_key, even when auth is
        # actually performed by custom headers. Pass a placeholder when
        # the user has left OPENAI_API_KEY empty.
        "api_key": config.OPENAI_API_KEY or "not-needed",
    }
    if config.OPENAI_BASE_URL:
        kwargs["base_url"] = config.OPENAI_BASE_URL
    timeout = getattr(config, "OPENAI_TIMEOUT_SECONDS", None)
    if timeout:
        kwargs["timeout"] = timeout
    headers = getattr(config, "OPENAI_CUSTOM_HEADERS", None)
    if headers:
        kwargs["default_headers"] = headers
    return kwargs


def make_llm(temperature: float = 0):
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.OPENAI_LLM_MODEL,
            temperature=temperature,
            **_openai_common_kwargs(),
        )

    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=config.OLLAMA_LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=temperature,
    )


def make_embeddings():
    if PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBED_MODEL,
            **_openai_common_kwargs(),
        )

    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(
        model=config.OLLAMA_EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
