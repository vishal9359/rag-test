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


def make_llm(temperature: float = 0):
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": config.OPENAI_LLM_MODEL,
            "api_key": config.OPENAI_API_KEY,
            "temperature": temperature,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return ChatOpenAI(**kwargs)

    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=config.OLLAMA_LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=temperature,
    )


def make_embeddings():
    if PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        kwargs = {
            "model": config.OPENAI_EMBED_MODEL,
            "api_key": config.OPENAI_API_KEY,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return OpenAIEmbeddings(**kwargs)

    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(
        model=config.OLLAMA_EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
