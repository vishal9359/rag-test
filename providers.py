"""LLM + embeddings provider factory.

Defaults to local Ollama. Set LLM_PROVIDER=openai to use any
OpenAI-compatible endpoint (real OpenAI, Azure OpenAI, vLLM, Together,
an internal company gateway, etc.).

Env vars:
    LLM_PROVIDER          ollama | openai          (default: ollama)

  Ollama:
    OLLAMA_BASE_URL       default http://localhost:11434
    OLLAMA_LLM_MODEL      default llama3.1:8b
    OLLAMA_EMBED_MODEL    default nomic-embed-text

  OpenAI / OpenAI-compatible:
    OPENAI_API_KEY        required
    OPENAI_BASE_URL       optional — set this for Azure, vLLM, internal gateway
    OPENAI_LLM_MODEL      default gpt-4o-mini
    OPENAI_EMBED_MODEL    default text-embedding-3-small

The Chroma persist directory is namespaced by provider so embeddings from
different providers (which have different vector dimensions) don't collide.
Switch providers freely without corrupting your index — just re-run ingest.py
the first time you use a new provider.
"""
import os
from pathlib import Path

PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()

PERSIST_DIR = str(Path(__file__).parent / f"chroma_db_{PROVIDER}")


def make_llm(temperature: float = 0):
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": os.environ.get("OPENAI_LLM_MODEL", "gpt-4o-mini"),
            "temperature": temperature,
        }
        if os.environ.get("OPENAI_BASE_URL"):
            kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
        return ChatOpenAI(**kwargs)

    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=os.environ.get("OLLAMA_LLM_MODEL", "llama3.1:8b"),
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=temperature,
    )


def make_embeddings():
    if PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        kwargs = {"model": os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")}
        if os.environ.get("OPENAI_BASE_URL"):
            kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
        return OpenAIEmbeddings(**kwargs)

    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(
        model=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
