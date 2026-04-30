"""Ingest sample_docs/<domain>/* into per-domain Chroma collections.

The supervisor routes by domain, so each domain gets its OWN collection — that
keeps retrieval focused and lets us measure relevance per agent.

Embeddings come from `providers.py`, so this works with either local Ollama
or an OpenAI-compatible endpoint depending on `PROVIDER` in `config.py`.

Run once (or after switching providers):
    python ingest.py
"""
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from providers import CHROMA_SETTINGS, PERSIST_DIR, PROVIDER, make_embeddings

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "sample_docs"

DOMAINS = {
    "hr": "hr_kb",
    "product": "product_kb",
}


def ingest_domain(subfolder: str, collection: str, embeddings) -> int:
    paths = list((DOCS_DIR / subfolder).rglob("*.txt"))
    if not paths:
        print(f"[skip] no .txt files under sample_docs/{subfolder}")
        return 0
    raw = []
    for p in paths:
        raw.extend(TextLoader(str(p), encoding="utf-8").load())
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=80
    ).split_documents(raw)
    Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=collection,
        client_settings=CHROMA_SETTINGS,
    )
    return len(chunks)


def main():
    print(f"Provider: {PROVIDER}  ->  persist dir: {PERSIST_DIR}")
    embeddings = make_embeddings()
    for subfolder, collection in DOMAINS.items():
        n = ingest_domain(subfolder, collection, embeddings)
        if n:
            print(f"Ingested {n} chunks into '{collection}'")
    print("Done.")


if __name__ == "__main__":
    main()
