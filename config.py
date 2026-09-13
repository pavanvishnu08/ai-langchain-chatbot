"""
config.py
---------
One place to control which LLM powers both chatbots, so the comparison
is fair: same brain, different amount of ResMed knowledge.

Two $0 options are supported out of the box:

1) GROQ (default) - free hosted API, very fast, generous free tier.
   Get a free key at https://console.groq.com/keys (no credit card needed)
   then set it as an environment variable:
       export GROQ_API_KEY="gsk_..."      (Mac/Linux)
       setx GROQ_API_KEY "gsk_..."        (Windows)

2) OLLAMA - fully local, fully offline, zero API key, zero signup.
   Install from https://ollama.com, then in a terminal run:
       ollama pull llama3.1
       ollama serve
   and set LLM_PROVIDER=ollama below (or as an env var).

Both are $0. Groq is easier to get running in 2 minutes; Ollama needs
nothing but your own machine and never leaves your computer.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# "groq" or "ollama"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# Models offered in the sidebar dropdown so you can switch LLMs without
# touching any code. First entry is the default on app start.
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "llama-3.1-8b-instant",
]

# Free, fast, good-quality Groq model. Others: "llama-3.1-70b-versatile"
GROQ_MODEL = os.getenv("GROQ_MODEL", GROQ_MODELS[0])

# Any model you've pulled locally with `ollama pull <name>`
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# Local, free, runs on CPU - no API key, no cost, this is what makes the
# RAG chatbot's *retrieval* step free even though the LLM call itself
# might hit a hosted API.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")

# Synthetic ResMed AI demo dataset (NOT real ResMed website content, NOT
# patient data) - see resmed_ai_demo_mock_data/README.md. This is what the
# RAG chatbot's vector store is built from.
MOCK_DATA_DIR = os.path.join(os.path.dirname(__file__), "resmed_ai_demo_mock_data")
KNOWLEDGE_BASE_DIR = os.path.join(MOCK_DATA_DIR, "knowledge_base")


def get_llm(temperature: float = 0.2, model: str | None = None):
    """Return a chat model instance based on LLM_PROVIDER. Swap providers
    without touching any other file. Pass `model` to override the
    configured default (e.g. from a UI dropdown)."""
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model or OLLAMA_MODEL, temperature=temperature)

    # default: groq
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at "
            "https://console.groq.com/keys and set it as an environment "
            "variable, or switch LLM_PROVIDER to 'ollama' in config.py "
            "to run fully offline for free."
        )
    return ChatGroq(model=model or GROQ_MODEL, temperature=temperature, api_key=api_key)
