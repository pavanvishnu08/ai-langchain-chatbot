"""
build_knowledge_base.py
------------------------
Turns orx_ai_demo_mock_data/knowledge_base/*.json into a searchable
FAISS vector index on disk. Run this once (and again any time you edit
the mock data):

    python build_knowledge_base.py

This is the "ingestion" step that a normal chatbot skips entirely - it's
the reason the LangChain bot can answer with grounded facts instead of
guessing.

IMPORTANT: this dataset is entirely synthetic, created for a LangChain/
RAG demonstration. It is NOT real ORX website content and NOT patient
data. See orx_ai_demo_mock_data/README.md.
"""

import json
import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL, KNOWLEDGE_BASE_DIR, VECTORSTORE_DIR

# Each knowledge base file gets its own "renderer" that turns one JSON
# record into a short, self-contained chunk of text plus a human-readable
# label (used as the retrieved "section" shown in the UI).


def _render_product(r):
    label = f"Product: {r['product_name']} ({r['product_id']})"
    text = (
        f"Product: {r['product_name']}\n"
        f"Category: {r['category']}\n"
        f"Description: {r['description']}\n"
        f"Features: {', '.join(r['features'])}\n"
        f"Support topics: {', '.join(r['support_topics'])}"
    )
    return label, text


def _render_faq(r):
    label = f"FAQ: {r['question']} ({r['faq_id']})"
    text = f"Question: {r['question']}\nAnswer: {r['answer']}\nCategory: {r['category']}"
    return label, text


def _render_accessory(r):
    label = f"Accessory: {r['name']} ({r['accessory_id']})"
    text = (
        f"Accessory: {r['name']}\n"
        f"Compatible products: {', '.join(r['compatible_demo_products'])}\n"
        f"Notes: {r['support_notes']}\n"
        f"Replacement rule: {r['replacement_rule']}"
    )
    return label, text


def _render_safety_guideline(r):
    label = f"Safety rule: {r['trigger']} ({r['rule_id']})"
    text = f"Trigger: {r['trigger']}\nRequired action: {r['action']}"
    return label, text


def _render_support_policy(r):
    label = f"Policy: {r['name']} ({r['policy_id']})"
    text = f"Policy: {r['name']}\n{r['policy']}"
    return label, text


def _render_troubleshooting(r):
    label = f"Troubleshooting: {r['issue']} - {r['severity']} ({r['scenario_id']})"
    text = (
        f"Issue: {r['issue']}\n"
        f"Severity: {r['severity']}\n"
        f"Recommended action: {r['recommended_demo_action']}\n"
        f"Escalate if: {', '.join(r['escalate_if'])}"
    )
    return label, text


FILE_RENDERERS = {
    "products.json": _render_product,
    "faq.json": _render_faq,
    "accessories.json": _render_accessory,
    "safety_guidelines.json": _render_safety_guideline,
    "support_policies.json": _render_support_policy,
    "troubleshooting.json": _render_troubleshooting,
}


def load_and_split_documents():
    docs = []
    for filename, renderer in FILE_RENDERERS.items():
        path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        for record in records:
            label, text = renderer(record)
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": "orx_ai_demo_mock_data (synthetic)",
                        "file": filename,
                        "section": label,
                    },
                )
            )
    return docs


def main():
    print("Loading and rendering the synthetic ORX AI demo knowledge base...")
    docs = load_and_split_documents()
    print(f"Created {len(docs)} chunks from {len(FILE_RENDERERS)} JSON files.")

    print(f"Loading free local embedding model: {EMBEDDING_MODEL}")
    print("(first run downloads ~90MB, then it's cached locally)")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"Saved vector index to: {VECTORSTORE_DIR}")
    print("Done. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
