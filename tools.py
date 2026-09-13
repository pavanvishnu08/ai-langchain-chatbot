"""
tools.py
--------
LangChain tools for the agent chatbot (see agent_chatbot.py), backed by
the structured JSON records in resmed_ai_demo_mock_data/. These are
NOT vector-searched - they're exact/filtered lookups over structured
records, which is the right tool for "look up customer C001" or "list
open tickets for product X" (a vector search would be the wrong tool
for that; a database-style filter is).

IMPORTANT: every record here is synthetic demo data. lookup_customer and
search_support_tickets simulate looking up an internal system - they are
not connected to any real ResMed system or real patients.
"""

import json
import os

from langchain_core.tools import tool

from config import MOCK_DATA_DIR


def _load(*parts):
    with open(os.path.join(MOCK_DATA_DIR, *parts), "r", encoding="utf-8") as f:
        return json.load(f)


_PRODUCTS = _load("knowledge_base", "products.json")
_CUSTOMERS = _load("customers", "customer_profiles.json")
_TICKETS = _load("support", "support_tickets.json")

# Lazily built so importing tools.py doesn't require the vector index to
# already exist on disk (e.g. before build_knowledge_base.py has run).
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings

        from config import EMBEDDING_MODEL, VECTORSTORE_DIR

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
        )
        _retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return _retriever


@tool
def search_knowledge(query: str) -> str:
    """Search the demo knowledge base (products, FAQs, accessories,
    safety guidelines, support policies, troubleshooting) for information
    relevant to the query. Use this for general "what is/how do I" style
    questions."""
    docs = _get_retriever().invoke(query)
    if not docs:
        return "No matching knowledge base entries found."
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('section', 'unknown')}]\n{d.page_content}" for d in docs
    )


@tool
def get_product(product_name: str) -> str:
    """Look up a specific demo product's details by name (e.g. 'AirSense 11',
    'AirFit F20'). Use this when the user names a specific product and you
    need its category, description, features, or support topics."""
    needle = product_name.strip().lower()
    matches = [p for p in _PRODUCTS if needle in p["product_name"].lower()]
    if not matches:
        return f"No demo product found matching '{product_name}'."
    return "\n\n".join(
        f"Product: {p['product_name']} ({p['product_id']})\n"
        f"Category: {p['category']}\n"
        f"Description: {p['description']}\n"
        f"Features: {', '.join(p['features'])}\n"
        f"Support topics: {', '.join(p['support_topics'])}"
        for p in matches
    )


@tool
def lookup_customer(customer_id: str) -> str:
    """Look up a demo customer's account record by customer ID (e.g. 'C001').
    Returns their name, primary product, account status, and preferred
    contact channel. All customer records are fictional demo data."""
    needle = customer_id.strip().upper()
    match = next((c for c in _CUSTOMERS if c["customer_id"].upper() == needle), None)
    if not match:
        return f"No demo customer found with ID '{customer_id}'."
    return (
        f"Customer {match['customer_id']}: {match['name']} (FICTIONAL DEMO RECORD)\n"
        f"Primary product: {match['primary_product']}\n"
        f"Account status: {match['account_status']}\n"
        f"Preferred contact channel: {match['preferred_channel']}"
    )


@tool
def search_support_tickets(
    customer_id: str = "", product: str = "", issue: str = ""
) -> str:
    """Search demo support tickets, optionally filtered by customer_id
    (e.g. 'C001'), product (e.g. 'AirFit F20'), and/or issue keyword
    (e.g. 'mask leakage'). Leave a field blank to not filter on it. Use
    this to check a customer's support history or find tickets about a
    given issue/product."""
    results = _TICKETS
    if customer_id.strip():
        needle = customer_id.strip().upper()
        results = [t for t in results if t["customer_id"].upper() == needle]
    if product.strip():
        needle = product.strip().lower()
        results = [t for t in results if needle in t["product"].lower()]
    if issue.strip():
        needle = issue.strip().lower()
        results = [t for t in results if needle in t["issue"].lower()]

    if not results:
        return "No matching demo support tickets found."

    results = results[:10]
    return "\n".join(
        f"{t['ticket_id']} | {t['customer_id']} | {t['product']} | {t['issue']} "
        f"| priority={t['priority']} | status={t['status']} | {t['created_date']}"
        for t in results
    )


@tool
def escalate_to_support(reason: str) -> str:
    """Escalate the current conversation to human/authorized support with
    a short reason (e.g. clinical question, unresolved issue, safety
    concern). This is a SIMULATED escalation for the demo - it does not
    contact a real support system. Always use this instead of attempting
    to resolve clinical, safety, or unsupported requests yourself."""
    return (
        "[DEMO ESCALATION SIMULATED] This conversation has been flagged for "
        f"human/authorized support review. Reason: {reason}. "
        "No real support system was contacted - this is a demo action."
    )


ALL_TOOLS = [
    search_knowledge,
    get_product,
    lookup_customer,
    search_support_tickets,
    escalate_to_support,
]
