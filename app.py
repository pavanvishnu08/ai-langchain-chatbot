"""
app.py
------
All-in-one Streamlit app. Run with:

    streamlit run app.py

Type one question and watch TWO chatbots answer it side by side, using
the SAME underlying model:

  1. Normal chatbot  - no data, answers from memory alone
  2. LangChain chatbot - a tool-calling agent that retrieves from the demo
                         knowledge base (RAG) AND can look up products,
                         customers, and support tickets, or escalate -
                         choosing whichever fits the question

Both grounded in a synthetic ResMed AI demo dataset - not real ResMed
data, not patient data.
"""

import os
import streamlit as st

from normal_chatbot import NormalChatbot
from agent_chatbot import AgentChatbot
from config import VECTORSTORE_DIR, GROQ_MODELS

st.set_page_config(page_title="LangChain vs. Normal Chatbot — ResMed AI Demo", layout="wide")

st.title("🩺 LangChain vs. a Normal Chatbot — ResMed AI Demo")
st.caption(
    "Same LLM on both sides. The only difference: the right side is a LangChain agent that "
    "retrieves from a synthetic ResMed AI demo knowledge base and can look up products, "
    "customers, and support tickets, or escalate. This dataset is entirely synthetic — not "
    "real ResMed data, not patient data."
)

with st.expander("💡 Good questions to try (click to expand)", expanded=False):
    st.markdown(
        """
- What is AirSense 11?
- What should I check if my mask leaks?
- What accessories are compatible with the Lumis 150?
- What should I do if tubing becomes disconnected?
- Can the assistant change my therapy settings?
- What should happen if I report severe breathing difficulty?
- How often should accessories be replaced?
- Show the support history for C001.
- Has C001 reported a mask leakage issue?

The first few work well on both sides. The last two only the **LangChain** chatbot can
really answer — it can look up customer C001's actual ticket history, which the Normal
chatbot has no access to at all.
        """
    )

# ---- Sidebar: setup status & controls -------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    provider = os.getenv("LLM_PROVIDER", "groq")
    st.write(f"**LLM provider:** `{provider}`")

    if provider == "groq" and not os.getenv("GROQ_API_KEY"):
        st.error(
            "GROQ_API_KEY not set. Get a free key at "
            "https://console.groq.com/keys, then set it as an environment "
            "variable and restart the app."
        )
    else:
        st.success("LLM configured ✅")

    if provider == "groq":
        selected_model = st.selectbox("🧠 Model", GROQ_MODELS, key="selected_model")
    else:
        selected_model = None

    # If the model changed, drop the cached bots so they're rebuilt below
    # with the newly selected model.
    if st.session_state.get("_active_model") != selected_model:
        st.session_state.pop("normal_bot", None)
        st.session_state.pop("agent_bot", None)
        st.session_state["_active_model"] = selected_model

    kb_ready = os.path.exists(VECTORSTORE_DIR)
    if kb_ready:
        st.success("Knowledge base index found ✅")
    else:
        st.warning(
            "No vector index yet. Run `python build_knowledge_base.py` "
            "in a terminal first, then reload this page."
        )

    st.divider()
    if st.button("🔄 Reset conversation"):
        for key in ("normal_bot", "agent_bot", "chat_log"):
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()
    st.caption(
        "Cost: $0. Embeddings + vector search run locally (free). "
        "Groq's free tier or a local Ollama model powers the LLM calls."
    )

# ---- Session state ----------------------------------------------------------
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []  # list of dicts: {question, normal_answer, rag_answer, sources}

if "normal_bot" not in st.session_state:
    try:
        st.session_state.normal_bot = NormalChatbot(model=selected_model)
    except Exception as e:
        st.session_state.normal_bot = None
        st.session_state.normal_bot_error = str(e)

if "agent_bot" not in st.session_state and kb_ready:
    try:
        st.session_state.agent_bot = AgentChatbot(model=selected_model)
    except Exception as e:
        st.session_state.agent_bot = None
        st.session_state.agent_bot_error = str(e)

# ---- Input -------------------------------------------------------------------
question = st.chat_input("Ask a question about ResMed products, support, or a customer ID (e.g. C001)...")

if question:
    normal_answer, agent_answer = None, None
    sources, tool_calls = [], []

    if st.session_state.get("normal_bot"):
        with st.spinner("Normal chatbot is answering from memory..."):
            try:
                normal_answer = st.session_state.normal_bot.ask(question)
            except Exception as e:
                normal_answer = f"⚠️ Error: {e}"
    else:
        normal_answer = f"⚠️ Not configured: {st.session_state.get('normal_bot_error', 'unknown error')}"

    if st.session_state.get("agent_bot"):
        with st.spinner("LangChain chatbot is choosing tools, then answering..."):
            try:
                agent_answer = st.session_state.agent_bot.ask(question)
                tool_calls = st.session_state.agent_bot.last_tool_calls
                sources = st.session_state.agent_bot.last_sources
            except Exception as e:
                agent_answer = f"⚠️ Error: {e}"
    else:
        agent_answer = (
            f"⚠️ Not configured: {st.session_state.get('agent_bot_error', 'build the knowledge base first')}"
        )

    st.session_state.chat_log.append(
        {
            "question": question,
            "normal": normal_answer,
            "agent": agent_answer,
            "tool_calls": tool_calls,
            "sources": sources,
        }
    )

# ---- Display conversation, newest first --------------------------------------
for turn in reversed(st.session_state.chat_log):
    st.markdown(f"### 🗨️ {turn['question']}")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**❌ Normal chatbot** _(no data, answers from memory)_")
        st.info(turn["normal"])

    with col2:
        st.markdown("**✅ LangChain chatbot** _(RAG + tools: products, customers, tickets, escalation)_")
        st.success(turn.get("agent"))
        if turn.get("tool_calls"):
            with st.expander("🛠️ Tools called"):
                for t in turn["tool_calls"]:
                    st.write("•", t)
        if turn.get("sources"):
            with st.expander("📚 Retrieved from these sections"):
                for s in turn["sources"]:
                    st.write("•", s)

    st.divider()
