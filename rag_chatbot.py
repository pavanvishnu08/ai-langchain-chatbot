"""
rag_chatbot.py
---------------
This is a LangChain-powered Retrieval-Augmented Generation (RAG) chatbot.
Unlike normal_chatbot.py, before answering it:

  1. Embeds the user's question with a free local embedding model
  2. Searches a FAISS vector index built from the synthetic ORX AI
     demo dataset (see build_knowledge_base.py and
     orx_ai_demo_mock_data/README.md) - NOT real ORX website
     content, NOT patient data
  3. Pulls back the most relevant chunks of that data
  4. Hands those chunks to the LLM as *grounding context*, with an
     instruction to only answer from that context and to say so when
     the answer isn't in it
  5. Remembers the conversation so far, so follow-up questions work

This is the architectural difference that matters: the LLM is the same
"brain" as normal_chatbot.py, but this bot is no longer answering from
memory alone - it's answering from a retrieved, checkable source.
"""

import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import EMBEDDING_MODEL, VECTORSTORE_DIR, get_llm

SYSTEM_TEMPLATE = """You are a ORX AI support demo assistant. Answer ONLY
using the CONTEXT below, which was retrieved from a synthetic demo
knowledge base (products, FAQs, accessories, safety guidelines, support
policies, troubleshooting). This is NOT real ORX data and NOT medical
advice.

Rules:
- If the answer is in the context, answer clearly and mention it's based
  on the demo knowledge base.
- If the context does NOT contain the answer, say plainly: "I don't have
  that information in my knowledge base," rather than guessing.
- Never invent statistics, prices, phone numbers, or medical claims that
  aren't in the context.
- Do not diagnose, prescribe, or change therapy settings. For clinical,
  emergency, or safety-flagged requests, say the request should be
  escalated to a qualified professional or authorized support.

CONTEXT:
{context}
"""


class RagChatbot:
    def __init__(self, model: str | None = None):
        if not os.path.exists(VECTORSTORE_DIR):
            raise RuntimeError(
                "No vector index found. Run `python build_knowledge_base.py` "
                "first to build one from the ORX knowledge base."
            )

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = FAISS.load_local(
            VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        self.llm = get_llm(model=model)
        self.history = []
        self.last_sources = []  # chunks used for the most recent answer

    def _format_docs(self, docs):
        return "\n\n---\n\n".join(d.page_content for d in docs)

    def ask(self, question: str) -> str:
        # Retrieval step - this is what normal_chatbot.py never does.
        docs = self.retriever.invoke(question)
        self.last_sources = docs
        context = self._format_docs(docs)

        messages = [SystemMessage(content=SYSTEM_TEMPLATE.format(context=context))]
        for role, content in self.history:
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=question))

        response = self.llm.invoke(messages)
        answer = response.content

        self.history.append(("user", question))
        self.history.append(("assistant", answer))
        return answer

    def reset(self):
        self.history = []
        self.last_sources = []


if __name__ == "__main__":
    bot = RagChatbot()
    print("RAG chatbot ready. Try: 'How many lives has ORX impacted in the last 12 months?'")
    while True:
        q = input("\nYou: ")
        if q.lower() in ("exit", "quit"):
            break
        print("Bot:", bot.ask(q))
        print("\n[Retrieved sections used:]")
        for d in bot.last_sources:
            print(" -", d.metadata.get("section", "unknown section"))
