"""
agent_chatbot.py
-----------------
A LangChain/LangGraph tool-calling agent - the second point of comparison
alongside normal_chatbot.py (no data, no tools).

This agent decides for itself, per turn, which tool(s) to call:

  - search_knowledge       -> FAISS-backed retrieval over the demo knowledge
                              base (this is the "RAG" step - same retriever
                              rag_chatbot.py uses, just invoked on demand
                              instead of on every single turn)
  - get_product            -> exact product lookup by name
  - lookup_customer        -> exact customer record lookup by ID
  - search_support_tickets -> filtered ticket search by customer/product/issue
  - escalate_to_support    -> simulated hand-off for clinical/safety/unresolved cases

This is what "agentic RAG" or "tool use" means in practice: instead of one
hardcoded retrieval step on every question, the LLM picks whichever
structured or unstructured lookup fits the question, possibly chaining
several calls before answering. rag_chatbot.py still exists as a simpler,
standalone example of fixed-step retrieval (see its own __main__ block).
"""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from config import get_llm
from tools import ALL_TOOLS

SYSTEM_PROMPT = """You are a ResMed AI support demo agent with access to tools
over a synthetic demo dataset (products, FAQs, accessories, safety rules,
support policies, troubleshooting, customer records, and support tickets).
This is NOT real ResMed data and NOT medical advice.

Rules:
- Use search_knowledge for general product/FAQ/troubleshooting questions.
- Use get_product when the user names a specific product.
- Use lookup_customer / search_support_tickets when the user references a
  customer ID or asks about support history.
- Never invent facts, ticket details, or customer information that the
  tools didn't return. If a tool finds nothing, say so.
- Do not diagnose, prescribe, or change therapy settings. For clinical,
  emergency, or safety-flagged requests, call escalate_to_support instead
  of answering yourself.
"""


class AgentChatbot:
    def __init__(self, model: str | None = None):
        llm = get_llm(model=model)
        self.agent = create_agent(model=llm, tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT)
        self.history = []  # list of (role, content) tuples for display
        self.last_tool_calls = []  # tool names used in the most recent turn
        self.last_sources = []  # knowledge base sections retrieved via search_knowledge

    def ask(self, question: str) -> str:
        messages = []
        for role, content in self.history:
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=question))

        result = self.agent.invoke({"messages": messages})
        result_messages = result["messages"]

        self.last_tool_calls = [
            call["name"]
            for m in result_messages
            if isinstance(m, AIMessage)
            for call in (m.tool_calls or [])
        ]

        # search_knowledge's tool output is formatted as "[section]\n<text>"
        # blocks (see tools.py) - pull the section labels back out so the UI
        # can show exactly which knowledge base entries were used, same as
        # rag_chatbot.py's last_sources.
        self.last_sources = [
            line.strip()[1:-1]
            for m in result_messages
            if isinstance(m, ToolMessage) and m.name == "search_knowledge"
            for line in m.content.splitlines()
            if line.strip().startswith("[") and line.strip().endswith("]")
        ]

        final_message = result_messages[-1]
        answer = final_message.content

        self.history.append(("user", question))
        self.history.append(("assistant", answer))
        return answer

    def reset(self):
        self.history = []
        self.last_tool_calls = []
        self.last_sources = []


if __name__ == "__main__":
    bot = AgentChatbot()
    print("Agent chatbot ready. Try: 'Show the support history for C001.'")
    while True:
        q = input("\nYou: ")
        if q.lower() in ("exit", "quit"):
            break
        print("Bot:", bot.ask(q))
        if bot.last_tool_calls:
            print("[Tools used:]", ", ".join(bot.last_tool_calls))
