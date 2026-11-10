"""
normal_chatbot.py
------------------
This is what most people mean by "just a chatbot": a direct call to an
LLM with a generic system prompt and the raw chat history. It has:

  - No connection to any ORX-specific data
  - No retrieval step
  - No way to check its own facts

It will happily answer questions about ORX using only whatever the
underlying model learned during pretraining - which is often outdated,
generic, or simply invented (a "hallucination"). This file exists so
you can compare it directly against rag_chatbot.py.
"""

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from config import get_llm

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question as best you "
    "can using your general knowledge."
)


class NormalChatbot:
    def __init__(self, model: str | None = None):
        self.llm = get_llm(model=model)
        self.history = []  # list of (role, content) tuples for display

    def ask(self, question: str) -> str:
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
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


if __name__ == "__main__":
    bot = NormalChatbot()
    print("Normal chatbot ready. Try: 'How many lives has ORX impacted in the last 12 months?'")
    while True:
        q = input("\nYou: ")
        if q.lower() in ("exit", "quit"):
            break
        print("Bot:", bot.ask(q))
