# ResMed AI Agent Demo — Synthetic Dataset

IMPORTANT: This dataset is entirely synthetic and created for a LangChain/RAG/agent demonstration.
It is NOT official ResMed internal data, NOT patient data, and NOT clinical guidance.

Suggested demo:
1. Ingest knowledge_base/*.json into a vector store.
2. Keep customers/support tickets as structured data for tools.
3. Create LangChain tools:
   - search_knowledge(query)
   - get_product(product_name)
   - lookup_customer(customer_id)
   - search_support_tickets(customer_id, product, issue)
   - escalate_to_support(reason)
4. Build an agent that chooses tools.
5. Return source IDs with answers.
6. Add safety rules so clinical/high-risk/unsupported requests are escalated.

Example questions:
- What is AirSense 11?
- What should I check if my mask leaks?
- Show the support history for C001.
- Has C001 reported a mask leakage issue?
- What should the assistant do if someone asks to change therapy pressure?
- What should happen when the knowledge base has no answer?

Do not use this dataset for real patient care or production medical decisions.
