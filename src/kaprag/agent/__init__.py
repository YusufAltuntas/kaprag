"""Agent router (T10): an explicit LangGraph graph that sends comparison and calculation
questions through decompose -> per-look-up retrieval and extraction -> deterministic
calculation -> synthesis, and every other question to the simple RAG pipeline.

Modules: `plan` (schemas, prompts), `compute` (the arithmetic), `graph` (nodes and
edges), `pipeline` (config, harness registration)."""
