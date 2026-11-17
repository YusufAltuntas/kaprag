"""The configurable retrieve -> generate RAG pipeline evaluated by `kaprag.eval`.

Modules, one responsibility each:
- `config`: the pipeline's params (`configs/<name>.yaml` -> `pipeline.params`).
- `retriever`: retrieval modes and their construction from the config.
- `query_filter`: bank/year metadata filter parsed from the question (one LLM call).
- `context`: numbered sources for the prompt.
- `generate`: system prompt, generation schema, the LLM call.
- `postprocess`: model output -> the eval/README.md section 5 answer (citations, numbers).
- `pipeline`: `RagPipeline` and its harness factory (registry key `rag`).
"""
