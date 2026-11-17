"""Evaluation harness (design: eval/README.md).

Modules:
- `schema`: gold question model, `load_questions` / `save_questions`.
- `answer`: `SystemAnswer`, the structured answer every pipeline returns.
- `pipeline`: `Pipeline` protocol, `PipelineOutput`, `RetrievedItem`, `register_pipeline`.
- `numeric`: Turkish number parsing, unit normalisation, tolerance comparison.
- `metrics`: per-question metrics and the headline `answer_score`.
- `judge`: LLM judge (`claude-sonnet-5-5`) and a fake judge for tests.
- `summary`: aggregation into `summary.json` groups.
- `config`, `runner`, `cli`: `configs/<name>.yaml`, the run folder, `scripts/run_eval.py`.
- `fake_pipeline`: the FAKE oracle pipeline used to test the harness.
"""
