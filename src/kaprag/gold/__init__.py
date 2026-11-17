"""Gold question set drafting (eval/README.md §7). Every draft is `author: llm`,
`verified: false`; only the user verifies, in the review app.

Stages, one module each, run in order by `pipeline.Pipeline` (CLI: scripts/draft_gold.py):
`pages` (step 1 sampling) -> `drafting` (2) -> `checking` (3) -> `probes` + `derive` (4)
-> `abstain` (5) -> `ambiguous` (6) -> `assemble` (quotas, ids, split, pilot; 7 and 9).
"""
