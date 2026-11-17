"""The committed gold set must always load through the strict schema.

Only validity is asserted: counts and `verified` change as the user reviews the set.
"""

from kaprag.eval.schema import DEFAULT_QUESTIONS, load_questions


def test_committed_gold_set_is_schema_valid():
    questions = load_questions(DEFAULT_QUESTIONS)
    assert questions
