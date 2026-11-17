"""Model output -> the section 5 `SystemAnswer` (eval/README.md), deterministically.

- Citations: the model cites source numbers (`citations` plus `[n]` markers in the
  answer text, first occurrence order). Each valid number becomes one `{doc, page}`
  per page of that chunk's `page_start..page_end` span; duplicates are removed;
  numbers outside `1..len(sources)` are dropped and reported.
- Numbers: the value as printed plus a unit with its scale word are normalised with
  `kaprag.eval.numeric` to canonical units (integer TL, %, adet). When that fails the
  answer text is kept, `value`/`unit` become null and the reason is reported. A value
  is never invented.
- An abstention or a clarification request carries no value.

`build_answer` returns `(answer, debug)`; `debug` goes to the run's traces.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from kaprag.eval.answer import Citation, SystemAnswer
from kaprag.eval.numeric import NumericParseError, normalize, parse_quantity, to_json_number
from kaprag.eval.pipeline import RetrievedItem
from kaprag.rag.generate import GeneratedAnswer

_MARKER = re.compile(r"\[(\d+)\]")


def map_citations(
    numbers: Iterable[int], sources: Sequence[RetrievedItem]
) -> tuple[list[Citation], list[int]]:
    """(citations, dropped numbers). Pages in citation order, de-duplicated."""
    citations: list[Citation] = []
    seen: set[tuple[str, int]] = set()
    dropped: list[int] = []
    for n in numbers:
        if not 1 <= n <= len(sources):
            if n not in dropped:
                dropped.append(n)
            continue
        item = sources[n - 1]
        for page in range(item.page_start, item.page_end + 1):
            if (item.doc, page) not in seen:
                seen.add((item.doc, page))
                citations.append(Citation(doc=item.doc, page=page))
    return citations, dropped


def normalize_value(
    value: str | float | int | None, unit: str | None
) -> tuple[tuple[int | float | None, str | None], str | None]:
    """((value, canonical unit), error). `((None, None), None)` when there is no value.

    A string value may carry its own unit marks (`%18,2`); it is parsed together with
    `unit` as one quantity, so exactly one number must appear in the two.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return (None, None), None
    try:
        if isinstance(value, str):
            quantity = parse_quantity(f"{value} {unit or ''}")
        else:
            quantity = normalize(value, unit)
    except NumericParseError as exc:
        return (None, None), str(exc)
    return (to_json_number(quantity.value), quantity.unit), None


def _cited_numbers(generated: GeneratedAnswer) -> list[int]:
    markers = [int(m) for m in _MARKER.findall(generated.answer)]
    return list(dict.fromkeys([*generated.citations, *markers]))


def build_answer(
    generated: GeneratedAnswer, sources: Sequence[RetrievedItem]
) -> tuple[SystemAnswer, dict[str, Any]]:
    cited = _cited_numbers(generated)
    citations, dropped = map_citations(cited, sources)
    clarification = (generated.clarification or "").strip() or None

    error = None
    value = unit = None
    if not generated.abstained and clarification is None:
        (value, unit), error = normalize_value(generated.value, generated.unit)

    answer = SystemAnswer(
        answer=generated.answer,
        value=value,
        unit=unit,
        basis=generated.basis,
        citations=citations,
        abstained=generated.abstained,
        clarification=clarification,
    )
    debug = {
        "cited_sources": [n for n in cited if n not in dropped],
        "dropped_citations": dropped,
        "raw_value": generated.value,
        "raw_unit": generated.unit,
        "normalization_error": error,
    }
    return answer, debug
