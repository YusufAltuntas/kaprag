"""Model output -> the section 5 `SystemAnswer` (eval/README.md), deterministically.

- Citations: only the model's `citations` field (the sources it declares as
  supporting the answer); `[n]` markers in the answer text are not harvested, since
  answers also mention sources to rule them out. Each valid number becomes one
  `{doc, page}` per page of that chunk's `page_start..page_end` span; duplicates are
  removed; numbers outside `1..len(sources)` are dropped and reported.
- Numbers: the value as printed plus a unit with its scale word are normalised with
  `kaprag.eval.numeric` to canonical units (integer TL, %, adet). When that fails the
  answer text is kept, `value`/`unit` become null and the reason is reported. A value
  is never invented.
- An abstention or a clarification request carries no value and no citations.

`build_answer` returns `(answer, debug)`; `debug` goes to the run's traces.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from kaprag.eval.answer import Citation, SystemAnswer
from kaprag.eval.numeric import NumericParseError, parse_quantity, to_json_number
from kaprag.eval.pipeline import RetrievedItem
from kaprag.rag.generate import GeneratedAnswer


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
    value: str | None, unit: str | None
) -> tuple[tuple[int | float | None, str | None], str | None]:
    """((value, canonical unit), error). `((None, None), None)` when there is no value.

    The value (as printed) is parsed together with `unit` as one quantity, so a `%`
    inside the value works (`%18,2`), while two numbers or two scale words (`"45,2
    milyar TL"` + `"milyar TL"`) fail.
    """
    if value is None or not value.strip():
        return (None, None), None
    try:
        quantity = parse_quantity(f"{value} {unit or ''}")
    except NumericParseError as exc:
        return (None, None), str(exc)
    return (to_json_number(quantity.value), quantity.unit), None


def build_answer(
    generated: GeneratedAnswer, sources: Sequence[RetrievedItem]
) -> tuple[SystemAnswer, dict[str, Any]]:
    clarification = (generated.clarification or "").strip() or None
    answering = not generated.abstained and clarification is None
    cited = list(dict.fromkeys(generated.citations)) if answering else []
    citations, dropped = map_citations(cited, sources)

    error = None
    value = unit = None
    if answering:
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
