"""JSON output: put the schema in the prompt, validate the reply with pydantic."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

_FENCE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def schema_instruction(schema: type[BaseModel]) -> str:
    """Text appended to the user prompt when a schema is requested."""
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, sort_keys=True)
    return (
        "\n\nRespond with a single JSON object that conforms to this JSON Schema. "
        "Output only the JSON object, with no prose and no code fences.\n"
        f"JSON Schema:\n{schema_json}"
    )


def retry_instruction(error: str) -> str:
    """Text appended for the single retry after an invalid JSON reply."""
    return (
        "\n\nYour previous reply was not valid for the schema above. "
        f"Validation error:\n{error}\n"
        "Reply again with only a JSON object that conforms to the schema."
    )


def parse_output(text: str, schema: type[BaseModel]) -> BaseModel:
    """Validate `text` (optionally wrapped in a ```json fence) against `schema`.

    Raises pydantic.ValidationError if it does not conform.
    """
    match = _FENCE.match(text)
    payload = match.group(1) if match else text
    return schema.model_validate_json(payload.strip())


__all__ = ["ValidationError", "parse_output", "retry_instruction", "schema_instruction"]
