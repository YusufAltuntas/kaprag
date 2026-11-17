"""Thin adapter from export items to Langfuse; the only module that imports the SDK.

Written against the Langfuse Python SDK v5 (`langfuse` 5.2.0, `observability` uv
group), which ingests observations through Langfuse's OpenTelemetry endpoint. The
SDK's own observation API stamps spans with the current time and random span ids,
so - like Langfuse's data-migration cookbook - observations are built here as
OpenTelemetry spans with the recorded timestamps and the deterministic ids, and sent
with an OTLP/HTTP exporter to `<host>/api/public/otel/v1/traces` using the same
headers and instrumentation scope as the SDK's own exporter. Attribute names come from
the SDK (`LangfuseOtelSpanAttributes`); values are JSON-encoded the way SDK v5 does.
Scores go synchronously through `Langfuse.api.scores.create` (POST /api/public/scores;
202 accepted, 207 with rejections, which are raised), existence checks through
`Langfuse.api.observations.get_many(trace_id=...)`.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from kaprag.observability.langfuse_mapping import ObservationItem, ScoreItem, TraceItem

INSTALL_HINT = "uv sync --group ml --group app --group observability"
# Trace metadata copied to every span, as `propagate_attributes` does, must encode to at
# most this many characters (SDK rule); these keys and longer values stay on the root.
PROPAGATED_METADATA_MAX_CHARS = 200
ROOT_ONLY_METADATA = frozenset({"pipeline_params", "description"})
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class LangfuseConfigError(Exception):
    """Credentials missing or the SDK not installed; the message never contains secrets."""


class LangfuseExportError(Exception):
    """The Langfuse endpoint rejected a batch of spans."""


@dataclass(frozen=True)
class LangfuseSettings:
    host: str
    public_key: str
    secret_key: str

    def __repr__(self) -> str:  # keep keys out of tracebacks and logs
        return f"LangfuseSettings(host={self.host!r}, public_key=..., secret_key=...)"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LangfuseSettings:
        """`LANGFUSE_HOST` (or the SDK's newer `LANGFUSE_BASE_URL`) and the two keys."""
        env = os.environ if env is None else env
        host = env.get("LANGFUSE_HOST") or env.get("LANGFUSE_BASE_URL") or ""
        values = {
            "LANGFUSE_HOST": host,
            "LANGFUSE_PUBLIC_KEY": env.get("LANGFUSE_PUBLIC_KEY") or "",
            "LANGFUSE_SECRET_KEY": env.get("LANGFUSE_SECRET_KEY") or "",
        }
        missing = [name for name, value in values.items() if not value.strip()]
        if missing:
            raise LangfuseConfigError(
                "missing environment variables: " + ", ".join(missing) + " (set them in .env; "
                "see .env.example). Use --dry-run to check a run without Langfuse."
            )
        return cls(
            host=host.strip().rstrip("/"),
            public_key=values["LANGFUSE_PUBLIC_KEY"].strip(),
            secret_key=values["LANGFUSE_SECRET_KEY"].strip(),
        )


def _ns(moment: datetime) -> int:
    delta = moment - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000


def _json(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def _metadata_attributes(prefix: str, metadata: Mapping[str, Any]) -> dict[str, str]:
    # SDK v5 JSON-encodes every metadata value (strings included); None is not written.
    return {
        f"{prefix}.{key}": json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        for key, value in metadata.items()
        if value is not None
    }


class LangfuseSink:
    """Sends export items to one Langfuse project. Construct only for a real export."""

    def __init__(
        self,
        settings: LangfuseSettings,
        *,
        timeout_s: float = 30.0,
        client: Any = None,
        exporter: Any = None,
    ) -> None:
        """`client` / `exporter` replace the `Langfuse` client and the OTLP exporter (tests)."""
        try:
            import langfuse
            from langfuse import Langfuse, LangfuseOtelSpanAttributes
            from langfuse.api.scores import CreateScoreBatchResults, CreateScoreRequest
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import ReadableSpan
            from opentelemetry.sdk.trace.export import SpanExportResult
            from opentelemetry.sdk.util.instrumentation import InstrumentationScope
            from opentelemetry.trace import SpanContext, SpanKind, Status, StatusCode, TraceFlags
        except ImportError as exc:
            raise LangfuseConfigError(
                f"the Langfuse SDK is not installed ({exc}); install it with `{INSTALL_HINT}`"
            ) from exc

        self._attr = LangfuseOtelSpanAttributes
        self._ReadableSpan = ReadableSpan
        self._SpanContext = SpanContext
        self._SpanKind = SpanKind
        self._Status = Status
        self._StatusCode = StatusCode
        self._TraceFlags = TraceFlags
        self._success = SpanExportResult.SUCCESS
        self._ScoreRequest = CreateScoreRequest
        self._ScoreRejections = CreateScoreBatchResults

        self._client = client or Langfuse(
            public_key=settings.public_key,
            secret_key=settings.secret_key,
            base_url=settings.host,
            timeout=int(timeout_s),
        )
        try:
            self._client.auth_check()  # raises on wrong keys / unreachable host
        except Exception:
            self._client.shutdown()
            raise

        auth = base64.b64encode(f"{settings.public_key}:{settings.secret_key}".encode()).decode()
        self._exporter = exporter or OTLPSpanExporter(
            endpoint=f"{settings.host}/api/public/otel/v1/traces",
            headers={
                "Authorization": f"Basic {auth}",
                "x-langfuse-sdk-name": "python",
                "x-langfuse-sdk-version": langfuse.__version__,
                "x-langfuse-public-key": settings.public_key,
                "x-langfuse-ingestion-version": "4",
            },
            timeout=timeout_s,
        )
        self._scope = InstrumentationScope(
            "langfuse-sdk", langfuse.__version__, attributes={"public_key": settings.public_key}
        )
        self._resource = Resource.create({"service.name": "kaprag-export"})

    def trace_exists(self, trace_id: str) -> bool:
        page = self._client.api.observations.get_many(trace_id=trace_id, limit=1, fields="core")
        return bool(page.data)

    def send_trace(self, trace: TraceItem) -> None:
        spans = [self._span(trace, obs) for obs in trace.observations]
        result = self._exporter.export(spans)
        if result != self._success:
            raise LangfuseExportError(f"OTLP export of {len(spans)} spans returned {result.name}")

    def send_scores(self, scores: list[ScoreItem]) -> None:
        if not scores:
            return
        request = [
            self._ScoreRequest(
                id=score.id,
                trace_id=score.trace_id,
                name=score.name,
                value=score.value,
                data_type=score.data_type,
                comment=score.comment,
                metadata=score.metadata,
                timestamp=score.timestamp,
            )
            for score in scores
        ]
        response = self._client.api.scores.create(request=request)
        if isinstance(response, self._ScoreRejections) and response.rejected:
            messages = "; ".join(e.message for e in response.errors)
            raise LangfuseExportError(f"{response.rejected} scores rejected: {messages}")

    def flush(self) -> None:
        self._client.flush()

    def close(self) -> None:
        self._client.shutdown()
        self._exporter.shutdown()

    def _context(self, trace_id: str, span_id: str) -> Any:
        return self._SpanContext(
            trace_id=int(trace_id, 16),
            span_id=int(span_id, 16),
            is_remote=False,
            trace_flags=self._TraceFlags(self._TraceFlags.SAMPLED),
        )

    def _span(self, trace: TraceItem, obs: ObservationItem) -> Any:
        a = self._attr
        attributes: dict[str, Any] = {a.OBSERVATION_TYPE: obs.type, a.OBSERVATION_LEVEL: obs.level}
        if obs.status_message:
            attributes[a.OBSERVATION_STATUS_MESSAGE] = obs.status_message
        if obs.input is not None:
            attributes[a.OBSERVATION_INPUT] = _json(obs.input)
        if obs.output is not None:
            attributes[a.OBSERVATION_OUTPUT] = _json(obs.output)
        attributes.update(_metadata_attributes(a.OBSERVATION_METADATA, obs.metadata))
        if obs.type == "generation":
            if obs.model:
                attributes[a.OBSERVATION_MODEL] = obs.model
            if obs.usage_details is not None:
                attributes[a.OBSERVATION_USAGE_DETAILS] = _json(obs.usage_details)
            if obs.cost_details is not None:
                attributes[a.OBSERVATION_COST_DETAILS] = _json(obs.cost_details)
        # Trace attributes go on every span, as `propagate_attributes` would set them:
        # Langfuse aggregations (cost by session, tag or trace name) only count
        # observations that carry them. Large metadata values stay on the root.
        attributes[a.TRACE_NAME] = trace.name
        attributes[a.TRACE_TAGS] = list(trace.tags)
        attributes[a.TRACE_SESSION_ID] = trace.session_id
        trace_metadata = _metadata_attributes(a.TRACE_METADATA, trace.metadata)
        if obs.parent_id is not None:
            trace_metadata = {
                key: value
                for key, value in trace_metadata.items()
                if key.removeprefix(f"{a.TRACE_METADATA}.") not in ROOT_ONLY_METADATA
                and len(value) <= PROPAGATED_METADATA_MAX_CHARS
            }
        attributes.update(trace_metadata)
        status = (
            self._Status(self._StatusCode.ERROR, obs.status_message)
            if obs.level == "ERROR"
            else self._Status(self._StatusCode.UNSET)
        )
        return self._ReadableSpan(
            name=obs.name,
            context=self._context(trace.id, obs.id),
            parent=None if obs.parent_id is None else self._context(trace.id, obs.parent_id),
            resource=self._resource,
            attributes=attributes,
            kind=self._SpanKind.INTERNAL,
            status=status,
            start_time=_ns(obs.start),
            end_time=_ns(obs.end),
            instrumentation_scope=self._scope,
        )
