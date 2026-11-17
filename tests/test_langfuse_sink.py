"""Langfuse sink against a fake `Langfuse` client and a fake OTLP exporter (no network).

Needs the `observability` dependency group; skipped when it is not installed (CI).
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("langfuse")
pytest.importorskip("opentelemetry.sdk")

from langfuse import LangfuseOtelSpanAttributes as A  # noqa: E402
from langfuse.api.scores import (  # noqa: E402
    CreateScoreBatchError,
    CreateScoreBatchResponse,
    CreateScoreBatchResults,
)
from opentelemetry.sdk.trace.export import SpanExportResult  # noqa: E402
from opentelemetry.trace import StatusCode  # noqa: E402

from kaprag.observability import export_cli  # noqa: E402
from kaprag.observability.langfuse_mapping import load_run  # noqa: E402
from kaprag.observability.langfuse_sink import (  # noqa: E402
    PROPAGATED_METADATA_MAX_CHARS,
    ROOT_ONLY_METADATA,
    LangfuseExportError,
    LangfuseSettings,
    LangfuseSink,
)

FIXTURE = Path(__file__).parent / "fixtures" / "langfuse" / "2026-01-02-router-dev"
SETTINGS = LangfuseSettings(host="http://127.0.0.1:1", public_key="pk-lf-x", secret_key="sk-lf-x")


class FakeExporter:
    def __init__(self, result=SpanExportResult.SUCCESS):
        self.result = result
        self.batches = []
        self.shut = 0

    def export(self, spans):
        self.batches.append(list(spans))
        return self.result

    def shutdown(self):
        self.shut += 1


class FakeClient:
    """Records the calls the sink makes on `Langfuse` and its `api`."""

    def __init__(self, existing=(), score_response=None, auth_error=None):
        self.existing = set(existing)
        self.score_response = score_response or CreateScoreBatchResponse()
        self.auth_error = auth_error
        self.observation_queries = []
        self.score_requests = []
        self.flushed = self.shut = 0
        self.api = SimpleNamespace(
            observations=SimpleNamespace(get_many=self._get_many),
            scores=SimpleNamespace(create=self._create_scores),
        )

    def auth_check(self):
        if self.auth_error:
            raise self.auth_error
        return True

    def _get_many(self, **kwargs):
        self.observation_queries.append(kwargs)
        found = kwargs["trace_id"] in self.existing
        return SimpleNamespace(data=[SimpleNamespace(id="x")] if found else [])

    def _create_scores(self, *, request):
        self.score_requests.append(list(request))
        return self.score_response

    def flush(self):
        self.flushed += 1

    def shutdown(self):
        self.shut += 1


@pytest.fixture
def run():
    return load_run(FIXTURE)


def make_sink(client=None, exporter=None):
    client = client or FakeClient()
    exporter = exporter or FakeExporter()
    return LangfuseSink(SETTINGS, client=client, exporter=exporter), client, exporter


def ns(text):
    moment = datetime.fromisoformat(text)
    return int(round((moment - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds() * 10**6)) * 1000


def spans_of(run, qid):
    sink, _, exporter = make_sink()
    trace = next(t for t in run.traces if t.qid == qid)
    sink.send_trace(trace)
    [batch] = exporter.batches
    return trace, {s.context.span_id: s for s in batch}, batch


def test_span_ids_parents_timestamps_and_scope(run):
    trace, by_id, batch = spans_of(run, "q001")
    assert len(batch) == len(trace.observations) == 8
    for obs, span in zip(trace.observations, batch, strict=True):
        assert span.name == obs.name
        assert span.context.trace_id == int(trace.id, 16)
        assert span.context.span_id == int(obs.id, 16)
        assert span.context.trace_flags.sampled
        if obs.parent_id is None:
            assert span.parent is None
        else:
            assert span.parent.span_id == int(obs.parent_id, 16)
            assert span.parent.span_id in by_id
        assert span.instrumentation_scope.name == "langfuse-sdk"
    root, generate = batch[0], next(s for s in batch if s.name == "generate")
    assert root.start_time == ns("2026-01-02T10:00:00+00:00")
    assert root.end_time == ns("2026-01-02T10:00:05+00:00")
    assert generate.start_time == ns("2026-01-02T10:00:02.987500+00:00")
    assert generate.end_time == ns("2026-01-02T10:00:03+00:00")


def test_generation_attributes(run):
    _, _, batch = spans_of(run, "q001")
    gen = next(s for s in batch if s.name == "generate").attributes
    assert gen[A.OBSERVATION_TYPE] == "generation"
    assert gen[A.OBSERVATION_LEVEL] == "DEFAULT"
    assert gen[A.OBSERVATION_MODEL] == "claude-haiku-5-5"
    assert json.loads(gen[A.OBSERVATION_USAGE_DETAILS]) == {
        "input": 2, "output": 300, "cache_read_input_tokens": 900,
        "cache_creation_input_tokens": 7000,
    }  # fmt: skip
    assert json.loads(gen[A.OBSERVATION_COST_DETAILS]) == {"total": 0.0011}
    assert json.loads(gen[A.OBSERVATION_INPUT]) == {"prompt_hash": "h-generate000"}
    # SDK v5 encoding: every metadata value JSON-encoded per key, strings included.
    assert gen[f"{A.OBSERVATION_METADATA}.role"] == '"pipeline"'
    assert gen[f"{A.OBSERVATION_METADATA}.cached"] == "true"
    assert gen[f"{A.OBSERVATION_METADATA}.source_latency_ms"] == "900.0"
    assert f"{A.OBSERVATION_METADATA}.lookup" not in gen  # None is not written
    assert A.OBSERVATION_OUTPUT not in gen


def test_span_attributes_and_error_status(run):
    _, _, batch = spans_of(run, "q001")
    retrieval = next(s for s in batch if s.name == "retrieval").attributes
    assert retrieval[A.OBSERVATION_TYPE] == "retriever"
    output = json.loads(retrieval[A.OBSERVATION_OUTPUT])
    assert [(r["chunk_id"], r["score"]) for r in output] == [
        ("akbank-2023:fixed:655", 0.998), ("akbank-2024:fixed:664", 0.996),
    ]  # fmt: skip
    assert A.OBSERVATION_MODEL not in retrieval
    assert A.OBSERVATION_COST_DETAILS not in retrieval

    _, _, batch = spans_of(run, "q003")
    root = batch[0]
    assert root.status.status_code == StatusCode.ERROR
    assert root.status.description == "RuntimeError: classify failed"
    assert root.attributes[A.OBSERVATION_LEVEL] == "ERROR"
    assert root.attributes[A.OBSERVATION_STATUS_MESSAGE] == "RuntimeError: classify failed"
    assert batch[1].status.status_code == StatusCode.ERROR


def test_trace_attributes_are_propagated_to_every_span(run):
    trace, _, batch = spans_of(run, "q002")
    short = {k for k, v in trace.metadata.items() if v is not None and k not in ROOT_ONLY_METADATA}
    for span in batch:
        attrs = span.attributes
        assert attrs[A.TRACE_NAME] == "router", span.name
        assert attrs[A.TRACE_SESSION_ID] == "2026-01-02-router-dev", span.name
        assert list(attrs[A.TRACE_TAGS]) == trace.tags, span.name
        prefix = f"{A.TRACE_METADATA}."
        keys = {k.removeprefix(prefix) for k in attrs if k.startswith(prefix)}
        if span.parent is None:
            assert keys == short | ROOT_ONLY_METADATA
            assert json.loads(attrs[f"{A.TRACE_METADATA}.pipeline_params"])["decompose"] == {
                "model": "claude-sonnet-5-5", "max_lookups": 8,
            }  # fmt: skip
        else:
            assert keys == short, span.name
            assert all(
                len(attrs[f"{A.TRACE_METADATA}.{k}"]) <= PROPAGATED_METADATA_MAX_CHARS for k in keys
            )
    sub = next(s for s in batch if s.name == "sub_answer:a").attributes
    assert sub[f"{A.TRACE_METADATA}.config_name"] == '"router"'
    assert sub[f"{A.TRACE_METADATA}.git_commit"] == '"9f1c2e7a0b3d4c5e6f708192a3b4c5d6e7f80912"'


def test_long_metadata_values_stay_on_the_root(run):
    trace = run.traces[0]
    trace.metadata["error"] = "x" * 300
    sink, _, exporter = make_sink()
    sink.send_trace(trace)
    [batch] = exporter.batches
    key = f"{A.TRACE_METADATA}.error"
    assert len(batch[0].attributes[key]) == 302
    assert all(key not in s.attributes for s in batch[1:])


def test_export_failure_raises(run):
    sink, _, _ = make_sink(exporter=FakeExporter(SpanExportResult.FAILURE))
    with pytest.raises(LangfuseExportError, match="FAILURE"):
        sink.send_trace(run.traces[0])


def test_trace_exists_queries_observations_by_trace_id(run):
    existing = run.traces[1].id
    sink, client, _ = make_sink(client=FakeClient(existing={existing}))
    assert sink.trace_exists(existing) is True
    assert sink.trace_exists(run.traces[0].id) is False
    assert client.observation_queries[0] == {"trace_id": existing, "limit": 1, "fields": "core"}


def test_score_requests(run):
    sink, client, _ = make_sink()
    scores = [s for s in run.scores if s.trace_id == run.traces[0].id]
    sink.send_scores(scores)
    sink.send_scores([])
    [request] = client.score_requests
    assert len(request) == 9
    text = next(r for r in request if r.name == "text_correctness")
    assert text.id == f"{run.traces[0].id}-text_correctness"
    assert text.trace_id == run.traces[0].id
    assert (text.value, text.data_type) == (1.0, "BOOLEAN")
    assert text.comment == "matches the reference"
    assert text.timestamp == datetime(2026, 1, 2, 10, 0, 3, tzinfo=UTC)
    body = text.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert body["traceId"] == run.traces[0].id and body["dataType"] == "BOOLEAN"


def test_rejected_scores_raise_and_partial_success_without_rejections_passes(run):
    rejected = CreateScoreBatchResults(
        accepted=8, rejected=1, errors=[CreateScoreBatchError(message="bad value")]
    )
    sink, _, _ = make_sink(client=FakeClient(score_response=rejected))
    with pytest.raises(LangfuseExportError, match="1 scores rejected: bad value"):
        sink.send_scores(run.scores[:9])
    ok = CreateScoreBatchResults(accepted=9, rejected=0, errors=[])
    sink, _, _ = make_sink(client=FakeClient(score_response=ok))
    sink.send_scores(run.scores[:9])


def test_failed_auth_check_shuts_the_client_down():
    client = FakeClient(auth_error=RuntimeError("401 unauthorized"))
    with pytest.raises(RuntimeError, match="401"):
        LangfuseSink(SETTINGS, client=client, exporter=FakeExporter())
    assert client.shut == 1


def test_cli_end_to_end_through_the_sink(capsys):
    client, exporter = FakeClient(), FakeExporter()

    def factory():
        return LangfuseSink(SETTINGS, client=client, exporter=exporter)

    assert export_cli.main(["--run", str(FIXTURE)], sink_factory=factory) == 0
    assert sum(len(b) for b in exporter.batches) == 26
    assert sum(len(r) for r in client.score_requests) == 24
    assert (client.flushed, client.shut, exporter.shut) == (1, 1, 1)
    assert "sent=3 skipped_existing=0 failed=0 scores=24 errors=0" in capsys.readouterr().out
