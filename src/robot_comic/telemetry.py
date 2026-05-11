"""OpenTelemetry instrumentation for robot_comic.

Gated behind ROBOT_INSTRUMENTATION env var:
  unset / empty  — no-op (zero overhead)
  trace          — Console exporter only (local dev / debugging)
  remote         — OTLP + Console exporters (SigNoz or compatible collector)

All public names are safe to call regardless of whether instrumentation is
enabled — callers never need to check ENABLED themselves.
"""

import os
import json
import logging
from typing import Any, Optional

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult, SimpleSpanProcessor
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
_RAW = os.getenv("ROBOT_INSTRUMENTATION", "").strip().lower()
ENABLED: bool = _RAW in {"trace", "remote"}
_REMOTE: bool = _RAW == "remote"

# ---------------------------------------------------------------------------
# OTel SDK initialisation
# ---------------------------------------------------------------------------
_SERVICE = "robot_comic"
_TURN_DURATION_BUCKETS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 5.0, 10.0]
_DEFAULT_BUCKETS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]


_SPAN_ATTRS_TO_KEEP = frozenset({
    "turn.id", "session.id", "turn.outcome", "robot.mode",
    "gen_ai.system", "gen_ai.operation.name", "gen_ai.request.model",
    "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens",
    "tool.name", "tool.id", "vad.duration_ms", "stt.type",
})


class CompactLineExporter(SpanExporter):
    """Writes one RCSPAN JSON line per span to stdout for live monitoring."""

    def export(self, spans: Any) -> SpanExportResult:
        """Serialize each span to a compact JSON line prefixed with RCSPAN."""
        for span in spans:
            dur_ms = (span.end_time - span.start_time) / 1_000_000
            attrs = {k: v for k, v in (span.attributes or {}).items() if k in _SPAN_ATTRS_TO_KEEP}
            line = {
                "name": span.name,
                "trace": format(span.context.trace_id, "032x"),
                "span": format(span.context.span_id, "016x"),
                "parent": format(span.parent.span_id, "016x") if span.parent else None,
                "dur_ms": round(dur_ms, 1),
                "status": span.status.status_code.name,
                "ts": span.end_time // 1_000_000,
                "attrs": attrs,
            }
            print(f"RCSPAN {json.dumps(line, separators=(',', ':'))}", flush=True)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """No-op shutdown."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """No-op flush — output is written synchronously."""
        return True


def _init_otel() -> None:
    """Set up TracerProvider and MeterProvider.  Called once at app startup."""
    resource = Resource.create({SERVICE_NAME: _SERVICE})

    # --- Tracing ---
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(CompactLineExporter()))

    if _REMOTE:
        try:
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            otlp_trace = OTLPSpanExporter()  # reads OTEL_EXPORTER_OTLP_ENDPOINT from env
            tracer_provider.add_span_processor(BatchSpanProcessor(otlp_trace))
            logger.info("OTel: OTLP trace exporter enabled")
        except ImportError:
            logger.warning("OTel: opentelemetry-exporter-otlp-proto-grpc not installed; skipping OTLP traces")

    trace.set_tracer_provider(tracer_provider)

    # --- Metrics ---
    readers = [PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=60_000)]

    if _REMOTE:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

            readers.append(PeriodicExportingMetricReader(OTLPMetricExporter(), export_interval_millis=10_000))
            logger.info("OTel: OTLP metric exporter enabled")
        except ImportError:
            logger.warning("OTel: opentelemetry-exporter-otlp-proto-grpc not installed; skipping OTLP metrics")

    meter_provider = MeterProvider(resource=resource, metric_readers=readers)
    metrics.set_meter_provider(meter_provider)

    logger.info("OTel: instrumentation initialised (mode=%s)", _RAW)


def init() -> None:
    """Initialise OTel if ROBOT_INSTRUMENTATION is set.  Safe to call multiple times."""
    if ENABLED:
        _init_otel()
        _init_instruments()


# ---------------------------------------------------------------------------
# Tracer / meter accessors
# ---------------------------------------------------------------------------

def get_tracer() -> trace.Tracer:
    """Return the robot_comic tracer (no-op when not initialised)."""
    return trace.get_tracer(_SERVICE)


def get_meter() -> metrics.Meter:
    """Return the robot_comic meter (no-op when not initialised)."""
    return metrics.get_meter(_SERVICE)


# ---------------------------------------------------------------------------
# Metric instruments  (module-level singletons, created after init_otel())
# ---------------------------------------------------------------------------
_meter: Optional[metrics.Meter] = None

# Histograms
turn_duration: Optional[metrics.Histogram] = None
llm_operation_duration: Optional[metrics.Histogram] = None
ttft: Optional[metrics.Histogram] = None
stt_duration: Optional[metrics.Histogram] = None
tts_duration: Optional[metrics.Histogram] = None
tts_first_audio: Optional[metrics.Histogram] = None

# Counters
frame_drops: Optional[metrics.Counter] = None
playback_underruns: Optional[metrics.Counter] = None
errors: Optional[metrics.Counter] = None


def _init_instruments() -> None:
    global _meter, turn_duration, llm_operation_duration, ttft
    global stt_duration, tts_duration, tts_first_audio
    global frame_drops, playback_underruns, errors

    _meter = get_meter()

    turn_duration = _meter.create_histogram(
        "robot.turn.duration",
        unit="s",
        description="End-to-end conversational turn duration",
    )
    llm_operation_duration = _meter.create_histogram(
        "gen_ai.client.operation.duration",
        unit="s",
        description="LLM request duration (OTel GenAI semconv)",
    )
    ttft = _meter.create_histogram(
        "gen_ai.server.time_to_first_token",
        unit="s",
        description="Time from LLM request sent to first token received",
    )
    stt_duration = _meter.create_histogram(
        "robot.stt.duration",
        unit="s",
        description="Speech-to-text transcription duration",
    )
    tts_duration = _meter.create_histogram(
        "robot.tts.duration",
        unit="s",
        description="Text-to-speech synthesis duration",
    )
    tts_first_audio = _meter.create_histogram(
        "robot.tts.time_to_first_audio",
        unit="s",
        description="Time from TTS request to first audio byte",
    )
    frame_drops = _meter.create_counter(
        "robot.audio.capture.frame_drops",
        unit="frames",
        description="Audio capture frames dropped due to consumer lag",
    )
    playback_underruns = _meter.create_counter(
        "robot.audio.playback.underruns",
        unit="events",
        description="Audio playback buffer underruns",
    )
    errors = _meter.create_counter(
        "robot.errors",
        unit="events",
        description="Application errors",
    )


# ---------------------------------------------------------------------------
# Helpers for recording metrics safely (no-op when instrument is None)
# ---------------------------------------------------------------------------

def record_turn(duration_s: float, attrs: dict[str, Any]) -> None:
    """Record robot.turn.duration histogram."""
    if turn_duration is not None:
        turn_duration.record(duration_s, attributes=attrs)


def record_llm_duration(duration_s: float, attrs: dict[str, Any]) -> None:
    """Record gen_ai.client.operation.duration histogram."""
    if llm_operation_duration is not None:
        llm_operation_duration.record(duration_s, attributes=attrs)


def record_ttft(duration_s: float, attrs: dict[str, Any]) -> None:
    """Record gen_ai.server.time_to_first_token histogram."""
    if ttft is not None:
        ttft.record(duration_s, attributes=attrs)


def record_stt(duration_s: float, attrs: dict[str, Any]) -> None:
    """Record robot.stt.duration histogram."""
    if stt_duration is not None:
        stt_duration.record(duration_s, attributes=attrs)


def record_tts(duration_s: float, attrs: dict[str, Any]) -> None:
    """Record robot.tts.duration histogram."""
    if tts_duration is not None:
        tts_duration.record(duration_s, attributes=attrs)


def record_tts_first_audio(duration_s: float, attrs: dict[str, Any]) -> None:
    """Record robot.tts.time_to_first_audio histogram."""
    if tts_first_audio is not None:
        tts_first_audio.record(duration_s, attributes=attrs)


def inc_frame_drops(count: int, attrs: dict[str, Any]) -> None:
    """Increment robot.audio.capture.frame_drops counter."""
    if frame_drops is not None and count > 0:
        frame_drops.add(count, attributes=attrs)


def inc_playback_underruns(attrs: dict[str, Any]) -> None:
    """Increment robot.audio.playback.underruns counter."""
    if playback_underruns is not None:
        playback_underruns.add(1, attributes=attrs)


def inc_errors(attrs: dict[str, Any]) -> None:
    """Increment robot.errors counter."""
    if errors is not None:
        errors.add(1, attributes=attrs)

