"""Observabilidad con Arize Phoenix (OpenTelemetry + OpenInference).

- `init_observability()` registra el TracerProvider que exporta a Phoenix e instrumenta LangChain/LangGraph:
  cada nodo del grafo, cada llamada al LLM (con tokens de entrada/salida y modelo, de donde Phoenix calcula el
  costo) y cada tool call quedan como spans.
- `traced_job` es el decorador del span raíz de cada ejecución (una corrida del worker = una traza).
- `retrieval_span` registra cada búsqueda del RAG híbrido como span RETRIEVER, con los documentos recuperados.
- El provider NO se registra como global: FastAPI trae telemetría nativa que usaría el provider global y
  mandaría a Phoenix un span por cada GET de polling, ensuciando las trazas y la latencia p95 del proyecto.
- `job_context()` etiqueta todos los spans de una ejecución con `session.id = conversation_id`: en la vista
  Sessions de Phoenix aparecen juntos todos los turnos de una conversación, incluida la reanudación después de
  la aprobación humana.
"""
import json
import logging
from contextlib import contextmanager
from typing import Awaitable, Callable, Iterator, Optional, Sequence, TypeVar

from openinference.instrumentation import OITracer, TraceConfig, using_attributes
from openinference.semconv.trace import DocumentAttributes, SpanAttributes
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider

from app.config import AppSettings

logger = logging.getLogger(__name__)

_provider: Optional[TracerProvider] = None

T = TypeVar("T")


def configure_logging(level: int = logging.INFO) -> None:
    """Logs de la app a INFO; los clientes HTTP (un log por request a Pinecone/LLM) solo a WARNING."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "pinecone", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_tracer() -> OITracer:
    """Tracer de OpenInference sobre el provider de Phoenix (no-op si la observabilidad está apagada)."""
    provider = _provider or trace_api.NoOpTracerProvider()
    return OITracer(provider.get_tracer("genpet.api"), config=TraceConfig())


def traced_job(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Decorador del span raíz de cada ejecución del grafo (span kind CHAIN, con input/output del job).

    Se aplica en tiempo de ejecución (al crear el Worker, después de init_observability), no al importar.
    """
    return get_tracer().chain(name="genpet.job")(fn)


def init_observability(settings: AppSettings) -> Optional[TracerProvider]:
    """Conecta el exportador OTLP de Phoenix e instrumenta LangChain. Idempotente."""
    global _provider
    if not settings.phoenix_enabled:
        logger.info("Observabilidad deshabilitada (PHOENIX_ENABLED=false)")
        return None
    if _provider is not None:
        return _provider

    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register

    _provider = register(
        endpoint=f"{settings.phoenix_collector_endpoint.rstrip('/')}/v1/traces",
        project_name=settings.phoenix_project_name,
        batch=True,                       # exporta en segundo plano: no agrega latencia a los requests
        set_global_tracer_provider=False,  # solo exportan los spans del agente (ver docstring del módulo)
        verbose=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=_provider)
    logger.info("Trazas -> Phoenix %s (proyecto %s)", settings.phoenix_collector_endpoint,
                settings.phoenix_project_name)
    return _provider


def shutdown_observability() -> None:
    """Vacía el buffer de spans pendientes antes de terminar el proceso."""
    global _provider
    if _provider is not None:
        from openinference.instrumentation.langchain import LangChainInstrumentor

        LangChainInstrumentor().uninstrument()
        _provider.force_flush()
        _provider.shutdown()
        _provider = None


@contextmanager
def retrieval_span(query: str, mode: str, categoria: Optional[str]) -> Iterator[Callable[[Sequence], None]]:
    """Span RETRIEVER de OpenInference para la búsqueda híbrida: Phoenix muestra cada documento recuperado con
    su score, fuente y qué recuperador lo encontró (útil para depurar la calidad de la recuperación).

    Devuelve una función `record(chunks)` para registrar los resultados al final de la búsqueda.
    """
    with get_tracer().start_as_current_span("rag.hybrid_search", openinference_span_kind="retriever") as span:
        span.set_input(query)
        span.set_attribute(SpanAttributes.METADATA, json.dumps({"mode": mode, "categoria": categoria}))

        def record(chunks: Sequence) -> None:
            prefix = SpanAttributes.RETRIEVAL_DOCUMENTS
            for i, chunk in enumerate(chunks):
                span.set_attribute(f"{prefix}.{i}.{DocumentAttributes.DOCUMENT_ID}", chunk.chunk_id)
                span.set_attribute(f"{prefix}.{i}.{DocumentAttributes.DOCUMENT_CONTENT}", chunk.contenido)
                span.set_attribute(f"{prefix}.{i}.{DocumentAttributes.DOCUMENT_SCORE}", chunk.score)
                span.set_attribute(f"{prefix}.{i}.{DocumentAttributes.DOCUMENT_METADATA}", json.dumps(
                    {"fuente": chunk.fuente, "seccion": chunk.seccion, "categoria": chunk.categoria,
                     "recuperado_por": chunk.recuperado_por}, ensure_ascii=False))

        yield record


@contextmanager
def job_context(job_id: str, conversation_id: str, kind: str, query: str) -> Iterator[None]:
    """Atributos OpenInference que heredan todos los spans de la ejecución (sesión, metadata y tags)."""
    metadata = {"job_id": job_id, "conversation_id": conversation_id, "run_kind": kind, "query": query[:200]}
    with using_attributes(session_id=conversation_id, metadata=metadata, tags=["genpet", "api", kind]):
        yield
