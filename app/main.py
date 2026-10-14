"""API REST de GenPet Intelligence System (FastAPI). Documentación interactiva en /docs.

Levantar local:  uvicorn app.main:app --port 8000
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncContextManager, AsyncIterator, Callable

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.routes import router
from app.config import get_app_settings
from app.core.runtime import Runtime, build_runtime
from app.observability import configure_logging

load_dotenv()
configure_logging()
logger = logging.getLogger("app")

RuntimeFactory = Callable[[], AsyncContextManager[Runtime]]

DESCRIPTION = """Sistema multi-agente sobre la base documental de GenPet.

Un **supervisor** delega en un **investigador** (RAG híbrido BM25 + Pinecone con filtros de metadatos) y en un
**analista** (herramientas de cálculo); un **sintetizador** redacta la respuesta. Las conversaciones se
persisten en Redis con el checkpointer de LangGraph, cada ejecución se traza en Arize Phoenix y las acciones
críticas (enviar el informe) esperan aprobación humana.

Flujo: `POST /tasks` → polling a `GET /tasks/{job_id}` → (si es crítica) `POST /tasks/{job_id}/approve`.
Para repreguntar, mandá el `conversation_id` de la respuesta en el próximo `POST /tasks`."""


def _default_runtime() -> AsyncContextManager[Runtime]:
    settings = get_app_settings()
    return build_runtime(settings, with_worker=settings.run_worker_in_api)


def create_app(runtime_factory: RuntimeFactory = _default_runtime) -> FastAPI:
    """Crea la app. `runtime_factory` se inyecta en los tests (Redis falso + LLMs guionados)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with runtime_factory() as runtime:
            app.state.store = runtime.store
            app.state.runtime = runtime
            logger.info("API lista (worker %s)", "en proceso" if runtime.worker else "externo: python -m app.worker")
            yield

    app = FastAPI(title="GenPet · Intelligence System", version="1.0.0", description=DESCRIPTION,
                  lifespan=lifespan)
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    return app


app = create_app()
