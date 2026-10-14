"""Composición de dependencias: conexiones compartidas por la API y el worker, con su ciclo de vida.

Siempre: Redis (jobs, conversaciones y cola) y la observabilidad. Con worker, además: base de conocimiento
(Pinecone + embeddings), checkpointer, modelo, grafo compilado y los consumidores de la cola. Una API con el
worker en otro proceso no necesita ni el grafo, ni el RAG, ni la API key del LLM.
"""
import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from langgraph.graph.state import CompiledStateGraph
from redis.asyncio import Redis

from app.config import AppSettings, get_app_settings, get_llm_settings, get_rag_settings, get_tavily_api_key
from app.core.jobs import JobStore
from app.core.worker import Worker
from app.graph.builder import build_graph, build_nodes
from app.graph.checkpointer import redis_checkpointer
from app.llm import build_chat_model
from app.observability import init_observability, shutdown_observability
from app.rag.knowledge_base import open_knowledge_base
from app.tools import RedisOutbox

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    redis: Redis
    store: JobStore
    graph: Optional[CompiledStateGraph]
    worker: Optional[Worker]


@asynccontextmanager
async def build_runtime(settings: Optional[AppSettings] = None, with_worker: bool = True) -> AsyncIterator[Runtime]:
    settings = settings or get_app_settings()
    async with AsyncExitStack() as stack:
        init_observability(settings)
        stack.callback(shutdown_observability)
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        stack.push_async_callback(redis.aclose)
        await redis.ping()
        store = JobStore(redis, prefix=settings.key_prefix, ttl_seconds=settings.job_ttl_seconds)
        graph, worker = None, None
        if with_worker:
            llm_settings = get_llm_settings()
            kb = await stack.enter_async_context(open_knowledge_base(get_rag_settings()))
            checkpointer = await stack.enter_async_context(
                redis_checkpointer(settings.redis_url, settings.job_ttl_seconds))
            nodes = build_nodes(build_chat_model(llm_settings), kb, RedisOutbox(redis, store.outbox_key),
                                get_tavily_api_key())
            graph = build_graph(nodes, checkpointer)
            worker = Worker(store, graph, settings.worker_concurrency, settings.job_timeout_seconds)
            worker.start()
            stack.push_async_callback(worker.stop)
            logger.info("Orquestador listo: LLM %s, RAG %s", llm_settings.label, kb.retriever.mode)
        yield Runtime(redis=redis, store=store, graph=graph, worker=worker)


async def run_standalone_worker() -> None:
    async with build_runtime(with_worker=True):
        logger.info("Worker escuchando la cola de Redis (Ctrl+C para salir)")
        await asyncio.Event().wait()
