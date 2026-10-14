"""Worker: consume la cola de Redis, ejecuta el grafo en segundo plano y actualiza el estado del job.

Corre de dos formas (mismo código):
- dentro de la API (RUN_WORKER_IN_API=true): N corrutinas lanzadas en el lifespan de FastAPI;
- como proceso aparte: `python -m app.worker` (así corre en docker-compose.yml).

Todo es async (Redis, LLM, Pinecone, checkpointer), así que las N ejecuciones concurrentes comparten un solo
event loop sin bloquearlo. Cualquier excepción del grafo (error del proveedor, timeout, límite de pasos) se
captura y deja el job en FAILED con el motivo: el cliente que hace polling nunca queda esperando para siempre.
"""
import asyncio
import logging
from contextlib import suppress
from typing import List

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.config import RECURSION_LIMIT
from app.core.jobs import JobStatus, JobStore, QueueItem, utc_now
from app.graph.state import turn_input
from app.observability import job_context, traced_job

logger = logging.getLogger(__name__)


def summarize_flow(values: dict) -> List[str]:
    """Secuencia de destinos que eligió el supervisor (ej. investigador -> analista -> sintetizador)."""
    return [d["next"] for d in values.get("decisions", [])]


def graph_config(conversation_id: str) -> dict:
    # thread_id = conversation_id: todos los turnos (y la reanudación tras la aprobación) usan el mismo checkpoint.
    return {"configurable": {"thread_id": conversation_id}, "recursion_limit": RECURSION_LIMIT}


class Worker:
    def __init__(self, store: JobStore, graph: CompiledStateGraph, concurrency: int = 5,
                 job_timeout_seconds: float = 300, poll_timeout: float = 1.0):
        self.store = store
        self.graph = graph
        self.concurrency = concurrency
        self.job_timeout_seconds = job_timeout_seconds
        self.poll_timeout = poll_timeout
        self._tasks: List[asyncio.Task] = []
        self._stopping = asyncio.Event()
        # Span raíz de cada ejecución en Phoenix (decorador de OpenInference sobre el método ya ligado).
        self._traced_run = traced_job(self._run_graph)

    # --- ciclo de vida -------------------------------------------------------------------------------------

    def start(self) -> None:
        self._stopping.clear()
        self._tasks = [asyncio.create_task(self._consume(i), name=f"worker-{i}") for i in range(self.concurrency)]
        logger.info("Worker iniciado con %d consumidores", self.concurrency)

    async def stop(self) -> None:
        self._stopping.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._tasks = []

    async def _consume(self, index: int) -> None:
        while not self._stopping.is_set():
            try:
                item = await self.store.dequeue(timeout=self.poll_timeout)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - Redis caído un instante: se reintenta sin matar al consumidor
                logger.exception("worker-%d: error leyendo la cola; reintento en 1s", index)
                await asyncio.sleep(1)
                continue
            if item is not None:
                await self.process(item)

    # --- ejecución de un job -------------------------------------------------------------------------------

    async def _run_graph(self, run: dict) -> dict:
        """Ejecuta (o reanuda) el turno y devuelve el resumen que queda como output del span raíz."""
        config = graph_config(run["conversation_id"])
        if run["kind"] == "resume":
            graph_input = Command(resume=run["resume"])
        else:
            graph_input = turn_input(run["query"], run.get("pending_action"))
        await self.graph.ainvoke(graph_input, config=config)
        snapshot = await self.graph.aget_state(config)
        values = snapshot.values
        summary = {"final_answer": values.get("final_answer"), "delegations": values.get("delegations"),
                   "flow": summarize_flow(values)}
        if snapshot.interrupts:
            return {"status": JobStatus.WAITING_APPROVAL.value, "approval_request": snapshot.interrupts[0].value,
                    **summary}
        return {"status": JobStatus.DONE.value, "action_result": values.get("action_result"),
                "approval": values.get("approval"), **summary}

    async def process(self, item: QueueItem) -> None:
        job_id, kind = item["job_id"], item.get("kind", "start")
        job = await self.store.get(job_id)
        if job is None:
            logger.warning("Job %s no existe (¿expiró?): se descarta", job_id)
            return
        await self.store.set_status(job_id, JobStatus.RUNNING, started_at=utc_now(), error=None)
        run = {"job_id": job_id, "conversation_id": job["conversation_id"], "kind": kind, "query": job["query"],
               "pending_action": job.get("pending_action"), "resume": item.get("resume")}
        try:
            with job_context(job_id, job["conversation_id"], kind, job["query"]):
                result = await asyncio.wait_for(self._traced_run(run), timeout=self.job_timeout_seconds)
        except asyncio.TimeoutError:
            await self._fail(job_id, f"Timeout: la ejecución superó {self.job_timeout_seconds}s")
            return
        except asyncio.CancelledError:
            await self._fail(job_id, "El worker se detuvo durante la ejecución")
            raise
        except Exception as exc:  # noqa: BLE001 - cualquier falla del grafo termina en FAILED, nunca en un loop
            logger.exception("Job %s falló", job_id)
            await self._fail(job_id, f"{type(exc).__name__}: {exc}")
            return

        status = JobStatus(result.pop("status"))
        if status == JobStatus.DONE:
            result["finished_at"] = utc_now()
        await self.store.set_status(job_id, status, **result)
        logger.info("Job %s -> %s", job_id, status.value)

    async def _fail(self, job_id: str, error: str) -> None:
        with suppress(Exception):  # si Redis tampoco responde, al menos queda en el log
            await self.store.set_status(job_id, JobStatus.FAILED, error=error[:2000], finished_at=utc_now())
        logger.error("Job %s -> FAILED: %s", job_id, error)

