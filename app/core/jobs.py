"""Estado de jobs, conversaciones y cola de trabajo en Redis (cliente asíncrono: nunca bloquea el event loop).

- Job: hash `<prefijo>:job:<id>` con cada campo serializado en JSON. Expira a los `ttl_seconds`.
- Conversación: lista `<prefijo>:conversation:<id>` con los job_id de sus turnos, en orden. El estado del grafo
  de la conversación (historial de mensajes) vive en el checkpointer con `thread_id = conversation_id`.
- Cola: lista `<prefijo>:queue` (RPUSH para encolar, BLPOP para consumir). Cada item es
  `{"job_id": ..., "kind": "start" | "resume", "resume": {...}}`: el mismo worker arranca un turno nuevo o lo
  reanuda después de la aprobación humana. Como la cola vive en Redis, el worker puede correr dentro de la API
  o en otro proceso.

Ciclo de vida de un job: PENDING -> RUNNING -> (WAITING_APPROVAL -> PENDING -> RUNNING) -> DONE | FAILED
Una conversación admite un solo turno activo a la vez: los turnos comparten el checkpoint del grafo.
"""
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Literal, Optional, TypedDict

from redis.asyncio import Redis
from redis.exceptions import WatchError


class JobStatus(str, Enum):
    PENDING = "PENDING"                    # encolado, esperando un worker libre
    RUNNING = "RUNNING"                    # el grafo se está ejecutando
    WAITING_APPROVAL = "WAITING_APPROVAL"  # pausado en el nodo Human-in-the-loop
    DONE = "DONE"                          # terminó (con o sin acción ejecutada)
    FAILED = "FAILED"                      # excepción, timeout o límite de pasos


TERMINAL_STATUSES = {JobStatus.DONE, JobStatus.FAILED}


class ConversationNotFound(LookupError):
    pass


class ConversationBusy(RuntimeError):
    def __init__(self, conversation_id: str, job_id: str, status: str):
        super().__init__(f"La conversación {conversation_id} tiene un turno en curso ({job_id}: {status})")
        self.job_id = job_id
        self.status = status


class QueueItem(TypedDict, total=False):
    job_id: str
    kind: Literal["start", "resume"]
    resume: dict


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class JobStore:
    def __init__(self, redis: Redis, prefix: str = "genpet", ttl_seconds: int = 7 * 24 * 3600):
        self.redis = redis
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds

    @property
    def queue_key(self) -> str:
        return f"{self.prefix}:queue"

    @property
    def outbox_key(self) -> str:
        return f"{self.prefix}:outbox"

    def job_key(self, job_id: str) -> str:
        return f"{self.prefix}:job:{job_id}"

    def conversation_key(self, conversation_id: str) -> str:
        return f"{self.prefix}:conversation:{conversation_id}"

    @staticmethod
    def _encode(fields: dict) -> dict:
        return {k: json.dumps(v, ensure_ascii=False) for k, v in fields.items()}

    async def _active_turn(self, client, conversation_id: str) -> Optional[tuple]:
        """(job_id, status) del último turno si todavía no terminó; None si la conversación está libre."""
        last_job_id = await client.lindex(self.conversation_key(conversation_id), -1)
        if last_job_id is None:
            return None
        status = await client.hget(self.job_key(last_job_id), "status")
        if status is None or JobStatus(json.loads(status)) in TERMINAL_STATUSES:
            return None
        return last_job_id, json.loads(status)

    async def create(self, query: str, pending_action: Optional[dict] = None,
                     conversation_id: Optional[str] = None) -> dict:
        """Crea el job en PENDING, lo agrega a su conversación y lo encola, todo en una transacción.

        Sin `conversation_id` se abre una conversación nueva. Con uno, debe existir (ConversationNotFound) y no
        tener un turno en curso (ConversationBusy). WATCH sobre la conversación evita que dos turnos simultáneos
        se cuelen a la vez.
        """
        is_new = conversation_id is None
        conversation_id = conversation_id or uuid.uuid4().hex
        conv_key = self.conversation_key(conversation_id)
        job_id = uuid.uuid4().hex
        async with self.redis.pipeline(transaction=True) as pipe:
            while True:
                try:
                    await pipe.watch(conv_key)
                    if not is_new:
                        if not await pipe.exists(conv_key):
                            raise ConversationNotFound(conversation_id)
                        active = await self._active_turn(pipe, conversation_id)
                        if active:
                            raise ConversationBusy(conversation_id, *active)
                    now = utc_now()
                    job = {"job_id": job_id, "conversation_id": conversation_id, "status": JobStatus.PENDING.value,
                           "query": query, "created_at": now, "updated_at": now, "pending_action": pending_action}
                    item: QueueItem = {"job_id": job_id, "kind": "start"}
                    pipe.multi()
                    pipe.hset(self.job_key(job_id), mapping=self._encode(job))
                    pipe.expire(self.job_key(job_id), self.ttl_seconds)
                    pipe.rpush(conv_key, job_id)
                    pipe.expire(conv_key, self.ttl_seconds)
                    pipe.rpush(self.queue_key, json.dumps(item))
                    await pipe.execute()
                    return job
                except WatchError:
                    continue  # otro turno se agregó entre el WATCH y el EXEC: se vuelve a validar

    async def get(self, job_id: str) -> Optional[dict]:
        raw = await self.redis.hgetall(self.job_key(job_id))
        if not raw:
            return None
        return {k: json.loads(v) for k, v in raw.items()}

    async def conversation(self, conversation_id: str) -> Optional[List[dict]]:
        """Turnos de la conversación en orden (los jobs que ya expiraron se omiten). None si no existe."""
        job_ids = await self.redis.lrange(self.conversation_key(conversation_id), 0, -1)
        if not job_ids:
            return None
        jobs = [await self.get(job_id) for job_id in job_ids]
        return [job for job in jobs if job is not None]

    async def update(self, job_id: str, **fields: Any) -> None:
        fields["updated_at"] = utc_now()
        await self.redis.hset(self.job_key(job_id), mapping=self._encode(fields))

    async def set_status(self, job_id: str, status: JobStatus, **fields: Any) -> None:
        await self.update(job_id, status=status.value, **fields)

    async def dequeue(self, timeout: float = 1.0) -> Optional[QueueItem]:
        """Espera hasta `timeout` segundos un item de la cola (BLPOP no bloquea el loop: es I/O async)."""
        popped = await self.redis.blpop([self.queue_key], timeout=timeout)
        if popped is None:
            return None
        _key, payload = popped
        return json.loads(payload)

    async def request_resume(self, job_id: str, decision: dict) -> bool:
        """Registra la decisión humana y encola la reanudación, solo si el job está en WAITING_APPROVAL.

        Compare-and-set con WATCH/MULTI: dos aprobaciones simultáneas no reanudan el grafo dos veces.
        Devuelve False si el job no estaba esperando aprobación.
        """
        key = self.job_key(job_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            while True:
                try:
                    await pipe.watch(key)
                    status = await pipe.hget(key, "status")
                    if status is None or json.loads(status) != JobStatus.WAITING_APPROVAL.value:
                        await pipe.unwatch()
                        return False
                    item: QueueItem = {"job_id": job_id, "kind": "resume", "resume": decision}
                    pipe.multi()
                    pipe.hset(key, mapping=self._encode({"status": JobStatus.PENDING.value, "approval": decision,
                                                         "updated_at": utc_now()}))
                    pipe.rpush(self.queue_key, json.dumps(item, ensure_ascii=False))
                    await pipe.execute()
                    return True
                except WatchError:
                    continue  # otro cliente modificó el job entre el WATCH y el EXEC: se reintenta
