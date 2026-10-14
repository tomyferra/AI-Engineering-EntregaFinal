"""API de punta a punta con Redis falso (fakeredis) y LLMs guionados: encolado, polling, FAILED, conversaciones
y aprobación humana. Ejercita el mismo Worker que corre en producción."""
import asyncio
from contextlib import asynccontextmanager

import fakeredis
import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.supervisor import FINISH, SupervisorDecision
from app.core.jobs import JobStatus, JobStore
from app.core.runtime import Runtime
from app.core.worker import Worker
from app.graph.state import RESEARCHER
from app.main import create_app
from app.tools import InMemoryOutbox
from tests.conftest import fake_model, scripted_router
from tests.test_graph import build_test_graph

TERMINAL = {JobStatus.DONE.value, JobStatus.FAILED.value, JobStatus.WAITING_APPROVAL.value}


def ok_graph(kb, outbox, turns: int = 1, seen=None):
    decisions = []
    for _ in range(turns):
        decisions += [SupervisorDecision(reasoning="faltan datos", next=RESEARCHER, instruction="buscar ingresos"),
                      SupervisorDecision(reasoning="ok", next=FINISH)]
    research = fake_model([AIMessage(f"[REAL] Q{i + 1}: $4.427.580") for i in range(turns)])
    synth = fake_model([AIMessage(f"Respuesta {i + 1}") for i in range(turns)])
    return build_test_graph(kb, scripted_router(decisions, seen), research, fake_model([]), synth, outbox=outbox,
                            checkpointer=InMemorySaver())


def broken_graph(kb):
    def sin_creditos(_input):
        raise ConnectionError("402 Payment Required")

    return build_test_graph(kb, RunnableLambda(sin_creditos), fake_model([]), fake_model([]), fake_model([]),
                            checkpointer=InMemorySaver())


@asynccontextmanager
async def client_for(graph, job_timeout_seconds: float = 30):
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    store = JobStore(redis, prefix="test")

    @asynccontextmanager
    async def runtime_factory():
        worker = Worker(store, graph, concurrency=2, job_timeout_seconds=job_timeout_seconds, poll_timeout=0.05)
        worker.start()
        try:
            yield Runtime(redis=redis, store=store, graph=graph, worker=worker)
        finally:
            await worker.stop()

    app = create_app(runtime_factory)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client, store


async def poll(client, job_id: str, statuses=TERMINAL, timeout: float = 10) -> dict:
    async def _wait():
        while True:
            body = (await client.get(f"/tasks/{job_id}")).json()
            if body["status"] in statuses:
                return body
            await asyncio.sleep(0.02)

    return await asyncio.wait_for(_wait(), timeout)


async def test_post_encola_y_devuelve_job_id_sin_esperar_al_grafo(kb):
    async with client_for(ok_graph(kb, InMemoryOutbox())) as (client, _store):
        response = await client.post("/tasks", json={"query": "¿Cuánto facturamos en el Q1?"})

        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "PENDING" and body["critical"] is False and len(body["job_id"]) == 32
        assert body["links"]["conversation"] == f"/conversations/{body['conversation_id']}"

        done = await poll(client, body["job_id"])
        assert done["status"] == "DONE" and done["final_answer"] == "Respuesta 1"
        assert done["flow"] == [RESEARCHER, "sintetizador"] and done["finished_at"]


async def test_conversacion_de_varios_turnos(kb):
    seen = []
    async with client_for(ok_graph(kb, InMemoryOutbox(), turns=2, seen=seen)) as (client, _store):
        first = (await client.post("/tasks", json={"query": "¿Cuánto facturamos en el Q1?"})).json()
        await poll(client, first["job_id"])

        second = (await client.post("/tasks", json={"query": "¿Y en el Q2?",
                                                    "conversation_id": first["conversation_id"]})).json()
        assert second["conversation_id"] == first["conversation_id"]
        assert (await poll(client, second["job_id"]))["final_answer"] == "Respuesta 2"

        # El supervisor del segundo turno recibió el turno anterior como historial.
        assert "¿Cuánto facturamos en el Q1?" in seen[2]["history"] and "Respuesta 1" in seen[2]["history"]

        conversation = (await client.get(f"/conversations/{first['conversation_id']}")).json()
        assert [t["query"] for t in conversation["turns"]] == ["¿Cuánto facturamos en el Q1?", "¿Y en el Q2?"]
        assert [t["final_answer"] for t in conversation["turns"]] == ["Respuesta 1", "Respuesta 2"]


async def test_conversacion_inexistente_o_con_turno_en_curso(kb):
    async with client_for(ok_graph(kb, InMemoryOutbox())) as (client, store):
        unknown = await client.post("/tasks", json={"query": "consulta", "conversation_id": "a" * 32})
        assert unknown.status_code == 404
        assert (await client.get(f"/conversations/{'a' * 32}")).status_code == 404

        job = await store.create("consulta en curso")
        await store.set_status(job["job_id"], JobStatus.WAITING_APPROVAL)
        busy = await client.post("/tasks", json={"query": "otra consulta", "conversation_id": job["conversation_id"]})
        assert busy.status_code == 409 and job["job_id"] in busy.json()["detail"]


async def test_turnos_simultaneos_en_la_misma_conversacion_solo_entra_uno(kb):
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    store = JobStore(redis, prefix="test")
    first = await store.create("consulta")
    await store.set_status(first["job_id"], JobStatus.DONE)

    results = await asyncio.gather(*(store.create(f"turno {i}", conversation_id=first["conversation_id"])
                                     for i in range(5)), return_exceptions=True)

    assert sum(not isinstance(r, Exception) for r in results) == 1
    assert len(await store.conversation(first["conversation_id"])) == 2


async def test_excepcion_del_agente_deja_el_job_en_failed(kb):
    async with client_for(broken_graph(kb)) as (client, _store):
        job_id = (await client.post("/tasks", json={"query": "¿Cuánto facturamos?"})).json()["job_id"]

        failed = await poll(client, job_id)

        assert failed["status"] == "FAILED"
        assert "ConnectionError" in failed["error"] and "402" in failed["error"]


async def test_timeout_deja_el_job_en_failed(kb):
    async def lento(_input):
        await asyncio.sleep(5)

    graph = build_test_graph(kb, RunnableLambda(lento), fake_model([]), fake_model([]), fake_model([]),
                             checkpointer=InMemorySaver())
    async with client_for(graph, job_timeout_seconds=0.2) as (client, _store):
        job_id = (await client.post("/tasks", json={"query": "consulta lenta"})).json()["job_id"]

        failed = await poll(client, job_id)

        assert failed["status"] == "FAILED" and failed["error"].startswith("Timeout")


async def test_flujo_hitl_completo_por_api(kb):
    outbox = InMemoryOutbox()
    async with client_for(ok_graph(kb, outbox)) as (client, _store):
        created = (await client.post("/tasks", json={"query": "Ingresos del Q1",
                                                     "send_report_to": "ceo@genpet.com.ar"})).json()
        assert created["critical"] is True

        waiting = await poll(client, created["job_id"])
        assert waiting["status"] == "WAITING_APPROVAL"
        assert waiting["approval_request"]["action"]["recipient"] == "ceo@genpet.com.ar"
        assert outbox.sent == []

        approve = await client.post(f"/tasks/{created['job_id']}/approve", json={"approved": True, "reviewer": "ana"})
        assert approve.status_code == 202

        done = await poll(client, created["job_id"], statuses={"DONE", "FAILED"})
        assert done["status"] == "DONE" and done["action_result"]["executed"] is True
        assert done["approval"]["reviewer"] == "ana"
        assert [m["recipient"] for m in outbox.sent] == ["ceo@genpet.com.ar"]

        # Una segunda aprobación ya no aplica: el job no está esperando.
        again = await client.post(f"/tasks/{created['job_id']}/approve", json={"approved": True})
        assert again.status_code == 409


async def test_aprobaciones_simultaneas_reanudan_una_sola_vez():
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    store = JobStore(redis, prefix="test")
    job = await store.create("consulta")
    await redis.delete(store.queue_key)
    await store.set_status(job["job_id"], JobStatus.WAITING_APPROVAL)

    results = await asyncio.gather(*(store.request_resume(job["job_id"], {"approved": True}) for _ in range(5)))

    assert results.count(True) == 1
    assert await redis.llen(store.queue_key) == 1


@pytest.mark.parametrize("payload", [
    {"query": "hey"},
    {"query": "consulta válida", "send_report_to": "no-es-email"},
    {"query": "consulta válida", "conversation_id": "no-es-un-id"},
    {"query": "x" * 2001},
])
async def test_validacion_de_entrada(kb, payload):
    async with client_for(ok_graph(kb, InMemoryOutbox())) as (client, _store):
        assert (await client.post("/tasks", json=payload)).status_code == 422


async def test_job_inexistente_y_health(kb):
    async with client_for(ok_graph(kb, InMemoryOutbox())) as (client, _store):
        assert (await client.get("/tasks/no-existe")).status_code == 404
        assert (await client.post("/tasks/no-existe/approve", json={})).status_code == 404
        assert (await client.get("/health")).json() == {"status": "ok", "redis": "ok", "worker": "in-process"}
        assert (await client.post("/tasks/x/approve", json={"reviewer": ""})).status_code == 422
