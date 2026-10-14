"""Memoria conversacional: el checkpointer conserva el historial entre turnos (y entre instancias del grafo,
como después de reiniciar el servidor) y cada turno arranca con sus campos de trabajo limpios."""
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.context import conversation_history
from app.agents.supervisor import FINISH, SupervisorDecision
from app.graph.state import RESEARCHER, SYNTHESIZER
from tests.conftest import fake_model, run_query, scripted_router
from tests.test_graph import build_test_graph


def _turn_graph(kb, checkpointer, answer: str, seen: list, synth_model=None):
    router = scripted_router([
        SupervisorDecision(reasoning="faltan datos", next=RESEARCHER, instruction="buscar"),
        SupervisorDecision(reasoning="ok", next=FINISH),
    ], seen)
    synth_model = synth_model or fake_model([AIMessage(answer)])
    return build_test_graph(kb, router, fake_model([AIMessage("[REAL] dato")]), fake_model([]), synth_model,
                            checkpointer=checkpointer)


async def test_historial_sobrevive_a_una_nueva_instancia_del_grafo(kb):
    checkpointer = InMemorySaver()  # en producción: AsyncRedisSaver (persistente)
    seen = []
    await run_query(_turn_graph(kb, checkpointer, "Q1: $4.427.580", seen), "¿Cuánto facturamos en el Q1?",
                    thread_id="conv")

    # "Reinicio": otro grafo compilado, mismo checkpointer y mismo thread_id.
    synth = fake_model([AIMessage("Q2: $7.931.680")])
    final = await run_query(_turn_graph(kb, checkpointer, "", seen, synth_model=synth), "¿Y en el Q2?",
                            thread_id="conv")

    humanos = [m.text for m in final["messages"] if isinstance(m, HumanMessage)]
    assert humanos == ["¿Cuánto facturamos en el Q1?", "¿Y en el Q2?"]
    assert "Q1: $4.427.580" in seen[-2]["history"]  # el supervisor del turno 2 ve el turno 1
    assert seen[-2]["question"] == "¿Y en el Q2?"
    prompt_sintesis = synth.received[0][-1].text
    assert "¿Cuánto facturamos en el Q1?" in prompt_sintesis and "¿Y en el Q2?" in prompt_sintesis


async def test_cada_turno_reinicia_aportes_decisiones_y_contadores(kb):
    checkpointer = InMemorySaver()
    seen = []
    await run_query(_turn_graph(kb, checkpointer, "uno", seen), "primera consulta", thread_id="conv")
    final = await run_query(_turn_graph(kb, checkpointer, "dos", seen), "segunda consulta", thread_id="conv")

    assert len(final["contributions"]) == 1 and final["contributions"][0]["attempt"] == 1
    assert [d["next"] for d in final["decisions"]] == [RESEARCHER, SYNTHESIZER]
    assert final["delegations"] == 1 and final["final_answer"] == "dos"


async def test_conversaciones_distintas_no_se_mezclan(kb):
    checkpointer = InMemorySaver()
    seen = []
    await run_query(_turn_graph(kb, checkpointer, "uno", seen), "consulta A", thread_id="a")
    await run_query(_turn_graph(kb, checkpointer, "dos", seen), "consulta B", thread_id="b")
    assert seen[-2]["history"] == "(primera consulta de la conversación)"


def test_historial_omite_mensajes_internos_y_limita_turnos():
    messages = []
    for i in range(5):
        messages += [HumanMessage(f"pregunta {i}"), AIMessage("interno", name=RESEARCHER),
                     AIMessage(f"respuesta {i}", name=SYNTHESIZER)]
    messages.append(HumanMessage("actual"))

    history = conversation_history({"messages": messages}, max_turns=2)

    assert "interno" not in history and "actual" not in history
    assert "pregunta 2" not in history and "pregunta 3" in history and "respuesta 4" in history
