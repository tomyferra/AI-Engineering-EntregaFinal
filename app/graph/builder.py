"""Construcción del grafo: supervisor + especialistas + sintetizador + Human-in-the-loop, con checkpointer.

    START -> supervisor --(route_from_supervisor)--> investigador -> supervisor   (ciclo de delegación)
                                                \\-> analista     -> supervisor
                                                \\-> sintetizador --(route_after_synthesis)--> END
                                                                   \\-> aprobacion_humana  [interrupt]
    aprobacion_humana --(route_after_approval)--> ejecutar_accion -> END   (aprobada)
                                             \\-> END                       (rechazada)

El checkpointer guarda el estado completo después de cada super-step con `thread_id = conversation_id`: es lo
que permite pausar en `aprobacion_humana` y reanudar más tarde desde cualquier worker, y que la conversación
(el historial de turnos) sobreviva a un reinicio del servidor.
"""
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents import (
    build_grounding_validator,
    build_supervisor_router,
    make_analyst_node,
    make_research_node,
    make_supervisor_node,
    make_synthesizer_node,
    route_from_supervisor,
)
from app.graph.hitl import approval_node, make_execute_action_node, route_after_approval, route_after_synthesis
from app.graph.state import ANALYST, APPROVAL, EXECUTE_ACTION, RESEARCHER, SUPERVISOR, SYNTHESIZER, OrchestratorState
from app.rag.knowledge_base import KnowledgeBase
from app.tools import Outbox, build_math_tools, build_report_sender_tool, build_research_tools

Node = Callable[[OrchestratorState], Awaitable[dict]]


@dataclass(frozen=True)
class OrchestratorNodes:
    supervisor: Node
    researcher: Node
    analyst: Node
    synthesizer: Node
    execute_action: Node


def build_graph(nodes: OrchestratorNodes, checkpointer: Optional[BaseCheckpointSaver] = None) -> CompiledStateGraph:
    """Une los nodos. Recibe nodos y checkpointer ya construidos (inyección de dependencias): testeable sin APIs."""
    workflow = StateGraph(OrchestratorState)

    workflow.add_node(SUPERVISOR, nodes.supervisor)
    workflow.add_node(RESEARCHER, nodes.researcher)
    workflow.add_node(ANALYST, nodes.analyst)
    workflow.add_node(SYNTHESIZER, nodes.synthesizer)
    workflow.add_node(APPROVAL, approval_node)
    workflow.add_node(EXECUTE_ACTION, nodes.execute_action)

    workflow.add_edge(START, SUPERVISOR)
    workflow.add_conditional_edges(SUPERVISOR, route_from_supervisor,
                                   {RESEARCHER: RESEARCHER, ANALYST: ANALYST, SYNTHESIZER: SYNTHESIZER})
    # Topología jerárquica: los especialistas nunca hablan entre sí, siempre reportan al supervisor.
    workflow.add_edge(RESEARCHER, SUPERVISOR)
    workflow.add_edge(ANALYST, SUPERVISOR)
    # Human-in-the-loop: solo las tareas críticas pasan por la aprobación antes de ejecutar la acción.
    workflow.add_conditional_edges(SYNTHESIZER, route_after_synthesis, {APPROVAL: APPROVAL, END: END})
    workflow.add_conditional_edges(APPROVAL, route_after_approval, {EXECUTE_ACTION: EXECUTE_ACTION, END: END})
    workflow.add_edge(EXECUTE_ACTION, END)

    return workflow.compile(checkpointer=checkpointer)


def build_nodes(model: BaseChatModel, kb: KnowledgeBase, outbox: Outbox,
                tavily_api_key: Optional[str] = None) -> OrchestratorNodes:
    """Nodos de producción: cada especialista recibe solo sus tools (el investigador busca, el analista calcula)."""
    return OrchestratorNodes(
        supervisor=make_supervisor_node(build_supervisor_router(model)),
        researcher=make_research_node(model, build_research_tools(kb, tavily_api_key),
                                      validator=build_grounding_validator(kb.corpus.documents.values())),
        analyst=make_analyst_node(model, build_math_tools()),
        synthesizer=make_synthesizer_node(model),
        execute_action=make_execute_action_node(build_report_sender_tool(outbox)),
    )
