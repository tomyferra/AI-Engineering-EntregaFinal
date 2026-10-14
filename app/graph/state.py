"""Estado compartido del orquestador multi-agente.

Hereda de MessagesState (la conversación con el usuario) y agrega campos estructurados para saber qué agente
aportó qué, qué decidió el supervisor en cada paso, cuándo cortar el flujo y el estado del flujo
Human-in-the-loop. Todo se persiste en Redis con el checkpointer, con `thread_id = conversation_id`:

- `messages` acumula TODOS los turnos de la conversación (memoria entre consultas, sobrevive a reinicios).
- Los campos de trabajo (aportes, decisiones, contadores) son de UN turno: cada consulta nueva los reinicia con
  `Overwrite`, que pisa el valor saltando el reducer acumulativo.
"""
import operator
from typing import Annotated, List, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langgraph.types import Overwrite

# Nombres de los nodos del grafo: son el "vocabulario" con el que el supervisor enruta.
SUPERVISOR = "supervisor"
RESEARCHER = "investigador"
ANALYST = "analista"
SYNTHESIZER = "sintetizador"
APPROVAL = "aprobacion_humana"
EXECUTE_ACTION = "ejecutar_accion"

SpecialistName = Literal["investigador", "analista"]
NextNode = Literal["investigador", "analista", "sintetizador"]


class ToolCallRecord(TypedDict):
    """Una llamada real a una herramienta dentro del especialista (evidencia verificable, no autodeclarada)."""

    tool: str
    args: dict
    result: str


class Contribution(TypedDict):
    """Aporte de un especialista: quién produjo qué, con qué instrucción y con qué tools."""

    agent: SpecialistName
    attempt: int  # 1 = primer intento; >1 = refinamiento pedido por el supervisor
    instruction: str
    content: str
    tool_calls: List[ToolCallRecord]
    issues: List[str]  # problemas detectados por la validación determinista (vacío = aporte validado)


class RoutingDecision(TypedDict):
    """Decisión del supervisor en un paso. `forced` indica que una guarda de seguridad pisó al LLM."""

    step: int
    next: NextNode
    instruction: str
    reasoning: str
    forced: bool


class PendingAction(TypedDict):
    """Acción con efectos secundarios que la consulta pide ejecutar (crítica: requiere aprobación humana)."""

    type: Literal["enviar_informe"]
    recipient: str
    reason: str


class ApprovalDecision(TypedDict):
    """Decisión humana recibida por POST /tasks/{id}/approve (entra al grafo con Command(resume=...))."""

    approved: bool
    reviewer: str
    comment: str


class OrchestratorState(MessagesState):
    next_agent: NextNode
    current_instruction: str
    contributions: Annotated[List[Contribution], operator.add]
    decisions: Annotated[List[RoutingDecision], operator.add]
    delegations: int
    task_completed: bool
    final_answer: str
    pending_action: Optional[PendingAction]
    approval: Optional[ApprovalDecision]
    action_result: Optional[dict]


def turn_input(question: str, pending_action: Optional[PendingAction] = None) -> dict:
    """Entrada de un turno nuevo: agrega la pregunta al historial y reinicia los campos de trabajo."""
    return {
        "messages": [HumanMessage(question)],
        "next_agent": SYNTHESIZER,
        "current_instruction": "",
        "contributions": Overwrite([]),
        "decisions": Overwrite([]),
        "delegations": 0,
        "task_completed": False,
        "final_answer": "",
        "pending_action": pending_action,
        "approval": None,
        "action_result": None,
    }
