"""Armado del contexto acotado que recibe cada nodo (evita la "contaminación de contexto").

Ningún especialista ve el historial completo del sistema: recibe la instrucción del supervisor y solo los
aportes previos que necesita para su tarea. El historial de la conversación lo ven solo el supervisor (para
escribir instrucciones autocontenidas) y el sintetizador (para responder en contexto).
"""
import json
from typing import Iterable, List, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config import HISTORY_TURNS
from app.graph.state import SYNTHESIZER, Contribution, OrchestratorState, ToolCallRecord


def user_request(state: OrchestratorState) -> str:
    """La última consulta del usuario (el objetivo global de la orquestación)."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.text
    return ""


def conversation_history(state: OrchestratorState, max_turns: int = HISTORY_TURNS,
                         answer_max_chars: int = 1500) -> str:
    """Turnos anteriores de la conversación (pregunta + respuesta final), sin el turno actual.

    Solo lo ven el supervisor y el sintetizador: es lo que permite resolver repreguntas como "¿y en el Q3?".
    Los mensajes internos de los especialistas de turnos viejos no se incluyen.
    """
    turns: List[List[str]] = []
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            turns.append([message.text, ""])
        elif isinstance(message, AIMessage) and message.name == SYNTHESIZER and turns:
            turns[-1][1] = message.text
    previous = turns[:-1][-max_turns:]
    if not previous:
        return "(primera consulta de la conversación)"
    return "\n\n".join(f"Usuario: {question}\nRespuesta: {answer[:answer_max_chars] or '(sin respuesta)'}"
                       for question, answer in previous)


def filter_contributions(contributions: Sequence[Contribution], agents: Iterable[str]) -> List[Contribution]:
    permitidos = set(agents)
    return [c for c in contributions if c["agent"] in permitidos]


def format_tool_calls(tool_calls: Sequence[ToolCallRecord]) -> str:
    if not tool_calls:
        return "(ninguna)"
    return "\n".join(f"- {tc['tool']}({json.dumps(tc['args'], ensure_ascii=False)}) -> "
                     f"{' '.join(tc['result'].split())}" for tc in tool_calls)


def format_contributions(contributions: Sequence[Contribution], with_evidence: bool = False) -> str:
    """Aportes en texto. `with_evidence` agrega el registro real de tool calls (lo usa el supervisor para validar)."""
    if not contributions:
        return "(todavía no hay aportes)"
    bloques = []
    for c in contributions:
        validacion = "OK" if not c["issues"] else "FALLÓ -> " + " ".join(c["issues"])
        evidencia = f"Registro de herramientas:\n{format_tool_calls(c['tool_calls'])}\n" if with_evidence else ""
        bloques.append(f"### Aporte de {c['agent']} (intento {c['attempt']})\n"
                       f"Instrucción recibida: {c['instruction']}\n"
                       f"{evidencia}"
                       f"Validación automática: {validacion}\n\n{c['content']}")
    return "\n\n".join(bloques)


def last_ai_text(messages: Sequence[BaseMessage]) -> str:
    """Texto final de un agente ReAct (su último mensaje)."""
    return messages[-1].text if messages else ""
