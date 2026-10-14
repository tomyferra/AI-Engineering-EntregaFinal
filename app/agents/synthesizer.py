"""Sintetizador: fase final que integra los aportes en una única respuesta para el usuario (cadena LCEL)."""
from typing import Awaitable, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.agents.context import conversation_history, format_contributions, user_request
from app.graph.state import SYNTHESIZER, OrchestratorState

SYNTHESIZER_PROMPT = """Sos el Sintetizador de un equipo multi-agente. Redactás la respuesta final al usuario \
integrando SOLO los aportes de los especialistas.

Resolución de conflictos entre agentes:
- Para cifras derivadas (porcentajes, diferencias, proyecciones) prevalece el analista: fueron calculadas con \
herramientas.
- Para datos de origen (montos, metas, fechas) prevalece el investigador: vienen de una fuente citada.
- Si dos aportes se contradicen y no se puede resolver con estas reglas, mostrá ambos valores y aclaralo.
- Si el analista calculó más de una vez, usá su último aporte (es el que incorpora los datos más recientes).
- Si un aporte tiene "Validación automática: FALLÓ", no uses los montos señalados.
- No hagas cálculos propios: todo número derivado sale del analista. Si algo no fue calculado, decilo.
- No reinterpretes los datos: una META u OBJETIVO nunca se presenta como resultado real.
- Si falta información, decilo explícitamente; nunca completes con datos inventados.
- La conversación previa sirve para entender la consulta actual (por ejemplo, una repregunta), no como fuente \
de datos: los números salen de los aportes de este turno.
- No ejecutás acciones. Si la consulta pide enviar, mandar o compartir el informe, redactá solo el informe: el \
envío lo hace otro paso y únicamente con aprobación humana. Nunca digas que el informe fue o será enviado.

Formato: español rioplatense, claro y breve. Empezá por la respuesta directa, después el detalle en viñetas \
con montos en formato $1.234.567 y al final una línea "Fuentes:" con los archivos citados."""

SYNTHESIZER_INPUT = """Conversación previa:
{history}

Consulta actual del usuario:
{question}

Aportes de los especialistas:
{contributions}"""

SynthesizerNode = Callable[[OrchestratorState], Awaitable[dict]]


def build_synthesis_chain(model: BaseChatModel) -> Runnable:
    """Cadena LCEL: prompt -> modelo -> texto."""
    prompt = ChatPromptTemplate.from_messages([("system", SYNTHESIZER_PROMPT), ("human", SYNTHESIZER_INPUT)])
    return prompt | model | StrOutputParser()


def make_synthesizer_node(model: BaseChatModel) -> SynthesizerNode:
    chain = build_synthesis_chain(model)

    async def synthesizer(state: OrchestratorState) -> dict:
        answer = await chain.ainvoke({"history": conversation_history(state), "question": user_request(state),
                                      "contributions": format_contributions(state["contributions"])})
        return {"messages": [AIMessage(content=answer, name=SYNTHESIZER)], "final_answer": answer,
                "task_completed": True}

    return synthesizer
