"""Fixtures y dobles de prueba: LLMs guionados, embeddings deterministas y base de conocimiento sin red."""
import hashlib
from typing import Any, Iterable, List

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from app.agents.supervisor import SupervisorDecision
from app.config import RECURSION_LIMIT
from app.graph.state import turn_input
from app.rag.bm25 import tokenize
from app.rag.documents import Corpus, load_corpus
from app.rag.knowledge_base import KnowledgeBase
from app.rag.retriever import HybridRetriever


class FakeToolCallingModel(FakeMessagesListChatModel):
    """LLM falso con respuestas guionadas: permite testear el grafo sin API keys ni costos.

    Guarda los mensajes recibidos en cada llamada para verificar qué contexto vio cada agente.
    """

    received: List[List[BaseMessage]] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeToolCallingModel":
        return self

    def _generate(self, messages: List[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        self.received.append(list(messages))
        resultado = super()._generate(messages, *args, **kwargs)
        # Copia por llamada: si se devolviera siempre el mismo objeto, add_messages lo deduplicaría por id.
        mensaje = resultado.generations[0].message.model_copy(deep=True)
        mensaje.id = None
        return ChatResult(generations=[ChatGeneration(message=mensaje)])


def fake_model(responses: Iterable[BaseMessage]) -> FakeToolCallingModel:
    return FakeToolCallingModel(responses=list(responses), received=[])


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def scripted_router(decisions: Iterable[SupervisorDecision], seen: list = None) -> RunnableLambda:
    """Router del supervisor con decisiones guionadas (reemplaza a la cadena LCEL). `seen` guarda sus entradas."""
    pendientes = iter(decisions)

    def route(supervisor_input: dict) -> SupervisorDecision:
        if seen is not None:
            seen.append(supervisor_input)
        return next(pendientes)

    return RunnableLambda(route)


class KeywordEmbeddings(Embeddings):
    """Embeddings deterministas por hashing de tokens: textos con palabras en común quedan cerca (sin red)."""

    def __init__(self, dims: int = 256):
        self.dims = dims
        self.calls = 0

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dims
        for token in tokenize(text):
            vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dims] += 1.0
        vector[0] += 1e-6  # nunca un vector nulo
        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.calls += 1
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        self.calls += 1
        return self._embed(text)


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    return load_corpus()


@pytest.fixture(scope="session")
def kb(corpus) -> KnowledgeBase:
    """Base de conocimiento solo BM25: suficiente para los tests del grafo (la híbrida se prueba en test_rag)."""
    return KnowledgeBase(corpus, HybridRetriever(corpus))


async def run_query(graph, question: str, pending_action=None, thread_id: str = "t1") -> dict:
    """Ejecuta un turno como lo hace el worker (thread_id + recursion_limit) y devuelve el estado final."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    return await graph.ainvoke(turn_input(question, pending_action), config=config)
