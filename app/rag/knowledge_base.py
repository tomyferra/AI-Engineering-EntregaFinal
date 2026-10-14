"""Base de conocimiento lista para usar: corpus + retriever híbrido, con su ciclo de vida (conexiones e ingesta).

`open_knowledge_base(settings)` es un async context manager: abre el cliente de Pinecone (o arma el store en
memoria), ingesta el corpus si el namespace no lo tiene y cierra las conexiones al salir.
"""
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from langchain_core.embeddings import Embeddings

from app.config import RAGSettings, VectorStoreKind
from app.llm import build_embeddings
from app.rag.documents import Corpus, load_corpus
from app.rag.reranker import PineconeReranker
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import InMemoryVectorStore, VectorStore, open_pinecone

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KnowledgeBase:
    corpus: Corpus
    retriever: HybridRetriever


def embedding_text(chunk) -> str:
    """Texto que se embebe: título y sección dan contexto a fragmentos cortos (por ejemplo, una tabla)."""
    return f"{chunk.title} · {chunk.section}\n{chunk.text}"


async def ingest(corpus: Corpus, store: VectorStore, embeddings: Embeddings, force: bool = False) -> int:
    """Embebe y sube el corpus. Idempotente: si el namespace ya tiene todos los chunks no hace nada.

    Los ids son deterministas, así que reingestar sobrescribe; `force` vacía antes (borra chunks viejos).
    """
    existentes = await store.count()
    if existentes == len(corpus.chunks) and not force:
        logger.info("El vector store ya tiene los %d chunks del corpus: se omite la ingesta", existentes)
        return 0
    if existentes:
        await store.clear()
    vectors = await embeddings.aembed_documents([embedding_text(c) for c in corpus.chunks])
    await store.upsert(corpus.chunks, vectors)
    logger.info("Ingesta completa: %d chunks de %d documentos", len(corpus.chunks), len(corpus.documents))
    return len(corpus.chunks)


def _retriever(corpus: Corpus, settings: RAGSettings, **backends) -> HybridRetriever:
    return HybridRetriever(corpus, top_k=settings.top_k, candidates_k=settings.candidates_k,
                           bm25_weight=settings.bm25_weight, vector_weight=settings.vector_weight,
                           rrf_k=settings.rrf_k, **backends)


@asynccontextmanager
async def open_knowledge_base(settings: RAGSettings, corpus: Optional[Corpus] = None,
                              embeddings: Optional[Embeddings] = None) -> AsyncIterator[KnowledgeBase]:
    corpus = corpus or load_corpus()
    if settings.vector_store == VectorStoreKind.NONE:
        logger.info("RAG en modo solo BM25 (VECTOR_STORE=none)")
        yield KnowledgeBase(corpus, _retriever(corpus, settings))
        return

    embeddings = embeddings or build_embeddings(settings)
    if settings.vector_store == VectorStoreKind.MEMORY:
        store = InMemoryVectorStore()
        await ingest(corpus, store, embeddings)
        yield KnowledgeBase(corpus, _retriever(corpus, settings, vector_store=store, embeddings=embeddings))
        return

    async with open_pinecone(settings) as (pc, store):
        if settings.auto_ingest:
            await ingest(corpus, store, embeddings)
        reranker = PineconeReranker(pc, settings.rerank_model) if settings.rerank_model else None
        retriever = _retriever(corpus, settings, vector_store=store, embeddings=embeddings, reranker=reranker)
        logger.info("RAG listo: modo %s sobre %s/%s", retriever.mode, settings.index_name, settings.namespace)
        yield KnowledgeBase(corpus, retriever)
