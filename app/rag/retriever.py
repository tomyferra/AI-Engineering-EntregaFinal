"""Recuperación híbrida: BM25 (léxico) + vectorial (semántico) en paralelo, fusión RRF y re-ranking opcional.

    consulta ──┬── BM25 (thread) ──────────────┐
               └── embedding -> vector store ──┴── Reciprocal Rank Fusion ── [reranker] ── top_k chunks

Si la búsqueda vectorial o el reranker fallan (red, cuota), se degrada a lo disponible y se registra un
warning: el investigador sigue teniendo resultados léxicos en vez de cortar toda la ejecución.
"""
import asyncio
import logging
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

from app.rag.bm25 import BM25Index
from app.rag.documents import Corpus
from app.rag.reranker import Reranker
from app.rag.vector_store import Hit, VectorStore

logger = logging.getLogger(__name__)

RetrieverName = Literal["bm25", "vector"]


class RetrievedChunk(BaseModel):
    """Fragmento recuperado con su procedencia: de qué documento sale y qué recuperador lo encontró."""

    chunk_id: str
    fuente: str = Field(description="Archivo de origen, para citar")
    seccion: str
    categoria: str
    contenido: str
    score: float = Field(description="Relevancia del reranker o, sin reranker, score RRF de la fusión")
    recuperado_por: List[RetrieverName]


def reciprocal_rank_fusion(rankings: Dict[RetrieverName, Sequence[Hit]], weights: Dict[RetrieverName, float],
                           k: int = 60) -> List[Tuple[str, float, List[RetrieverName]]]:
    """RRF: score(d) = Σ w_r / (k + rank_r(d)). Combina rankings sin normalizar scores de escalas distintas."""
    scores: Dict[str, float] = defaultdict(float)
    sources: Dict[str, List[RetrieverName]] = defaultdict(list)
    for name, hits in rankings.items():
        for rank, (chunk_id, _score) in enumerate(hits, start=1):
            scores[chunk_id] += weights.get(name, 1.0) / (k + rank)
            sources[chunk_id].append(name)
    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [(cid, scores[cid], sources[cid]) for cid in ordered]


class HybridRetriever:
    def __init__(self, corpus: Corpus, *, vector_store: Optional[VectorStore] = None,
                 embeddings: Optional[Embeddings] = None, reranker: Optional[Reranker] = None,
                 top_k: int = 4, candidates_k: int = 8, bm25_weight: float = 0.5, vector_weight: float = 0.5,
                 rrf_k: int = 60):
        if (vector_store is None) != (embeddings is None):
            raise ValueError("vector_store y embeddings van juntos (o ninguno de los dos)")
        self.corpus = corpus
        self.bm25 = BM25Index(corpus.chunks)
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.reranker = reranker
        self.top_k = top_k
        self.candidates_k = candidates_k
        self.weights: Dict[RetrieverName, float] = {"bm25": bm25_weight, "vector": vector_weight}
        self.rrf_k = rrf_k

    @property
    def mode(self) -> str:
        base = "hybrid" if self.vector_store is not None else "bm25"
        return f"{base}+rerank" if self.reranker is not None else base

    async def _vector_hits(self, query: str, categoria: Optional[str]) -> List[Hit]:
        if self.vector_store is None:
            return []
        try:
            vector = await self.embeddings.aembed_query(query)
            return await self.vector_store.query(vector, self.candidates_k, categoria)
        except Exception as exc:  # noqa: BLE001 - se degrada a BM25 en vez de cortar la investigación
            logger.warning("Búsqueda vectorial no disponible (%s: %s); se usa solo BM25", type(exc).__name__, exc)
            return []

    async def _rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        if self.reranker is None or len(candidates) <= 1:
            return candidates[:top_k]
        try:
            ranked = await self.reranker.rerank(query, [c.contenido for c in candidates], top_n=top_k)
        except Exception as exc:  # noqa: BLE001 - sin reranker se conserva el orden de la fusión
            logger.warning("Reranker no disponible (%s: %s); se usa el orden RRF", type(exc).__name__, exc)
            return candidates[:top_k]
        return [candidates[i].model_copy(update={"score": round(score, 4)}) for i, score in ranked]

    async def search(self, query: str, top_k: Optional[int] = None,
                     categoria: Optional[str] = None) -> List[RetrievedChunk]:
        top_k = top_k or self.top_k
        bm25_hits, vector_hits = await asyncio.gather(
            asyncio.to_thread(self.bm25.search, query, self.candidates_k, categoria),
            self._vector_hits(query, categoria),
        )
        fused = reciprocal_rank_fusion({"bm25": bm25_hits, "vector": vector_hits}, self.weights, self.rrf_k)
        candidates = []
        for chunk_id, score, sources in fused[: self.candidates_k]:
            chunk = self.corpus.get(chunk_id)
            if chunk is None:  # vector de una versión vieja del corpus que sigue en el índice
                continue
            candidates.append(RetrievedChunk(chunk_id=chunk.id, fuente=chunk.source, seccion=chunk.section,
                                             categoria=chunk.categoria, contenido=chunk.text,
                                             score=round(score, 4), recuperado_por=sources))
        return await self._rerank(query, candidates, top_k)
