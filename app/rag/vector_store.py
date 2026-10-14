"""Vector stores asíncronos: Pinecone (producción) y uno en memoria (sin servicios externos / tests).

Ambos cumplen el mismo protocolo, así el retriever no sabe (ni le importa) dónde viven los vectores.
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
from pinecone import AsyncIndex, AsyncPinecone, ServerlessSpec

from app.config import RAGSettings
from app.rag.documents import Chunk

logger = logging.getLogger(__name__)

Hit = Tuple[str, float]  # (chunk_id, similitud)


class VectorStore(Protocol):
    async def count(self) -> int: ...

    async def clear(self) -> None: ...

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    async def query(self, vector: Sequence[float], k: int, categoria: Optional[str] = None) -> List[Hit]: ...


class InMemoryVectorStore:
    """Similitud coseno con numpy sobre una matriz en memoria (el corpus entra entero en RAM)."""

    def __init__(self) -> None:
        self._ids: List[str] = []
        self._categorias: List[str] = []
        self._matrix = np.empty((0, 0))

    async def count(self) -> int:
        return len(self._ids)

    async def clear(self) -> None:
        self.__init__()

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        nuevos: Dict[str, Tuple[str, np.ndarray]] = {c.id: (c.categoria, np.asarray(v, dtype=float))
                                                      for c, v in zip(chunks, vectors)}
        actuales = {i: (cat, row) for i, cat, row in zip(self._ids, self._categorias, self._matrix)}
        actuales.update(nuevos)
        self._ids = list(actuales)
        self._categorias = [cat for cat, _ in actuales.values()]
        filas = np.vstack([row for _, row in actuales.values()])
        self._matrix = filas / np.linalg.norm(filas, axis=1, keepdims=True)

    async def query(self, vector: Sequence[float], k: int, categoria: Optional[str] = None) -> List[Hit]:
        if not self._ids:
            return []
        q = np.asarray(vector, dtype=float)
        sims = self._matrix @ (q / np.linalg.norm(q))
        orden = np.argsort(-sims)
        hits = [(self._ids[i], float(sims[i])) for i in orden
                if categoria is None or self._categorias[i] == categoria]
        return hits[:k]


class PineconeVectorStore:
    """Un namespace de un índice serverless de Pinecone, con el cliente asyncio oficial (sin bloquear el loop)."""

    def __init__(self, index: AsyncIndex, namespace: str):
        self._index = index
        self.namespace = namespace

    async def count(self) -> int:
        stats = await self._index.describe_index_stats()
        ns = stats.namespaces.get(self.namespace)
        return ns.vector_count if ns else 0

    async def clear(self) -> None:
        if await self.count():
            await self._index.delete(delete_all=True, namespace=self.namespace)

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        records = [{"id": c.id, "values": list(v), "metadata": c.metadata()} for c, v in zip(chunks, vectors)]
        await self._index.upsert(vectors=records, namespace=self.namespace, batch_size=100, show_progress=False)

    async def query(self, vector: Sequence[float], k: int, categoria: Optional[str] = None) -> List[Hit]:
        # El filtro por metadatos se resuelve en Pinecone: solo compiten los chunks de esa categoría.
        response = await self._index.query(vector=list(vector), top_k=k, namespace=self.namespace,
                                           filter={"categoria": {"$eq": categoria}} if categoria else None)
        return [(match.id, float(match.score)) for match in response.matches]


async def ensure_index(pc: AsyncPinecone, settings: RAGSettings, timeout_s: float = 120) -> AsyncIndex:
    """Abre el índice; si no existe lo crea (serverless). Si existe, valida que la dimensión coincida."""
    name = settings.index_name
    if await pc.indexes.exists(name):
        description = await pc.indexes.describe(name)
        if description.dimension != settings.dimension:
            raise ValueError(f"El índice '{name}' tiene dimensión {description.dimension}, pero los embeddings "
                             f"generan {settings.dimension}. Usá otro PINECONE_INDEX_NAME o EMBEDDING_DIMENSION.")
    else:
        logger.info("Creando índice serverless '%s' (dim=%d, %s/%s)", name, settings.dimension, settings.cloud,
                    settings.region)
        await pc.indexes.create(name=name, dimension=settings.dimension, metric="cosine",
                                spec=ServerlessSpec(cloud=settings.cloud, region=settings.region))
    async with asyncio.timeout(timeout_s):
        while not (await pc.indexes.describe(name)).status.ready:
            await asyncio.sleep(2)
    return await pc.index(name)


@asynccontextmanager
async def open_pinecone(settings: RAGSettings) -> AsyncIterator[Tuple[AsyncPinecone, PineconeVectorStore]]:
    """Cliente de control (también da acceso al reranker hospedado) y el vector store del namespace."""
    async with AsyncPinecone(api_key=settings.pinecone_api_key.get_secret_value()) as pc:
        index = await ensure_index(pc, settings)
        async with index:
            yield pc, PineconeVectorStore(index, settings.namespace)
