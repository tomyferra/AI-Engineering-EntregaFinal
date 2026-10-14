"""Re-ranking de los candidatos fusionados con un cross-encoder hospedado (Pinecone Inference).

La fusión híbrida decide QUÉ candidatos entran; el reranker lee consulta y fragmento juntos y decide el ORDEN.
"""
from typing import List, Protocol, Sequence, Tuple

from pinecone import AsyncPinecone

Ranked = Tuple[int, float]  # (posición del documento en la lista de entrada, relevancia)


class Reranker(Protocol):
    async def rerank(self, query: str, documents: Sequence[str], top_n: int) -> List[Ranked]: ...


class PineconeReranker:
    def __init__(self, pc: AsyncPinecone, model: str):
        self._pc = pc
        self.model = model

    async def rerank(self, query: str, documents: Sequence[str], top_n: int) -> List[Ranked]:
        result = await self._pc.inference.rerank(model=self.model, query=query, documents=list(documents),
                                                 top_n=top_n, return_documents=False)
        return [(row.index, float(row.score)) for row in result.data]
