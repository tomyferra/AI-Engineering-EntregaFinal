"""Evalúa la recuperación contra data/golden_set.json y compara estrategias: BM25, vectorial, híbrida y
híbrida + re-ranking (Hit@k y MRR). Usa el vector store y el reranker configurados en .env.

Uso:  python -m scripts.evaluate_rag [--k 4] [--rerank-model bge-reranker-v2-m3]
"""
import argparse
import asyncio
import json
from typing import Dict, List, Sequence

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

from app.config import BASE_DIR, DATA_DIR, VectorStoreKind, get_rag_settings
from app.llm import build_embeddings
from app.rag.documents import load_corpus
from app.rag.reranker import PineconeReranker
from app.rag.retriever import HybridRetriever, RetrievedChunk
from app.rag.vector_store import open_pinecone


class CachedEmbeddings(Embeddings):
    """Embebe cada consulta una sola vez para todas las estrategias (cuida la cuota del proveedor)."""

    def __init__(self, inner: Embeddings):
        self.inner = inner
        self._cache: Dict[str, List[float]] = {}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError("usar aembed_query")

    async def aembed_query(self, text: str) -> List[float]:
        if text not in self._cache:
            self._cache[text] = await self.inner.aembed_query(text)
        return self._cache[text]


def first_relevant_rank(results: Sequence[RetrievedChunk], relevant: List[List[str]]) -> int:
    """Posición (1-based) del primer fragmento relevante; 0 si no aparece."""
    wanted = {tuple(r) for r in relevant}
    for rank, chunk in enumerate(results, start=1):
        if (chunk.fuente, chunk.seccion) in wanted:
            return rank
    return 0


async def evaluate(retriever: HybridRetriever, golden: List[dict], k: int) -> Dict[str, float]:
    ranks = [first_relevant_rank(await retriever.search(item["query"], top_k=k), item["relevant"])
             for item in golden]
    return {"hit_at_k": round(sum(1 for r in ranks if r) / len(ranks), 3),
            "hit_at_1": round(sum(1 for r in ranks if r == 1) / len(ranks), 3),
            "mrr": round(sum(1 / r for r in ranks if r) / len(ranks), 3)}


async def main(k: int, out: str, rerank_model: str) -> None:
    settings = get_rag_settings()
    if settings.vector_store != VectorStoreKind.PINECONE:
        raise SystemExit("La comparación usa Pinecone: configurá VECTOR_STORE=pinecone")
    golden = json.loads((DATA_DIR / "golden_set.json").read_text(encoding="utf-8"))
    corpus = load_corpus()
    embeddings = CachedEmbeddings(build_embeddings(settings))
    async with open_pinecone(settings) as (pc, store):
        common = {"candidates_k": settings.candidates_k, "rrf_k": settings.rrf_k}
        weights = {"bm25_weight": settings.bm25_weight, "vector_weight": settings.vector_weight}
        strategies = {
            "bm25": HybridRetriever(corpus, **common),
            "vector": HybridRetriever(corpus, vector_store=store, embeddings=embeddings, bm25_weight=0, **common),
            "hybrid": HybridRetriever(corpus, vector_store=store, embeddings=embeddings, **weights, **common),
        }
        if rerank_model:
            strategies["hybrid+rerank"] = HybridRetriever(
                corpus, vector_store=store, embeddings=embeddings,
                reranker=PineconeReranker(pc, rerank_model), **weights, **common)
        report = {"k": k, "queries": len(golden), "weights": weights, "rerank_model": rerank_model,
                  "results": {}}
        for name, retriever in strategies.items():
            report["results"][name] = await evaluate(retriever, golden, k)
            print(f"{name:<14} {report['results'][name]}", flush=True)
    path = BASE_DIR / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Reporte: {path}")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--out", default="docs/evidence/rag_eval.json")
    parser.add_argument("--rerank-model", default="bge-reranker-v2-m3", help="vacío = no comparar con rerank")
    args = parser.parse_args()
    asyncio.run(main(args.k, args.out, args.rerank_model))
