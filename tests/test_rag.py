"""RAG híbrido: chunking con metadatos, fusión RRF, búsqueda vectorial, re-ranking, degradación e ingesta."""
import pytest

from app.config import RAGSettings, VectorStoreKind
from app.rag.bm25 import tokenize
from app.rag.documents import DocumentMeta, split_markdown
from app.rag.knowledge_base import ingest, open_knowledge_base
from app.rag.retriever import HybridRetriever, reciprocal_rank_fusion
from app.rag.vector_store import InMemoryVectorStore
from tests.conftest import KeywordEmbeddings


async def hybrid_retriever(corpus, **kwargs) -> HybridRetriever:
    store, embeddings = InMemoryVectorStore(), KeywordEmbeddings()
    await ingest(corpus, store, embeddings)
    return HybridRetriever(corpus, vector_store=store, embeddings=embeddings, **kwargs)


class FlakyStore(InMemoryVectorStore):
    async def query(self, *args, **kwargs):
        raise ConnectionError("Pinecone no responde")


class ReverseReranker:
    async def rerank(self, query, documents, top_n):
        return [(i, 1.0 - i / 10) for i in reversed(range(len(documents)))][:top_n]


class BrokenReranker:
    async def rerank(self, query, documents, top_n):
        raise TimeoutError("rerank timeout")


# --- Corpus y chunking ----------------------------------------------------------------------------------------

def test_tokenize_normaliza_acentos_stopwords_y_plurales():
    assert tokenize("La Facturación del Trimestre") == ["factu", "trime"]
    assert tokenize("Objetivos anuales") == tokenize("objetivo anual")


def test_split_markdown_por_seccion_con_metadatos():
    meta = DocumentMeta(doc_id="x", source="x.md", title="Título", categoria="ventas", tags=["q1"])
    chunks = split_markdown(meta, "# Título\nIntro\n## Ventas\nA\n## Metas\nB")
    assert [c.section for c in chunks] == ["Introducción", "Ventas", "Metas"]
    assert [c.id for c in chunks] == ["x-c0", "x-c1", "x-c2"]  # ids deterministas: reingestar no duplica
    assert chunks[1].metadata()["categoria"] == "ventas" and chunks[1].metadata()["tags"] == ["q1"]


def test_secciones_largas_se_parten_sin_cortar_tablas():
    meta = DocumentMeta(doc_id="x", source="x.md", title="T")
    tabla = "\n".join(f"| fila {i} | {i * 1000} |" for i in range(40))
    chunks = split_markdown(meta, f"## Datos\n{tabla}", max_chars=300)
    assert len(chunks) > 1 and all(len(c.text) <= 300 for c in chunks)
    assert all(line.startswith("|") or line.startswith("##") for c in chunks for line in c.text.splitlines())


def test_corpus_carga_categorias_del_catalogo(corpus):
    assert len(corpus.documents) == 6 and len(corpus.chunks) >= 20
    assert corpus.categories == ["comercial", "empresa", "estrategia", "finanzas", "producto", "ventas"]
    assert len({c.id for c in corpus.chunks}) == len(corpus.chunks)


# --- Fusión y recuperación ------------------------------------------------------------------------------------

def test_rrf_premia_lo_que_encuentran_ambos_recuperadores():
    fused = reciprocal_rank_fusion({"bm25": [("a", 9.0), ("b", 5.0)], "vector": [("b", 0.9), ("c", 0.8)]},
                                   {"bm25": 0.5, "vector": 0.5}, k=60)
    assert [cid for cid, _, _ in fused] == ["b", "a", "c"]
    assert fused[0][2] == ["bm25", "vector"]


async def test_hibrido_combina_bm25_y_vectorial(corpus):
    retriever = await hybrid_retriever(corpus)
    resultados = await retriever.search("objetivo anual de ingresos")
    assert retriever.mode == "hybrid"
    assert resultados[0].seccion == "Objetivos anuales" and "$36.000.000" in resultados[0].contenido
    assert set(resultados[0].recuperado_por) == {"bm25", "vector"}


async def test_filtro_por_categoria_aplica_a_ambos_recuperadores(corpus):
    retriever = await hybrid_retriever(corpus)
    resultados = await retriever.search("ingresos del trimestre", categoria="estrategia")
    assert resultados and {r.fuente for r in resultados} == {"plan_comercial_2026.md"}


async def test_falla_vectorial_degrada_a_bm25(corpus):
    embeddings = KeywordEmbeddings()
    store = FlakyStore()
    retriever = HybridRetriever(corpus, vector_store=store, embeddings=embeddings)
    resultados = await retriever.search("metas trimestrales plan comercial")
    assert resultados and all(r.recuperado_por == ["bm25"] for r in resultados)


async def test_reranker_reordena_y_su_falla_conserva_el_orden_rrf(corpus):
    base = await hybrid_retriever(corpus)
    orden_rrf = [r.chunk_id for r in await base.search("ventas por canal", top_k=8)]

    reranked = await hybrid_retriever(corpus, reranker=ReverseReranker())
    assert reranked.mode == "hybrid+rerank"
    assert [r.chunk_id for r in await reranked.search("ventas por canal", top_k=3)] == orden_rrf[::-1][:3]

    roto = await hybrid_retriever(corpus, reranker=BrokenReranker())
    assert [r.chunk_id for r in await roto.search("ventas por canal", top_k=3)] == orden_rrf[:3]


def test_vector_store_y_embeddings_van_juntos(corpus):
    with pytest.raises(ValueError):
        HybridRetriever(corpus, vector_store=InMemoryVectorStore())


# --- Ingesta y ciclo de vida ----------------------------------------------------------------------------------

async def test_ingesta_idempotente(corpus):
    store, embeddings = InMemoryVectorStore(), KeywordEmbeddings()
    assert await ingest(corpus, store, embeddings) == len(corpus.chunks)
    assert await ingest(corpus, store, embeddings) == 0  # ya estaba todo: no vuelve a embeber
    assert embeddings.calls == 1
    assert await ingest(corpus, store, embeddings, force=True) == len(corpus.chunks)
    assert await store.count() == len(corpus.chunks)


async def test_knowledge_base_en_memoria_y_solo_bm25(corpus):
    settings = RAGSettings(vector_store=VectorStoreKind.MEMORY, embedding_provider="gemini",
                           embedding_model="fake", embedding_api_key="x")
    async with open_knowledge_base(settings, corpus, embeddings=KeywordEmbeddings()) as kb:
        assert kb.retriever.mode == "hybrid" and await kb.retriever.vector_store.count() == len(corpus.chunks)

    async with open_knowledge_base(RAGSettings(vector_store=VectorStoreKind.NONE), corpus) as kb:
        assert kb.retriever.mode == "bm25"
