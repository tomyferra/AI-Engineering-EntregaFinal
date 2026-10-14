"""Ingesta manual del corpus en el vector store configurado (el worker ya la hace al arrancar si falta).

Uso:  python -m scripts.ingest            # idempotente: no hace nada si el namespace ya tiene el corpus
      python -m scripts.ingest --force    # vacía el namespace y reindexa (después de editar data/docs)
"""
import argparse
import asyncio

from dotenv import load_dotenv

from app.config import VectorStoreKind, get_rag_settings
from app.llm import build_embeddings
from app.observability import configure_logging
from app.rag.documents import load_corpus
from app.rag.knowledge_base import ingest
from app.rag.vector_store import open_pinecone


async def main(force: bool) -> None:
    settings = get_rag_settings()
    if settings.vector_store != VectorStoreKind.PINECONE:
        raise SystemExit(f"VECTOR_STORE={settings.vector_store.value}: solo Pinecone necesita ingesta previa")
    corpus = load_corpus()
    async with open_pinecone(settings) as (_pc, store):
        subidos = await ingest(corpus, store, build_embeddings(settings), force=force)
        print(f"{subidos} chunks subidos a {settings.index_name}/{settings.namespace} "
              f"({len(corpus.chunks)} chunks en el corpus)")


if __name__ == "__main__":
    load_dotenv()
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="vaciar el namespace y reindexar")
    asyncio.run(main(parser.parse_args().force))
