"""RAG híbrido (Módulos 3 y 4): corpus con metadatos, BM25 + Pinecone, fusión RRF y re-ranking."""
from app.rag.documents import Chunk, Corpus, load_corpus
from app.rag.knowledge_base import KnowledgeBase, ingest, open_knowledge_base
from app.rag.retriever import HybridRetriever, RetrievedChunk

__all__ = ["Chunk", "Corpus", "HybridRetriever", "KnowledgeBase", "RetrievedChunk", "ingest", "load_corpus",
           "open_knowledge_base"]
