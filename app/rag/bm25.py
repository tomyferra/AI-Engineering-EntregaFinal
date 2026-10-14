"""Recuperación léxica BM25 en memoria: el complemento de los embeddings para términos exactos.

En este corpus abundan términos que un embedding no siempre distingue ("Q2", "x50", "septiembre", montos):
BM25 los encuentra por coincidencia literal. Es CPU pura sobre ~30 chunks (microsegundos), pero igual se
ejecuta en un thread (`asyncio.to_thread`) desde el retriever para no competir con el event loop.
"""
import re
import unicodedata
from typing import List, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi

from app.rag.documents import Chunk

_STOPWORDS = {
    "a", "al", "con", "cual", "cuales", "cuanto", "de", "del", "el", "en", "es", "esta", "la", "las", "lo",
    "los", "para", "por", "que", "se", "su", "sus", "un", "una", "y", "o", "como", "fue", "son",
}
# Stemming liviano por prefijo: "objetivo/objetivos" -> "objet", "anual/anuales" -> "anual".
STEM_LENGTH = 5


def _stem(token: str) -> str:
    return token[:STEM_LENGTH] if token.isalpha() else token


def tokenize(texto: str) -> List[str]:
    """Minúsculas, sin acentos, sin stopwords y con stemming: "Objetivos anuales" matchea "objetivo anual"."""
    sin_acentos = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return [_stem(t) for t in re.findall(r"\w+", sin_acentos.lower()) if t not in _STOPWORDS]


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk]):
        self._chunks = list(chunks)
        self._bm25 = BM25Okapi([tokenize(f"{c.title} {c.section} {c.text}") for c in self._chunks])

    def search(self, query: str, k: int, categoria: Optional[str] = None) -> List[Tuple[str, float]]:
        """(chunk_id, score) de los k mejores con score > 0, opcionalmente filtrados por categoría."""
        scores = self._bm25.get_scores(tokenize(query))
        ranking = sorted(range(len(self._chunks)), key=lambda i: scores[i], reverse=True)
        hits = [(self._chunks[i].id, float(scores[i])) for i in ranking
                if scores[i] > 0 and (categoria is None or self._chunks[i].categoria == categoria)]
        return hits[:k]
