"""Corpus del RAG: carga de los Markdown de GenPet, chunking por sección y metadatos del catálogo.

Cada chunk lleva metadatos (documento, sección, categoría y tags del catálogo) que viajan a Pinecone y permiten
filtrar la búsqueda por categoría: si la pregunta es de finanzas, no compiten fragmentos de políticas
comerciales. Los ids son deterministas (`<doc_id>-c<n>`), así reingestar sobrescribe en vez de duplicar.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from app.config import CATALOG_PATH, DOCS_DIR

CHUNK_MAX_CHARS = 1200
DEFAULT_CATEGORY = "general"


class DocumentMeta(BaseModel):
    doc_id: str
    source: str
    title: str
    categoria: str = DEFAULT_CATEGORY
    tags: List[str] = Field(default_factory=list)


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    doc_id: str
    source: str
    title: str
    section: str
    categoria: str
    tags: Tuple[str, ...] = ()
    text: str

    def metadata(self) -> dict:
        """Metadatos que se guardan junto al vector (Pinecone admite listas de strings para filtrar por tag)."""
        return {"doc_id": self.doc_id, "source": self.source, "title": self.title, "section": self.section,
                "categoria": self.categoria, "tags": list(self.tags), "text": self.text}


class Corpus(BaseModel):
    """Documentos completos (para validar montos contra las fuentes) y sus chunks (para recuperar)."""

    documents: Dict[str, str]
    metas: Dict[str, DocumentMeta]
    chunks: List[Chunk]
    _by_id: Dict[str, Chunk] = PrivateAttr(default_factory=dict)

    def model_post_init(self, _context: object) -> None:
        self._by_id = {c.id: c for c in self.chunks}

    @property
    def categories(self) -> List[str]:
        return sorted({m.categoria for m in self.metas.values()})

    def get(self, chunk_id: str) -> Optional[Chunk]:
        return self._by_id.get(chunk_id)


def _split_long(text: str, max_chars: int) -> List[str]:
    """Corta una sección larga en párrafos agrupados de hasta `max_chars` (nunca corta una fila de tabla)."""
    if len(text) <= max_chars:
        return [text]
    partes, actual = [], ""
    for parrafo in re.split(r"\n(?=\n|\|)", text):
        if actual and len(actual) + len(parrafo) > max_chars:
            partes.append(actual.strip())
            actual = ""
        actual += parrafo + "\n"
    if actual.strip():
        partes.append(actual.strip())
    return partes


def split_markdown(meta: DocumentMeta, text: str, max_chars: int = CHUNK_MAX_CHARS) -> List[Chunk]:
    """Un chunk por sección "## ..." (las tablas de cada sección quedan enteras). El título viaja en cada chunk."""
    chunks: List[Chunk] = []
    for bloque in re.split(r"(?=^## )", text, flags=re.MULTILINE):
        bloque = bloque.strip()
        if not bloque:
            continue
        primera_linea, *cuerpo = bloque.splitlines()
        if primera_linea.startswith("# ") and not "".join(cuerpo).strip():
            continue  # bloque que solo tiene el título del documento: no aporta contenido
        section = primera_linea[3:].strip() if primera_linea.startswith("## ") else "Introducción"
        for parte in _split_long(bloque, max_chars):
            chunks.append(Chunk(id=f"{meta.doc_id}-c{len(chunks)}", doc_id=meta.doc_id, source=meta.source,
                                title=meta.title, section=section, categoria=meta.categoria,
                                tags=tuple(meta.tags), text=parte))
    return chunks


def load_corpus(docs_dir: Path = DOCS_DIR, catalog_path: Path = CATALOG_PATH) -> Corpus:
    paths = sorted(docs_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No hay documentos .md en {docs_dir}")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {}
    documents, metas, chunks = {}, {}, []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        title_match = re.search(r"^# (.+)$", text, flags=re.MULTILINE)
        entry = catalog.get(path.stem, {})
        meta = DocumentMeta(doc_id=path.stem, source=path.name,
                            title=title_match.group(1).strip() if title_match else path.stem,
                            categoria=entry.get("categoria", DEFAULT_CATEGORY), tags=entry.get("tags", []))
        documents[path.name] = text
        metas[path.name] = meta
        chunks.extend(split_markdown(meta, text))
    return Corpus(documents=documents, metas=metas, chunks=chunks)
