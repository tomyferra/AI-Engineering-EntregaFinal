"""Tools del investigador sobre la base de conocimiento: el RAG híbrido expuesto como herramienta del agente."""
from typing import List, Optional

from langchain_core.tools import BaseTool, tool

from app.observability import retrieval_span
from app.rag.knowledge_base import KnowledgeBase
from app.tools.schemas import (
    DocumentInfo,
    DocumentListOutput,
    KnowledgeSearchInput,
    KnowledgeSearchOutput,
    ToolError,
)


def build_knowledge_tools(kb: KnowledgeBase) -> List[BaseTool]:
    """La KB se inyecta por closure: no viaja en el estado del grafo ni se serializa en el checkpoint."""
    categorias = kb.corpus.categories

    @tool(args_schema=KnowledgeSearchInput)
    async def buscar_base_conocimiento(consulta: str, categoria: Optional[str] = None) -> dict:
        """Busca en los documentos internos de GenPet (ventas, resultados financieros, plan comercial, políticas,
        productos y datos de la empresa) y devuelve los fragmentos más relevantes con su fuente.

        Usar consultas cortas y específicas, por ejemplo "ingresos netos por trimestre" o "metas trimestrales
        plan comercial". Si se conoce el tema, filtrar por categoría reduce el ruido. Si los resultados no
        alcanzan, reformular la consulta con otras palabras clave.
        """
        if categoria and categoria not in categorias:
            return ToolError(error=f"Categoría inexistente: {categoria}",
                             accion_sugerida=f"Usar una de {categorias} o dejarla vacía.").model_dump()
        with retrieval_span(consulta, kb.retriever.mode, categoria) as record:
            resultados = await kb.retriever.search(consulta, categoria=categoria)
            record(resultados)
        if not resultados:
            return ToolError(error="Sin resultados para la consulta.",
                             accion_sugerida="Reformular con otras palabras clave, quitar el filtro de categoría "
                                             "o usar listar_documentos.").model_dump()
        return KnowledgeSearchOutput(consulta=consulta, categoria=categoria, modo=kb.retriever.mode,
                                     resultados=resultados).model_dump()

    @tool
    async def listar_documentos() -> dict:
        """Lista los documentos de la base de conocimiento con su categoría, tags y secciones."""
        secciones: dict = {}
        for chunk in kb.corpus.chunks:
            secciones.setdefault(chunk.source, [])
            if chunk.section not in secciones[chunk.source]:
                secciones[chunk.source].append(chunk.section)
        documentos = [DocumentInfo(fuente=m.source, titulo=m.title, categoria=m.categoria, tags=m.tags,
                                   secciones=secciones.get(m.source, [])) for m in kb.corpus.metas.values()]
        return DocumentListOutput(categorias=categorias, documentos=documentos).model_dump()

    return [buscar_base_conocimiento, listar_documentos]
