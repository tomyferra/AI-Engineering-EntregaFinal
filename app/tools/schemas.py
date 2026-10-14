"""Contratos Pydantic de las herramientas de los agentes: entrada (args_schema) y salida.

La entrada la valida LangChain antes de ejecutar la tool: si el LLM manda un argumento inválido, el agente
recibe el error de validación como observación y puede corregirse. La salida se construye con estos modelos,
así cada observación que vuelve al LLM tiene siempre la misma forma.
"""
from typing import List, Optional

from pydantic import BaseModel, Field

from app.rag.retriever import RetrievedChunk

EMAIL_PATTERN = r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$"


class ToolError(BaseModel):
    """Salida de error uniforme: qué falló y qué puede hacer el agente para corregirlo."""

    error: str
    accion_sugerida: Optional[str] = None


# --- Base de conocimiento -------------------------------------------------------------------------------------

class KnowledgeSearchInput(BaseModel):
    consulta: str = Field(min_length=2, max_length=300, description="Consulta corta y específica")
    categoria: Optional[str] = Field(
        default=None, description="Filtra por categoría del documento (ver listar_documentos). Vacío = todas")


class KnowledgeSearchOutput(BaseModel):
    consulta: str
    categoria: Optional[str] = None
    modo: str = Field(description="Recuperación usada: hybrid, hybrid+rerank o bm25")
    resultados: List[RetrievedChunk]


class DocumentInfo(BaseModel):
    fuente: str
    titulo: str
    categoria: str
    tags: List[str]
    secciones: List[str]


class DocumentListOutput(BaseModel):
    categorias: List[str]
    documentos: List[DocumentInfo]


# --- Cómputo --------------------------------------------------------------------------------------------------

class CalculateInput(BaseModel):
    expresion: str = Field(min_length=1, max_length=500,
                           description='Expresión aritmética sin $ ni separadores de miles, ej. "4427580 + 7931680"')


class CalculateOutput(BaseModel):
    expresion: str
    resultado: float


class VariationInput(BaseModel):
    valor_inicial: float
    valor_final: float


class VariationOutput(BaseModel):
    valor_inicial: float
    valor_final: float
    diferencia: float
    variacion_pct: float


class GoalInput(BaseModel):
    valor_real: float
    meta: float


class GoalOutput(BaseModel):
    valor_real: float
    meta: float
    cumplimiento_pct: float
    diferencia: float
    meta_alcanzada: bool


class StatsInput(BaseModel):
    valores: List[float] = Field(min_length=1, max_length=1000, description="Serie de números")


class StatsOutput(BaseModel):
    n: int
    suma: float
    promedio: float
    mediana: float
    minimo: float
    maximo: float
    desvio_estandar: float


# --- Acción crítica -------------------------------------------------------------------------------------------

class SendReportInput(BaseModel):
    destinatario: str = Field(pattern=EMAIL_PATTERN)
    asunto: str = Field(min_length=1, max_length=200)
    cuerpo: str = Field(min_length=1)


class SendReportOutput(BaseModel):
    message_id: str
    recipient: str
    subject: str
    sent_at: str
