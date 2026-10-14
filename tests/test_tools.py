"""Herramientas de los agentes: cómputo seguro, contratos Pydantic de entrada/salida y búsqueda en la KB."""
import pytest
from pydantic import ValidationError

from app.agents.validation import build_grounding_validator, extract_amounts
from app.tools import InMemoryOutbox, build_report_sender_tool
from app.tools.knowledge import build_knowledge_tools
from app.tools.math_tools import calcular, cumplimiento_de_meta, estadisticas_descriptivas, safe_eval, variacion_porcentual
from app.tools.schemas import (
    CalculateOutput,
    DocumentListOutput,
    GoalOutput,
    KnowledgeSearchOutput,
    SendReportOutput,
    StatsOutput,
    ToolError,
    VariationOutput,
)


def test_safe_eval_resuelve_aritmetica():
    assert safe_eval("(7931680 - 4427580) / 4427580 * 100") == pytest.approx(79.1415, rel=1e-4)


@pytest.mark.parametrize("expresion", ["__import__('os').system('dir')", "abs(-1)", "x + 1", "2 ** 1000"])
def test_safe_eval_rechaza_codigo(expresion):
    with pytest.raises(ValueError):
        safe_eval(expresion)


async def test_calcular_valida_salida_y_devuelve_error_accionable():
    assert CalculateOutput(**await calcular.ainvoke({"expresion": "2 * (3 + 4)"})).resultado == 14
    error = ToolError(**await calcular.ainvoke({"expresion": "$4.427.580 + 1"}))
    assert error.accion_sugerida


async def test_variacion_y_cumplimiento():
    variacion = VariationOutput(**await variacion_porcentual.ainvoke({"valor_inicial": 100, "valor_final": 150}))
    assert variacion.variacion_pct == 50.0
    assert "error" in await variacion_porcentual.ainvoke({"valor_inicial": 0, "valor_final": 10})
    meta = GoalOutput(**await cumplimiento_de_meta.ainvoke({"valor_real": 4427580, "meta": 4000000}))
    assert meta.cumplimiento_pct == 110.69 and meta.meta_alcanzada is True


async def test_estadisticas_descriptivas():
    resultado = StatsOutput(**await estadisticas_descriptivas.ainvoke({"valores": [1, 2, 3, 4]}))
    assert resultado.suma == 10 and resultado.promedio == 2.5 and resultado.mediana == 2.5


@pytest.mark.parametrize("tool, args", [
    (calcular, {"expresion": ""}),
    (calcular, {"expresion": "1" * 501}),
    (variacion_porcentual, {"valor_inicial": "mucho", "valor_final": 1}),
    (estadisticas_descriptivas, {"valores": []}),
])
async def test_entradas_invalidas_se_rechazan_antes_de_ejecutar(tool, args):
    """args_schema: LangChain valida con Pydantic antes de llamar a la función (el agente ve el error)."""
    with pytest.raises(ValidationError):
        await tool.ainvoke(args)


async def test_envio_de_informe_valida_destinatario_y_salida():
    outbox = InMemoryOutbox()
    enviar = build_report_sender_tool(outbox)
    with pytest.raises(ValidationError):
        await enviar.ainvoke({"destinatario": "no-es-email", "asunto": "a", "cuerpo": "b"})
    assert outbox.sent == []

    salida = SendReportOutput(**await enviar.ainvoke({"destinatario": "a@b.com", "asunto": "a", "cuerpo": "b"}))
    assert salida.recipient == "a@b.com" and len(outbox.sent) == 1


async def test_busqueda_devuelve_fragmentos_validados_con_fuente(kb):
    buscar = {t.name: t for t in build_knowledge_tools(kb)}["buscar_base_conocimiento"]
    salida = KnowledgeSearchOutput(**await buscar.ainvoke({"consulta": "metas trimestrales plan comercial"}))
    assert salida.modo == "bm25"
    assert salida.resultados[0].fuente == "plan_comercial_2026.md"
    assert "$12.000.000" in salida.resultados[0].contenido


async def test_busqueda_filtrada_por_categoria(kb):
    buscar = {t.name: t for t in build_knowledge_tools(kb)}["buscar_base_conocimiento"]
    salida = KnowledgeSearchOutput(**await buscar.ainvoke({"consulta": "trimestre", "categoria": "finanzas"}))
    assert salida.resultados and {r.categoria for r in salida.resultados} == {"finanzas"}

    error = ToolError(**await buscar.ainvoke({"consulta": "trimestre", "categoria": "deportes"}))
    assert "finanzas" in error.accion_sugerida


async def test_busqueda_sin_resultados_sugiere_accion(kb):
    buscar = {t.name: t for t in build_knowledge_tools(kb)}["buscar_base_conocimiento"]
    error = ToolError(**await buscar.ainvoke({"consulta": "zzzz qqqq"}))
    assert "listar_documentos" in error.accion_sugerida


async def test_listar_documentos_expone_metadatos(kb):
    listar = {t.name: t for t in build_knowledge_tools(kb)}["listar_documentos"]
    salida = DocumentListOutput(**await listar.ainvoke({}))
    assert "finanzas" in salida.categorias and len(salida.documentos) == 6
    plan = next(d for d in salida.documentos if d.fuente == "plan_comercial_2026.md")
    assert plan.categoria == "estrategia" and "Objetivos anuales" in plan.secciones


def test_extract_amounts_ignora_anios_y_porcentajes():
    assert extract_amounts("Q1 2026: $4.427.580 (79.14%) y 4427580") == ["4.427.580", "4427580"]


def test_validador_de_grounding(kb):
    validar = build_grounding_validator(kb.corpus.documents.values())
    assert validar("Ingresos Q3: $14.402.182; pack x20: 1.371 packs; meta: $36.000.000") == []
    assert validar("Objetivo anual: $36.500.000") == ["El monto 36.500.000 no aparece en ninguna fuente."]
