"""Configuración: todo sale del entorno y lo obligatorio falla temprano con un mensaje claro."""
import pytest
from pydantic import ValidationError

from app.config import (
    LLMProvider,
    VectorStoreKind,
    get_app_settings,
    get_llm_settings,
    get_rag_settings,
)
from app.llm import build_chat_model, build_rate_limiter

ENV_VARS = ["LLM_PROVIDER", "LLM_MODEL", "GEMINI_API_KEY", "OPENAI_API_KEY", "VECTOR_STORE", "EMBEDDING_PROVIDER",
            "EMBEDDING_MODEL", "PINECONE_API_KEY", "PINECONE_INDEX_NAME", "RERANK_MODEL", "LLM_REQUESTS_PER_MINUTE",
            "REDIS_URL", "RUN_WORKER_IN_API", "PHOENIX_ENABLED"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_llm_sin_modelo_falla_con_el_nombre_de_la_variable(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(ValueError, match="LLM_MODEL"):
        get_llm_settings()


def test_llm_desde_entorno(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "GEMINI")
    monkeypatch.setenv("LLM_MODEL", "modelo-x")
    monkeypatch.setenv("GEMINI_API_KEY", "secreto")
    monkeypatch.setenv("LLM_REQUESTS_PER_MINUTE", "14")
    settings = get_llm_settings()
    assert settings.provider == LLMProvider.GEMINI and settings.label == "gemini:modelo-x"
    assert settings.requests_per_minute == 14 and "secreto" not in repr(settings)  # SecretStr no se filtra a logs
    assert build_chat_model(settings).model.endswith("modelo-x")


def test_rate_limiter_opcional():
    assert build_rate_limiter(None) is None
    assert build_rate_limiter(60) is not None


def test_pinecone_exige_indice_y_embeddings(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE", "pinecone")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    with pytest.raises(ValidationError, match="PINECONE_INDEX_NAME"):
        get_rag_settings()

    monkeypatch.setenv("PINECONE_API_KEY", "p")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "mi-indice")
    settings = get_rag_settings()
    assert settings.index_name == "mi-indice" and settings.rerank_model is None


def test_modo_solo_bm25_no_necesita_claves(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE", "none")
    assert get_rag_settings().vector_store == VectorStoreKind.NONE


def test_memoria_exige_embeddings(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE", "memory")
    with pytest.raises(ValidationError, match="EMBEDDING_PROVIDER"):
        get_rag_settings()


def test_infraestructura_con_defaults(monkeypatch):
    monkeypatch.setenv("RUN_WORKER_IN_API", "false")
    monkeypatch.setenv("PHOENIX_ENABLED", "no")
    settings = get_app_settings()
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.run_worker_in_api is False and settings.phoenix_enabled is False
