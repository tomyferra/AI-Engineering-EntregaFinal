"""Punto único de configuración: límites del orquestador y settings validados con Pydantic, leídos del entorno.

Nada que dependa del despliegue está escrito en el código: modelos, índice de Pinecone, namespace, URLs y
claves salen de variables de entorno (ver .env.example). Si falta una obligatoria, el proceso falla al arrancar
con un mensaje que dice cuál es.
"""
import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, SecretStr, model_validator

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
CATALOG_PATH = DATA_DIR / "catalog.json"

# --- Orquestador multi-agente ---------------------------------------------------------------------------------
# Condición de parada dura contra el "supervisor infinito": cantidad máxima de delegaciones a especialistas.
MAX_DELEGATIONS = 6
# Máximo de veces que se puede delegar en un mismo especialista (1 intento + refinamientos).
MAX_CALLS_PER_AGENT = 3
# Techo de seguridad de LangGraph (super-steps por invocación). Cada delegación consume 2 pasos
# (supervisor + especialista); el margen cubre síntesis, aprobación humana y ejecución de la acción.
RECURSION_LIMIT = 2 * MAX_DELEGATIONS + 8
# Techo de pasos del loop ReAct interno de cada especialista (modelo <-> tools).
SPECIALIST_RECURSION_LIMIT = 12
# Largo máximo del resultado de cada tool call que se guarda en el estado como evidencia.
TOOL_RESULT_MAX_CHARS = 300
# Turnos previos de la conversación que ven el supervisor y el sintetizador (memoria de corto plazo).
HISTORY_TURNS = 3

# --- Human-in-the-loop ----------------------------------------------------------------------------------------
# Destinatario por defecto cuando la consulta pide enviar el informe pero no indica un email.
DEFAULT_REPORT_RECIPIENT = "direccion@genpet.com.ar"


def _require_env(nombre: str) -> str:
    valor = os.getenv(nombre)
    if not valor:
        raise ValueError(f"Falta la variable de entorno {nombre} (ver .env.example)")
    return valor


def _env_bool(nombre: str, default: bool) -> bool:
    valor = os.getenv(nombre)
    return default if valor in (None, "") else valor.strip().lower() in ("1", "true", "yes", "si", "sí")


def _env_int(nombre: str, default: int) -> int:
    valor = os.getenv(nombre)
    return default if valor in (None, "") else int(valor)


# --- LLM (Módulos 1 y 2: clientes asíncronos con timeout, reintentos y rate limit) -----------------------------

class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"


ENV_VAR_POR_PROVIDER = {
    LLMProvider.OPENAI: "OPENAI_API_KEY",
    LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    LLMProvider.GEMINI: "GEMINI_API_KEY",
    LLMProvider.OPENROUTER: "OPENROUTER_API_KEY",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class LLMSettings(BaseModel):
    provider: LLMProvider
    model: str = Field(min_length=1)
    api_key: SecretStr
    temperature: float = Field(default=0.0, ge=0, le=2)
    # Tope de tokens de salida: sin él algunos proveedores reservan el máximo del modelo en cada request.
    max_tokens: int = Field(default=2048, gt=0)
    # Tope de requests por minuto compartido por todos los agentes y jobs (None = sin límite).
    requests_per_minute: Optional[int] = Field(default=None, gt=0)
    timeout_seconds: float = Field(default=60, gt=0)
    max_retries: int = Field(default=2, ge=0)

    @property
    def label(self) -> str:
        return f"{self.provider.value}:{self.model}"


def get_llm_settings() -> LLMSettings:
    """Settings del LLM de los agentes (llamar después de load_dotenv)."""
    provider = LLMProvider(_require_env("LLM_PROVIDER").lower())
    return LLMSettings(
        provider=provider,
        model=_require_env("LLM_MODEL"),
        api_key=SecretStr(_require_env(ENV_VAR_POR_PROVIDER[provider])),
        max_tokens=_env_int("LLM_MAX_TOKENS", 2048),
        requests_per_minute=_env_int("LLM_REQUESTS_PER_MINUTE", 0) or None,
        timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", 60),
        max_retries=_env_int("LLM_MAX_RETRIES", 2),
    )


def get_tavily_api_key() -> Optional[str]:
    """La búsqueda web es opcional: sin TAVILY_API_KEY el investigador usa solo la base de conocimiento."""
    return os.getenv("TAVILY_API_KEY") or None


# --- RAG (Módulos 3 y 4: recuperación híbrida BM25 + vectorial con metadatos) ---------------------------------

class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


ENV_VAR_POR_EMBEDDING_PROVIDER = {
    EmbeddingProvider.OPENAI: "OPENAI_API_KEY",
    EmbeddingProvider.GEMINI: "GEMINI_API_KEY",
}


class VectorStoreKind(str, Enum):
    PINECONE = "pinecone"  # producción: índice serverless administrado
    MEMORY = "memory"      # sin Pinecone: los mismos embeddings, indexados en memoria al arrancar
    NONE = "none"          # solo BM25 (sin embeddings): útil sin conexión o para comparar


class RAGSettings(BaseModel):
    vector_store: VectorStoreKind = VectorStoreKind.PINECONE
    embedding_provider: Optional[EmbeddingProvider] = None
    embedding_model: Optional[str] = None
    embedding_api_key: Optional[SecretStr] = None
    # gemini-embedding-001 se trunca a esta dimensión (MRL); text-embedding-3-small es nativo de 1536.
    dimension: int = Field(default=1536, gt=0)
    pinecone_api_key: Optional[SecretStr] = None
    index_name: Optional[str] = None
    namespace: str = "genpet-docs"
    cloud: str = "aws"
    region: str = "us-east-1"
    # Re-ranking con el modelo hospedado de Pinecone después de la fusión (vacío = sin re-ranking).
    rerank_model: Optional[str] = None
    # Si el namespace no tiene el corpus indexado, el worker lo ingesta al arrancar (idempotente).
    auto_ingest: bool = True
    top_k: int = Field(default=4, gt=0, le=10)
    candidates_k: int = Field(default=8, gt=0, le=50)
    # Pesos de la fusión: calibrados con scripts/evaluate_rag.py (ver README §6).
    bm25_weight: float = Field(default=0.2, ge=0)
    vector_weight: float = Field(default=0.8, ge=0)
    # Constante de Reciprocal Rank Fusion (60 es el valor del paper original).
    rrf_k: int = Field(default=60, gt=0)

    @model_validator(mode="after")
    def _check_backend(self) -> "RAGSettings":
        if self.vector_store != VectorStoreKind.NONE:
            faltantes = [n for n, v in (("EMBEDDING_PROVIDER", self.embedding_provider),
                                        ("EMBEDDING_MODEL", self.embedding_model),
                                        ("la API key de embeddings", self.embedding_api_key)) if not v]
            if faltantes:
                raise ValueError(f"VECTOR_STORE={self.vector_store.value} requiere {', '.join(faltantes)}")
        if self.vector_store == VectorStoreKind.PINECONE and not (self.pinecone_api_key and self.index_name):
            raise ValueError("VECTOR_STORE=pinecone requiere PINECONE_API_KEY y PINECONE_INDEX_NAME")
        return self


def get_rag_settings() -> RAGSettings:
    """Settings del RAG (llamar después de load_dotenv)."""
    provider = os.getenv("EMBEDDING_PROVIDER")
    embedding_provider = EmbeddingProvider(provider.lower()) if provider else None
    embedding_key = os.getenv(ENV_VAR_POR_EMBEDDING_PROVIDER[embedding_provider]) if embedding_provider else None
    pinecone_key = os.getenv("PINECONE_API_KEY")
    return RAGSettings(
        vector_store=VectorStoreKind((os.getenv("VECTOR_STORE") or VectorStoreKind.PINECONE.value).lower()),
        embedding_provider=embedding_provider,
        embedding_model=os.getenv("EMBEDDING_MODEL") or None,
        embedding_api_key=SecretStr(embedding_key) if embedding_key else None,
        dimension=_env_int("EMBEDDING_DIMENSION", 1536),
        pinecone_api_key=SecretStr(pinecone_key) if pinecone_key else None,
        index_name=os.getenv("PINECONE_INDEX_NAME") or None,
        namespace=os.getenv("PINECONE_NAMESPACE") or RAGSettings.model_fields["namespace"].default,
        cloud=os.getenv("PINECONE_CLOUD") or RAGSettings.model_fields["cloud"].default,
        region=os.getenv("PINECONE_REGION") or RAGSettings.model_fields["region"].default,
        rerank_model=os.getenv("RERANK_MODEL") or None,
        auto_ingest=_env_bool("RAG_AUTO_INGEST", True),
        bm25_weight=float(os.getenv("RAG_BM25_WEIGHT") or RAGSettings.model_fields["bm25_weight"].default),
        vector_weight=float(os.getenv("RAG_VECTOR_WEIGHT") or RAGSettings.model_fields["vector_weight"].default),
    )


# --- Infraestructura: Redis, worker y observabilidad (Módulo 7) -----------------------------------------------

class AppSettings(BaseModel):
    redis_url: str = "redis://localhost:6379/0"
    # Prefijo de las claves propias (jobs, cola, conversaciones, outbox). Los checkpoints usan las de RedisSaver.
    key_prefix: str = "genpet"
    # Jobs, conversaciones y checkpoints expiran a los 7 días (evita que Redis crezca sin límite).
    job_ttl_seconds: int = Field(default=7 * 24 * 3600, gt=0)
    # Worker: True = corre dentro del proceso de la API (lifespan); False = proceso aparte (python -m app.core.worker).
    run_worker_in_api: bool = True
    worker_concurrency: int = Field(default=5, gt=0)
    # Tiempo máximo de una ejecución del grafo antes de marcar el job como FAILED.
    job_timeout_seconds: int = Field(default=300, gt=0)
    phoenix_enabled: bool = True
    phoenix_collector_endpoint: str = "http://localhost:6006"
    phoenix_project_name: str = "genpet-intelligence"


def get_app_settings() -> AppSettings:
    defaults = AppSettings.model_fields
    return AppSettings(
        redis_url=os.getenv("REDIS_URL") or defaults["redis_url"].default,
        run_worker_in_api=_env_bool("RUN_WORKER_IN_API", True),
        worker_concurrency=_env_int("WORKER_CONCURRENCY", 5),
        job_timeout_seconds=_env_int("JOB_TIMEOUT_SECONDS", 300),
        phoenix_enabled=_env_bool("PHOENIX_ENABLED", True),
        phoenix_collector_endpoint=(os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
                                    or defaults["phoenix_collector_endpoint"].default),
        phoenix_project_name=os.getenv("PHOENIX_PROJECT_NAME") or defaults["phoenix_project_name"].default,
    )
