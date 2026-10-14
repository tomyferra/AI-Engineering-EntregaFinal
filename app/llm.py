"""Factories de clientes de modelos (evolución de los clientes del Módulo 1).

Cada proveedor se usa a través de su integración oficial de LangChain, que expone la interfaz asíncrona
(`ainvoke`, `astream`, `aembed_query`) sobre el SDK async del proveedor: ninguna llamada bloquea el event loop.
Timeout, reintentos con backoff y rate limit se configuran acá, una sola vez, para todos los agentes.
"""
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter

from app.config import OPENROUTER_BASE_URL, EmbeddingProvider, LLMProvider, LLMSettings, RAGSettings


def build_rate_limiter(requests_per_minute: Optional[int]) -> Optional[BaseRateLimiter]:
    """Token bucket compartido por todos los agentes y todos los jobs del worker (mismo modelo, misma cuota).

    Con varias ejecuciones concurrentes, las llamadas al LLM se espacian en vez de recibir 429 del proveedor.
    """
    if not requests_per_minute:
        return None
    return InMemoryRateLimiter(requests_per_second=requests_per_minute / 60, check_every_n_seconds=0.1,
                               max_bucket_size=1)


def build_chat_model(settings: LLMSettings) -> BaseChatModel:
    common = {"model": settings.model, "temperature": settings.temperature, "max_tokens": settings.max_tokens,
              "max_retries": settings.max_retries, "rate_limiter": build_rate_limiter(settings.requests_per_minute)}
    api_key = settings.api_key.get_secret_value()
    if settings.provider in (LLMProvider.OPENAI, LLMProvider.OPENROUTER):
        from langchain_openai import ChatOpenAI

        # OpenRouter expone una API compatible con OpenAI: mismo cliente, otra base_url.
        base_url = OPENROUTER_BASE_URL if settings.provider == LLMProvider.OPENROUTER else None
        return ChatOpenAI(api_key=api_key, base_url=base_url, timeout=settings.timeout_seconds, **common)
    if settings.provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(api_key=api_key, default_request_timeout=settings.timeout_seconds, **common)
    if settings.provider == LLMProvider.GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(google_api_key=api_key, timeout=settings.timeout_seconds, **common)
    raise ValueError(f"LLM provider no soportado: {settings.provider}")


def build_embeddings(settings: RAGSettings) -> Embeddings:
    """Modelo de embeddings del RAG, siempre con la dimensión del índice (evita el mismatch de dimensiones)."""
    if settings.embedding_api_key is None or settings.embedding_model is None:
        raise ValueError("Faltan EMBEDDING_MODEL o la API key del proveedor de embeddings")
    api_key = settings.embedding_api_key.get_secret_value()
    if settings.embedding_provider == EmbeddingProvider.OPENAI:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(api_key=api_key, model=settings.embedding_model, dimensions=settings.dimension)
    if settings.embedding_provider == EmbeddingProvider.GEMINI:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(google_api_key=api_key, model=settings.embedding_model,
                                            output_dimensionality=settings.dimension)
    raise ValueError(f"Embedding provider no soportado: {settings.embedding_provider}")
