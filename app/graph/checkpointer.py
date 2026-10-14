"""Checkpointer de LangGraph sobre Redis: el estado de cada conversación sobrevive a reinicios del servidor."""
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.redis.aio import AsyncRedisSaver


@asynccontextmanager
async def redis_checkpointer(redis_url: str, ttl_seconds: int) -> AsyncIterator[AsyncRedisSaver]:
    """AsyncRedisSaver (requiere Redis Stack: RedisJSON + RediSearch). Los checkpoints expiran como los jobs."""
    ttl = {"default_ttl": max(1, ttl_seconds // 60), "refresh_on_read": True}  # default_ttl va en minutos
    async with AsyncRedisSaver.from_conn_string(redis_url, ttl=ttl) as saver:  # __aenter__ crea los índices
        yield saver
