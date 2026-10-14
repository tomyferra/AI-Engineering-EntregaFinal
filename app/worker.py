"""Entrada del worker como proceso aparte: `python -m app.worker` (servicio `worker` de docker-compose.yml)."""
import asyncio
import sys
from contextlib import suppress

from dotenv import load_dotenv

from app.core.runtime import run_standalone_worker
from app.observability import configure_logging

if __name__ == "__main__":
    load_dotenv()
    sys.stdout.reconfigure(encoding="utf-8")
    configure_logging()
    with suppress(KeyboardInterrupt):
        asyncio.run(run_standalone_worker())
