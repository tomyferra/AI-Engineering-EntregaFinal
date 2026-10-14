"""Pruebas de punta a punta contra la API levantada (docker compose up): cada una genera su traza en Phoenix.

Escenarios:
  1. rag         consulta documental pura (investigador -> sintetizador)
  2. calculo     multi-paso con cálculos (investigador -> analista -> sintetizador)
  3. metas       cumplimiento de metas vs. reales con guardas de consistencia
  4. repregunta  segundo turno en la conversación del escenario 2 (memoria del checkpointer)
  5. hitl        acción crítica: pausa en WAITING_APPROVAL, aprobación humana y ejecución

Uso:  python -m scripts.run_scenarios                      # los 5, en orden
      python -m scripts.run_scenarios --only rag calculo   # algunos (los resultados se acumulan en el JSON)
El escenario 4 retoma la conversación del 2 guardada en el JSON: se puede reiniciar el stack entre ambos
(`docker compose restart api worker`) para comprobar que la conversación sobrevive al reinicio.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import httpx

from app.config import BASE_DIR

OUT = BASE_DIR / "docs" / "evidence" / "scenarios.json"
TERMINAL = {"DONE", "FAILED"}

SCENARIOS = {
    "rag": {"query": "¿Qué descuento tiene un distribuidor, cuál es el pedido mínimo y cuánto tarda un envío a "
                     "CABA?"},
    "calculo": {"query": "¿Cuánto crecieron porcentualmente los ingresos netos del Q1 al Q2 y del Q2 al Q3 de "
                         "2026?"},
    "metas": {"query": "¿Se cumplieron las metas trimestrales de ingresos netos del plan comercial en Q1, Q2 y Q3? "
                       "¿Cuánto falta facturar en el Q4 para llegar al objetivo anual?"},
    "repregunta": {"query": "¿Y cómo evolucionó la ganancia neta en esos mismos trimestres?",
                   "continue": "calculo"},
    "hitl": {"query": "Prepará un resumen de las ventas por canal de 2026 y enviá el informe a "
                      "direccion@genpet.com.ar", "approve": True},
}


async def wait_for(client: httpx.AsyncClient, job_id: str, statuses: set, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = (await client.get(f"/tasks/{job_id}")).raise_for_status().json()
        if job["status"] in statuses:
            return job
        await asyncio.sleep(2)
    raise TimeoutError(f"El job {job_id} no llegó a {statuses} en {timeout}s")


async def run_scenario(client: httpx.AsyncClient, name: str, spec: dict, conversation_id: Optional[str],
                       timeout: float) -> dict:
    payload = {"query": spec["query"]}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    started = time.monotonic()
    accepted = (await client.post("/tasks", json=payload)).raise_for_status().json()
    print(f"[{name}] job {accepted['job_id']} (conversación {accepted['conversation_id']}, "
          f"crítica={accepted['critical']})")
    job = await wait_for(client, accepted["job_id"], TERMINAL | {"WAITING_APPROVAL"}, timeout)
    if job["status"] == "WAITING_APPROVAL" and spec.get("approve") is not None:
        print(f"[{name}] esperando aprobación: {job['approval_request']['question']}")
        decision = {"approved": spec["approve"], "reviewer": "run_scenarios", "comment": "aprobación de la prueba"}
        (await client.post(f"/tasks/{job['job_id']}/approve", json=decision)).raise_for_status()
        job = await wait_for(client, job["job_id"], TERMINAL, timeout)
    elapsed = round(time.monotonic() - started, 1)
    print(f"[{name}] {job['status']} en {elapsed}s · flujo: {' -> '.join(job.get('flow') or [])}")
    return {"scenario": name, "seconds": elapsed, **job}


async def main(names: list, base_url: str, timeout: float) -> None:
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        (await client.get("/health")).raise_for_status()
        for name in names:
            spec = SCENARIOS[name]
            conversation_id = None
            if "continue" in spec:
                previous = results.get(spec["continue"])
                if previous is None:
                    raise SystemExit(f"'{name}' continúa la conversación de '{spec['continue']}': corrélo antes")
                conversation_id = previous["conversation_id"]
            results[name] = await run_scenario(client, name, spec, conversation_id, timeout)
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    failed = [n for n in names if results[n]["status"] != "DONE"]
    print(f"Resultados: {Path(OUT)}" + (f" · FALLARON: {failed}" if failed else " · todos DONE"))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    asyncio.run(main(args.only, args.base_url, args.timeout))
