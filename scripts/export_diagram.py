"""Exporta el diagrama del grafo compilado a docs/graph.mmd (Mermaid) y docs/graph.png (usa mermaid.ink).

No necesita API keys: la topología no depende de los modelos, así que se compila con nodos vacíos.
Uso:  python -m scripts.export_diagram
"""
import logging

from app.config import BASE_DIR
from app.graph.builder import OrchestratorNodes, build_graph


async def _noop(_state: dict) -> dict:
    return {}


def main() -> None:
    out_dir = BASE_DIR / "docs"
    out_dir.mkdir(exist_ok=True)
    drawable = build_graph(OrchestratorNodes(_noop, _noop, _noop, _noop, _noop)).get_graph()
    (out_dir / "graph.mmd").write_text(drawable.draw_mermaid(), encoding="utf-8")
    print(f"Diagrama exportado: {out_dir / 'graph.mmd'}")
    try:
        (out_dir / "graph.png").write_bytes(drawable.draw_mermaid_png())
        print(f"Diagrama exportado: {out_dir / 'graph.png'}")
    except Exception as exc:  # noqa: BLE001 - el PNG es opcional; el .mmd ya documenta el grafo
        logging.warning("No se pudo generar el PNG del grafo: %s", exc)


if __name__ == "__main__":
    main()
