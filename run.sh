#!/usr/bin/env bash
# Levanta el sistema completo con una sola instrucción: ./run.sh
#   ./run.sh          -> construye y levanta redis + phoenix + api + worker, espera a que la API responda
#   ./run.sh test     -> corre los tests dentro del contenedor (sin API keys ni servicios externos)
#   ./run.sh down     -> detiene el stack (los datos de Redis y Phoenix quedan en sus volúmenes)
set -euo pipefail
cd "$(dirname "$0")"

if ! docker compose version >/dev/null 2>&1; then
  echo "Se necesita Docker con el plugin 'docker compose'." >&2
  exit 1
fi

case "${1:-up}" in
  down)
    docker compose down
    exit 0
    ;;
  test)
    [ -f .env ] || cp .env.example .env
    docker compose build api
    docker compose run --rm --no-deps -e PHOENIX_ENABLED=false api python -m pytest -q
    exit 0
    ;;
  up) ;;
  *)
    echo "Uso: ./run.sh [up|test|down]" >&2
    exit 1
    ;;
esac

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Se creó .env a partir de .env.example: completá GEMINI_API_KEY y PINECONE_API_KEY y volvé a correr ./run.sh"
  exit 1
fi

docker compose up -d --build

echo -n "Esperando a la API"
for _ in $(seq 1 60); do
  if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
    echo " lista."
    echo "  API (Swagger):  http://localhost:8000/docs"
    echo "  Trazas Phoenix: http://localhost:6006"
    echo "  Logs worker:    docker compose logs -f worker"
    exit 0
  fi
  echo -n "."
  sleep 2
done
echo
echo "La API no respondió a tiempo. Revisá: docker compose logs api worker" >&2
exit 1
