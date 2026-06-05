#!/usr/bin/env bash
# Build the frontend (if needed) and serve the whole HILO console — UI + API —
# from a single FastAPI server bound to 0.0.0.0 so it is reachable over the network.
#
#   ./serve.sh            # build if missing, then serve on 0.0.0.0:8765
#   ./serve.sh --rebuild  # force a fresh frontend build first
#   HILO_PORT=9000 ./serve.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HILO_HOST:-0.0.0.0}"
PORT="${HILO_PORT:-8765}"

if [[ "${1:-}" == "--rebuild" || ! -f "$HERE/frontend/dist/index.html" ]]; then
  echo "▶ building frontend…"
  ( cd "$HERE/frontend" && { [ -d node_modules ] || npm install; } && npm run build )
fi

echo "▶ serving HILO console on http://${HOST}:${PORT}/"
echo "  reach it from another machine at  http://<this-server-ip>:${PORT}/"
cd "$HERE/backend"
exec uvicorn app:app --host "$HOST" --port "$PORT"
