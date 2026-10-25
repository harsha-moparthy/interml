#!/usr/bin/env bash
# Start, stop, or check the learning site.
#   app/serve.sh start    start the service (127.0.0.1:8765) and the web page (127.0.0.1:5173) in the background
#   app/serve.sh stop     stop both
#   app/serve.sh status   show which one is running
#   app/serve.sh logs     follow both logs
# Then open http://127.0.0.1:5173/ . Port 8000 belongs to another app and is never used.
set -euo pipefail
APP="$(cd "$(dirname "$0")" && pwd)"
RUN="$APP/.run"
API_PORT="${PLACEMENT_API_PORT:-8765}"
WEB_PORT="${PLACEMENT_WEB_PORT:-5173}"
mkdir -p "$RUN"

alive() { [ -f "$RUN/$1.pid" ] && kill -0 "$(cat "$RUN/$1.pid")" 2>/dev/null; }

start_one() {  # name, directory, command...
  local name="$1" dir="$2"; shift 2
  if alive "$name"; then echo "$name already running (pid $(cat "$RUN/$name.pid"))"; return; fi
  # setsid gives the server its own process group; the shell inside writes its own pid, then becomes the server.
  ( cd "$dir" && setsid bash -c 'echo $$ >"$0"; exec "$@"' "$RUN/$name.pid" "$@" >"$RUN/$name.log" 2>&1 & )
  for _ in $(seq 1 20); do [ -s "$RUN/$name.pid" ] && break; sleep 0.1; done
}

stop_one() {
  local name="$1"
  if alive "$name"; then
    local pid; pid="$(cat "$RUN/$name.pid")"
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true   # the whole process group
    echo "$name stopped"
  else
    echo "$name not running"
  fi
  rm -f "$RUN/$name.pid"
}

case "${1:-}" in
  start)
    [ -f "$APP/data/exercises.db" ] || { echo "No exercises.db. Run the import first (see app/README.md)." >&2; exit 1; }
    [ -d "$APP/frontend/node_modules" ] || { echo "Run 'npm install' in app/frontend first." >&2; exit 1; }
    start_one api "$APP/backend" "$APP/.venv/bin/uvicorn" placement.main:app --host 127.0.0.1 --port "$API_PORT"
    PLACEMENT_API_PORT="$API_PORT" PLACEMENT_WEB_PORT="$WEB_PORT" \
      start_one web "$APP/frontend" "$APP/frontend/node_modules/.bin/vite" --host 127.0.0.1 --port "$WEB_PORT"
    for _ in $(seq 1 40); do
      curl -sf "http://127.0.0.1:$WEB_PORT/api/health" >/dev/null 2>&1 && { echo "Ready: http://127.0.0.1:$WEB_PORT/"; exit 0; }
      sleep 0.5
    done
    echo "Did not become ready. See $RUN/api.log and $RUN/web.log" >&2; exit 1 ;;
  stop)   stop_one web; stop_one api ;;
  status)
    for n in api web; do alive "$n" && echo "$n running (pid $(cat "$RUN/$n.pid"))" || echo "$n stopped"; done ;;
  logs)   tail -n 30 -f "$RUN/api.log" "$RUN/web.log" ;;
  *)      echo "Usage: app/serve.sh start|stop|status|logs" >&2; exit 2 ;;
esac
