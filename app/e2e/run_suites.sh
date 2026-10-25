#!/usr/bin/env bash
# Run the browser suites against throwaway servers (fixture lessons, copied data). Real data and servers are untouched.
# Usage: app/e2e/run_suites.sh [suite ...]    suites: search questions articles ui demo python   (default: all)
# Needs: playwright-core on NODE_PATH (npm i playwright-core in a scratch dir), and Chromium at $CHROME.
set -euo pipefail
APP="$(cd "$(dirname "$0")/.." && pwd)"
SHOTS="${SHOTS:-$(mktemp -d)}"
API=8791; WEB=5191
D="$(mktemp -d)"; mkdir "$D/data"; cp "$APP/data/exercises.db" "$D/data/"; cp -r "$APP/data/models" "$D/data/models" 2>/dev/null || true
export PLACEMENT_DATA_DIR="$D/data" PLACEMENT_CONTENT_DIR="$APP/e2e/fixtures/content" PLACEMENT_MEDIA_DIR="$APP/e2e/fixtures/media"
cd "$APP/backend"
"$APP/.venv/bin/python" -m placement.articles >/dev/null
"$APP/.venv/bin/uvicorn" placement.main:app --host 127.0.0.1 --port $API >"$D/api.log" 2>&1 & API_PID=$!
cd "$APP/frontend"
PLACEMENT_API_PORT=$API PLACEMENT_WEB_PORT=$WEB setsid bash -c 'echo $$ >"$0"; exec node_modules/.bin/vite --host 127.0.0.1 --port "$PLACEMENT_WEB_PORT"' "$D/web.pid" >"$D/web.log" 2>&1 &
trap 'kill $API_PID 2>/dev/null; kill -- -"$(cat "$D/web.pid" 2>/dev/null)" 2>/dev/null; wait 2>/dev/null; rm -rf "$D"' EXIT
for i in $(seq 1 40); do curl -sf "http://127.0.0.1:$WEB/api/health" >/dev/null && break; sleep 0.5; done
export WEB_PORT=$WEB
SUITES=("$@"); [ ${#SUITES[@]} -eq 0 ] && SUITES=(questions articles ui demo search python)
STATUS=0
for s in "${SUITES[@]}"; do
  f="$APP/e2e/$s.mjs"; [ "$s" = questions ] && f="$APP/e2e/questions.mjs"
  out="$(node "$f" "$SHOTS" 2>&1 || true)"
  pass=$(grep -c '^PASS' <<<"$out" || true); fail=$(grep -c '^FAIL' <<<"$out" || true)
  echo "== $s: $pass pass, $fail fail"
  if [ "$fail" != 0 ] || ! grep -qE 'console(/page)? errors' <<<"$out"; then grep '^FAIL' <<<"$out" || true; echo "$out" | tail -25; STATUS=1; fi
done
echo "screenshots in $SHOTS"
exit $STATUS
