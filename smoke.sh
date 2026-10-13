#!/usr/bin/env bash
# Clean-environment smoke test: from a fresh checkout to a verified answer. Each stage prints PASS/FAIL; the first FAIL stops
# the run and the summary lists exactly which stages executed. Nothing is "ready" until its stage printed PASS.
#   ./smoke.sh            online mode (needs .env with PROVIDER + key)
#   ./smoke.sh offline    adds the Ollama smoke test (pulls qwen2.5:7b, ~4.7 GB)
#   ./smoke.sh --dry-run  print the stages without running them
set -uo pipefail
cd "$(dirname "$0")"
MODE="${1:-online}"; DRY=0; [ "$MODE" = "--dry-run" ] && { DRY=1; MODE=online; }
LOG=out/smoke.log; mkdir -p out
if ! : > "$LOG" 2>/dev/null; then   # Docker creates a missing bind-mount folder as root; the host user then cannot write to it
  echo "out/ is not writable by $(id -un). Fix: sudo chown -R $(id -u):$(id -g) out   then run ./smoke.sh again"; exit 1
fi
declare -a RESULTS
stage() { # name, command...
  local name="$1"; shift
  if [ "$DRY" = 1 ]; then echo "[dry-run] $name: $*"; RESULTS+=("DRY   $name"); return 0; fi
  echo "== $name"; echo "== $name: $*" >> "$LOG"
  if bash -c "$*" >> "$LOG" 2>&1; then echo "PASS  $name"; RESULTS+=("PASS  $name"); return 0
  else echo "FAIL  $name  (see $LOG)"; RESULTS+=("FAIL  $name"); summary; exit 1; fi
}
summary() { echo; echo "---- summary ($MODE) ----"; printf '%s\n' "${RESULTS[@]}"; echo "not listed above = NOT executed"; }

stage "0 prerequisites"      'command -v docker && docker compose version && test -f data/ar.pdf && test -f idx/manifest.json'
stage "1 docker build"       'docker compose build'
stage "2 real models (tcb check)" 'docker compose run --rm app tcb check'
stage "3 index validate (fail-closed)" 'docker compose run --rm app tcb validate /app/idx --snapshot'
stage "4 tests: unit+integration" 'docker compose run --rm app pytest -q'
stage "5 server up"          'docker compose up -d app && for i in $(seq 1 60); do curl -fsS http://127.0.0.1:8000/health && exit 0; sleep 3; done; exit 1'
stage "6 e2e answer + citation" "curl -fsS -X POST http://127.0.0.1:8000/chat -H 'content-type: application/json' -d '{\"question\":\"Tỷ lệ CASA của Techcombank năm 2025 là bao nhiêu?\"}' | tee out/smoke_chat.json | grep -q '40,4' && grep -q 'tr. 5' out/smoke_chat.json"
stage "7 e2e refusal"        "curl -fsS -X POST http://127.0.0.1:8000/chat -H 'content-type: application/json' -d '{\"question\":\"Techcombank dự báo lợi nhuận trước thuế năm 2027 là bao nhiêu?\"}' | tee out/smoke_refuse.json | grep -q 'not_in_report'"
stage "8 server down"        'docker compose down'
stage "9 eval on q10 (quality, informational)" 'docker compose run --rm app tcb eval data/q10.json --out out/runs --label smoke'
if [ "$MODE" = "offline" ]; then
  stage "10 ollama smoke"    'docker compose --profile offline run --rm app-offline tcb smoke --out out/smoke.json'
fi
summary
