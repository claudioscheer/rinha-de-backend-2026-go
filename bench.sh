#!/usr/bin/env bash
# bench.sh — run the official k6 contest test against a local docker-compose
# stack with the Rinha CPU/memory caps applied, and print the final score.
#
#   ./bench.sh                 # full cycle: build, up, /ready, k6, score, down
#   ./bench.sh --no-build      # reuse the existing rinha-2026-go-api:latest
#   ./bench.sh --keep-up       # leave the stack running after the test
#   ./bench.sh --quick         # skip k6, just verify constraints + /ready
#
# Requires: docker, docker compose, jq, curl. k6 is optional (falls back to
# the grafana/k6 docker image).

set -euo pipefail
cd "$(dirname "$0")"

KEEP_UP=0
NO_BUILD=0
QUICK=0
for arg in "$@"; do
    case "$arg" in
        --keep-up)  KEEP_UP=1 ;;
        --no-build) NO_BUILD=1 ;;
        --quick)    QUICK=1 ;;
        --help|-h)
            sed -n '2,12p' "$0"
            exit 0 ;;
        *)
            echo "unknown arg: $arg" >&2
            exit 2 ;;
    esac
done

need() { command -v "$1" >/dev/null || { echo "missing: $1" >&2; exit 1; }; }
need docker
need jq
need curl

if ! docker info >/dev/null 2>&1; then
    echo "docker daemon not reachable" >&2
    exit 1
fi

cleanup() {
    local rc=$?
    if [[ $KEEP_UP -eq 0 ]]; then
        echo "==> tearing down"
        docker compose down -v --remove-orphans >/dev/null 2>&1 || true
    else
        echo "==> stack left running (use 'docker compose down' to stop)"
    fi
    exit "$rc"
}
trap cleanup EXIT

if [[ $NO_BUILD -eq 0 ]]; then
    echo "==> docker compose build"
    docker compose build
fi

echo "==> docker compose up"
docker compose up -d --force-recreate

echo "==> verifying applied container limits"
for svc in api1 api2 lb; do
    out=$(docker inspect --format \
        '{{.Name}} cpus={{.HostConfig.NanoCpus}} mem_bytes={{.HostConfig.Memory}}' \
        "$(docker compose ps -q "$svc")" 2>/dev/null) || {
            echo "  $svc: container missing"; continue;
        }
    echo "  $out"
done

echo "==> waiting for /ready (up to 60s)"
ready=0
for i in $(seq 1 60); do
    if curl -fsS http://localhost:9999/ready >/dev/null 2>&1; then
        echo "  ready in ${i}s"
        ready=1
        break
    fi
    sleep 1
done
if [[ $ready -ne 1 ]]; then
    echo "  timed out — container logs follow:" >&2
    docker compose logs --tail=80
    exit 1
fi

if [[ $QUICK -eq 1 ]]; then
    echo "==> --quick: skipping k6, dumping current container memory:"
    docker stats --no-stream --format \
        'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}'
    exit 0
fi

echo "==> running k6 (test ramps to 900 RPS over ~120s; total ~140s)"
export K6_NO_USAGE_REPORT=true
if command -v k6 >/dev/null; then
    k6 run --quiet test/test.js
else
    echo "  k6 not installed locally; using grafana/k6:latest in docker"
    docker run --rm -i \
        --network=host \
        -e K6_NO_USAGE_REPORT=true \
        -v "$(pwd)/test:/test" \
        -w /test \
        grafana/k6:latest run --quiet test.js
fi

echo "==> post-test container snapshot"
docker stats --no-stream --format \
    'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}'

if [[ ! -s test/results.json ]]; then
    echo "  results.json missing or empty — k6 likely failed" >&2
    exit 1
fi

echo "==> raw results"
jq . test/results.json

FINAL=$(jq -r '.scoring.final_score' test/results.json)
DET=$(jq -r '.scoring.detection_score.value' test/results.json)
P99S=$(jq -r '.scoring.p99_score.value' test/results.json)
P99=$(jq -r '.p99' test/results.json)
FAIL=$(jq -r '.scoring.failure_rate' test/results.json)

cat <<EOF

=========================================================
  final_score:     $FINAL  (max 6000)
    detection:     $DET  (max 3000)
    p99 latency:   $P99S  (max 3000) — observed p99 = $P99
  failure rate:    $FAIL
=========================================================
EOF
