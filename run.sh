#!/usr/bin/env bash
# Usage: ./run.sh <local|cluster> <job> [extra args for the job]
#   local             runs with --master local[8]; HDFS paths still resolve via $HADOOP_CONF_DIR
#   cluster           uses spark.master from $SPARK_CONF_DIR/spark-defaults.conf
#   <job>             a file in src/yellow_taxi/jobs/ (the .py suffix is optional)
# Examples:
#   ./run.sh local prepare.py                  # train: /yellow-taxi/raw/taxi/2024-25 -> /yellow-taxi/prepared/train/
#   ./run.sh cluster prepare.py --mode test    # test: /yellow-taxi/raw/taxi/2022, 2023, 2026 -> /yellow-taxi/prepared/test/year=*/
set -euo pipefail

cd "$(dirname "$0")"

JOBS_DIR="src/yellow_taxi/jobs"

list_jobs() {
  for f in "$JOBS_DIR"/*.py; do
    [[ "$(basename "$f")" == "__init__.py" ]] || echo "  $(basename "$f")"
  done
}

MODE="${1:-local}"
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <local|cluster> <job> [extra args]" >&2
  echo "Available jobs:" >&2
  list_jobs >&2
  exit 1
fi
JOB="${2%.py}.py"
SCRIPT="$JOBS_DIR/$JOB"
if [[ ! -f "$SCRIPT" ]]; then
  echo "Job not found: $SCRIPT" >&2
  echo "Available jobs:" >&2
  list_jobs >&2
  exit 1
fi
shift 2

case "$MODE" in
  local)   MASTER_ARGS=(--master "local[8]") ;;
  cluster) MASTER_ARGS=() ;;
  *) echo "Unknown mode '$MODE' (use local or cluster)" >&2; exit 1 ;;
esac

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYSPARK_PYTHON="$PWD/.venv/bin/python"

exec .venv/bin/spark-submit "${MASTER_ARGS[@]}" --name "$(basename "$SCRIPT" .py)" "$SCRIPT" "$@"
