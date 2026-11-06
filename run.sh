#!/usr/bin/env bash
# Usage: ./run.sh <local|cluster> <script> [extra args for the script]
#   local             runs with --master local[8]; HDFS paths still resolve via $HADOOP_CONF_DIR
#   cluster           uses spark.master from $SPARK_CONF_DIR/spark-defaults.conf
# Examples:
#   ./run.sh local src/yellow_taxi_trip_duration/main.py
#   ./run.sh cluster src/yellow_taxi_trip_duration/main.py
set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-local}"
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <local|cluster> <script> [extra args]" >&2
  exit 1
fi
SCRIPT="$2"
if [[ ! -f "$SCRIPT" ]]; then
  echo "Script not found: $SCRIPT" >&2
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

exec .venv/bin/spark-submit "${MASTER_ARGS[@]}" --name PA2 "$SCRIPT" "$@"
