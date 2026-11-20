"""Measure the existing taxi and weather preparation workflow on Spark Standalone."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from pyspark.sql import SparkSession, functions as F

from yellow_taxi.dataset import build_prepared


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxi", nargs="+", required=True, help="Raw taxi Parquet files or directories")
    parser.add_argument("--weather", required=True, help="Raw hourly weather PSV file or directory")
    parser.add_argument("--workers", required=True, type=int, help="Expected number of distinct executor hosts")
    parser.add_argument("--repeats", type=int, default=3, help="Timed runs after one warm-up")
    parser.add_argument("--output", type=Path, required=True, help="Write the completed measurements here")
    args = parser.parse_args()
    if args.workers < 1 or args.repeats < 1:
        parser.error("--workers and --repeats must be positive")
    return args


def executor_layout(spark):
    # Spark 4.1's application status store is also the source for the Executors UI.
    active = spark.sparkContext._jsc.sc().statusStore().executorList(True).iterator()
    executors = []
    while active.hasNext():
        executor = active.next()
        if executor.id() != "driver":
            host, _, _ = executor.hostPort().rpartition(":")
            executors.append({"host": host, "cores": executor.totalCores()})
    return sorted(executors, key=lambda executor: executor["host"])


def measure(spark, args):
    start = perf_counter()
    frame = build_prepared(spark, mode="train", raw_paths=args.taxi, weather_path=args.weather)
    # A plain count can let Spark prune features and the weather join.
    # Referencing every column forces materialization without timing output writes.
    result = frame.agg(
        F.count("*").alias("trip_rows"),
        F.sum("weather_missing").alias("trips_missing_weather"),
        F.bit_xor(F.xxhash64(*frame.columns)).alias("row_checksum"),
    ).first().asDict()
    elapsed = perf_counter() - start
    if result["trip_rows"] == 0:
        raise RuntimeError("Preparation produced no trips")
    return {"seconds": round(elapsed, 3), **result}


def main():
    args = parse_args()
    spark = SparkSession.builder.appName(f"taxi-prepare-benchmark-{args.workers}").getOrCreate()
    try:
        spark.sparkContext.setLogLevel("WARN")
        warmup = measure(spark, args)
        layout = executor_layout(spark)
        if len(layout) != args.workers or len({executor["host"] for executor in layout}) != args.workers:
            raise RuntimeError(f"Expected one executor on each of {args.workers} workers, got {layout}")
        print(f"Warm-up: {warmup} on {layout}", flush=True)

        runs = []
        keys = ("trip_rows", "trips_missing_weather", "row_checksum")
        for index in range(1, args.repeats + 1):
            run = measure(spark, args)
            if any(run[key] != warmup[key] for key in keys) or executor_layout(spark) != layout:
                raise RuntimeError(f"Output or executor layout changed in run {index}: {run}")
            runs.append({"trial": index, **run})
            print(f"Trial {index}: {run}", flush=True)

        result = {
            "spark_version": spark.version,
            "executor_memory": spark.sparkContext.getConf().get("spark.executor.memory"),
            "shuffle_partitions": spark.conf.get("spark.sql.shuffle.partitions"),
            "executors": layout,
            "taxi": args.taxi,
            "weather": args.weather,
            "warmup": warmup,
            "runs": runs,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
