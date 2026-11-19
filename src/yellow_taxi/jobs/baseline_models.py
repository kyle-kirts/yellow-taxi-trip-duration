import argparse
import json
from pathlib import Path
from time import perf_counter

from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F

from yellow_taxi.dataset import TEST_PATH, TRAIN_PATH, load_prepared

SEED = 42


def add_data_arguments(parser):
    parser.add_argument("--cleaned", default=TRAIN_PATH, help="Prepared training root with fold partitions")
    parser.add_argument("--sample", type=float, default=0.01, help="Fraction of each dataset to use (1 = all trips)")
    parser.add_argument("--test-path", default=TEST_PATH, help="Separate prepared test root")
    parser.add_argument("--test-year", type=int, help="Score one prepared test year (default: all available years)")


def validate_data_arguments(parser, args):
    if not 0 < args.sample <= 1:
        parser.error("--sample must be in (0, 1]")
    if args.test_year is not None and not 1 <= args.test_year <= 9999:
        parser.error("--test-year must be a valid year")


def read_model_trips(spark, args, mode):
    """Read one prepared root on demand so selection can finish before opening test."""
    path = args.cleaned if mode == "train" else args.test_path
    trips = load_prepared(spark, mode=mode, path=path)
    partition = "fold" if mode == "train" else "year"
    if partition not in trips.columns:
        raise ValueError(f"Expected '{partition}' in prepared {mode} root {path}")
    if mode == "test" and args.test_year is not None:
        trips = trips.filter(F.col("year") == args.test_year)
    if args.sample < 1:
        trips = trips.sample(False, args.sample, seed=SEED)
    return trips


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    add_data_arguments(parser)
    parser.add_argument("--output", type=Path, help="Local JSON results file")
    args = parser.parse_args(argv)
    validate_data_arguments(parser, args)
    if args.output is None:
        args.output = Path("results") / f"baseline_prepared_test_{args.test_year or 'all'}.json"
    return args


def run(spark, args):
    train = read_model_trips(spark, args, "train")
    test = read_model_trips(spark, args, "test")
    return {
        "training_input": args.cleaned, "test_path": args.test_path, "test_year": args.test_year,
        "evaluation_split": "external_test",
        "sample_fraction": args.sample,
        **evaluate_baselines(train, test)
    }


def evaluate_baselines(train, test, *, round_errors=True):
    # STEP 3: Compute training statistics
    started = perf_counter()
    stats = (
        train.rollup("PULocationID")
        .agg(
            F.grouping("PULocationID").alias("is_total"),
            F.count("*").alias("n"),
            F.avg("trip_duration_min").alias("mean_min"),
            F.expr(
                "percentile_approx("
                "trip_duration_min, 0.5, 10000)"
            ).alias("median_min")
        )
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    try:
        stats.count()
        overall = stats.filter(
            F.col("is_total") == 1
        ).first()

        if overall is None or overall["n"] == 0:
            raise ValueError(
                "No training rows; check the prepared training input and sample size"
            )

        fit_seconds = perf_counter() - started
        overall_mean = float(overall["mean_min"])
        overall_median = float(overall["median_min"])

        # STEP 4: Compute mean duration for each pickup zone
        zone_means = (
            stats.filter(
                F.col("is_total") == 0
            )
            .select(
                "PULocationID",
                F.col("mean_min").alias("zone_mean")
            )
        )

        # STEP 5: Generate predictions
        predictions = (
            test.join(
                F.broadcast(zone_means),
                on="PULocationID",
                how="left"
            )
            .withColumn(
                "global_mean",
                F.lit(overall_mean)
            )
            .withColumn(
                "global_median",
                F.lit(overall_median)
            )
            .withColumn(
                "pickup_zone_mean",
                F.coalesce(
                    F.col("zone_mean"),
                    F.lit(overall_mean)
                )
            )
        )

        # STEP 6: Evaluate prediction errors
        expressions = [
            F.count("*").alias("test_rows"),
            F.sum(
                F.when(
                    F.col("zone_mean").isNull(), 1
                ).otherwise(0)
            ).alias("unseen_zone_rows")
        ]

        model_names = [
            "global_mean",
            "global_median",
            "pickup_zone_mean"
        ]

        for name in model_names:
            error = (
                F.col("trip_duration_min") - F.col(name)
            )

            expressions.extend([
                F.avg(
                    F.abs(error)
                ).alias(f"{name}_mae"),

                F.sqrt(
                    F.avg(error * error)
                ).alias(f"{name}_rmse"),
                F.sum(
                    (F.abs(error) <= 5.0).cast("long")
                ).alias(f"{name}_within_5_minutes")
            ])

        result = (
            predictions.agg(*expressions)
            .first()
            .asDict()
        )

        if result["test_rows"] == 0:
            raise ValueError(
                "No test rows; check the evaluation input and sample size"
            )

        # STEP 7: Prepare the results
        report = {
            "train_rows": int(overall["n"]),
            "test_rows": int(result["test_rows"]),
            "test_rows_with_unseen_pickup_zone":
                int(result["unseen_zone_rows"]),
            "baselines_joint_fit_seconds": round(fit_seconds, 3),
            "baselines": {
                name: {
                    "MAE_minutes": float(result[f"{name}_mae"]),
                    "RMSE_minutes": float(result[f"{name}_rmse"]),
                    "within_5_minutes_count": int(result[f"{name}_within_5_minutes"]),
                    "within_5_minutes_pct": round(
                        100 * result[f"{name}_within_5_minutes"] / result["test_rows"], 4
                    )
                }
                for name in model_names
            }
        }

        if round_errors:
            for score in report["baselines"].values():
                for metric in ("MAE_minutes", "RMSE_minutes"):
                    score[metric] = round(score[metric], 4)
        return report
    finally:
        stats.unpersist()


def main(argv=None):
    args = parse_args(argv)
    spark = SparkSession.builder.appName(
        "nyc-taxi-baselines"
    ).getOrCreate()
    try:
        report = run(spark, args)
    finally:
        spark.stop()

    print("\n=== BASELINE RESULTS ===")
    print(json.dumps(report, indent=2))

    # STEP 8: Save results locally
    output_path = args.output
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    output_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8"
    )
    print("Saved metrics:", output_path)


if __name__ == "__main__":
    main()
