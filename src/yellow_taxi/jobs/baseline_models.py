import argparse
import json
from datetime import date
from pathlib import Path
from time import perf_counter

from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F

from yellow_taxi.dataset import TRAIN_PATH


DEFAULT_SAMPLE = 0.01
DEFAULT_CUTOFF = "2025-10-01"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cleaned",
        default=TRAIN_PATH,
        help="Prepared Parquet directory"
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=DEFAULT_SAMPLE,
        help="Fraction to test first; use 1.0 for full data"
    )
    parser.add_argument(
        "--cutoff",
        type=date.fromisoformat,
        default=date.fromisoformat(DEFAULT_CUTOFF),
        help="Test on pickups on/after this date"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/baseline_metrics.json")
    )
    parser.add_argument(
        "--validation-cutoff",
        type=date.fromisoformat,
        default=None,
        help="Train before this date, validate up to (but excluding) --cutoff"
    )
    args = parser.parse_args(argv)

    if not (0 < args.sample <= 1):
        parser.error("--sample must be in (0, 1]")
    if args.validation_cutoff and args.validation_cutoff >= args.cutoff:
        parser.error("--validation-cutoff must be earlier than --cutoff")
    return args


def run(spark, args):
    # STEP 1: Read prepared data
    df = spark.read.parquet(args.cleaned)

    if args.sample < 1:
        df = df.sample(False, args.sample, seed=42)

    print("Reading existing cleaned data:", args.cleaned)

    required = [
        "tpep_pickup_datetime",
        "PULocationID",
        "trip_duration_min"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected cleaned columns: {missing}"
        )

    df = (
        df.select(*required)
        .dropna(subset=required)
        .filter(
            F.col("trip_duration_min").between(1, 240)
        )
    )

    # STEP 2: Split the data by time
    pickup = F.col("tpep_pickup_datetime")
    train_end = F.to_timestamp(
        F.lit((args.validation_cutoff or args.cutoff).isoformat())
    )
    train = df.filter(pickup < train_end)
    test = df.filter(pickup >= train_end)
    if args.validation_cutoff:
        test = test.filter(
            pickup < F.to_timestamp(F.lit(args.cutoff.isoformat()))
        )

    return {
        "cutoff": args.cutoff.isoformat(),
        "validation_cutoff": (
            args.validation_cutoff.isoformat() if args.validation_cutoff else None
        ),
        "evaluation_split": "validation" if args.validation_cutoff else "test",
        "sample_fraction": args.sample,
        **evaluate_baselines(train, test)
    }


def evaluate_baselines(train, test):
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
                "No training rows; check the cutoff and data"
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
                "No test rows; check the cutoff and data"
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
                    "MAE_minutes": round(
                        float(result[f"{name}_mae"]), 4
                    ),
                    "RMSE_minutes": round(
                        float(result[f"{name}_rmse"]), 4
                    ),
                    "within_5_minutes_count": int(result[f"{name}_within_5_minutes"]),
                    "within_5_minutes_pct": round(
                        100 * result[f"{name}_within_5_minutes"] / result["test_rows"], 4
                    )
                }
                for name in model_names
            }
        }

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
