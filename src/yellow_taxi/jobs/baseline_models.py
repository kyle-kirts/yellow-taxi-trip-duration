import argparse
import json
from datetime import date
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F


def main():
    parser = argparse.ArgumentParser()
    if args.cleaned is None and args.raw_root is None:
        args.cleaned = "/yellow-taxi/prepared/train/"
    parser.add_argument(
        "--cleaned",
        default=None,
        help="Optional existing cleaned Parquet directory (local or HDFS)"
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=0.01,
        help="Fraction to test first; use 1.0 for full data"
    )
    parser.add_argument(
        "--cutoff",
        default="2025-10-01",
        help="Test on pickups on/after this date"
    )
    parser.add_argument(
        "--output",
        default="results/baseline_metrics.json"
    )
    parser.add_argument(
        "--validation-cutoff",
        default=None,
        help="Train before this date, validate up to (but excluding) --cutoff"
    )
    args = parser.parse_args()
    try:
        cutoff_date = date.fromisoformat(args.cutoff)
        if args.validation_cutoff and date.fromisoformat(args.validation_cutoff) >= cutoff_date:
            parser.error("--validation-cutoff must be earlier than --cutoff")
    except ValueError:
        parser.error("Use YYYY-MM-DD for cutoff dates")

    if not (0 < args.sample <= 1):
        parser.error("--sample must be in (0, 1]")

    spark = SparkSession.builder.appName(
        "nyc-taxi-baselines"
    ).getOrCreate()

    try:
        # STEP 1: Load and clean the taxi data
        if args.cleaned:
            df = spark.read.parquet(args.cleaned)

            if args.sample < 1:
                df = df.sample(False, args.sample, seed=42)

            print("Reading existing cleaned data:", args.cleaned)

        else:
            from yellow_taxi.taxi_clean import clean

            df = (
                spark.read
                .option("mergeSchema", "true")
                .parquet(
                    f"{args.raw_root}/2024/",
                    f"{args.raw_root}/2025/"
                )
                .withColumn("source_path", F.input_file_name())
            )

            if args.sample < 1:
                df = df.sample(False, args.sample, seed=42)

            print("Applying Kyle's cleaning rules")
            df = clean(df)

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
        train_end = F.to_timestamp(F.lit(args.validation_cutoff or args.cutoff))
        train = df.filter(pickup < train_end)
        test = df.filter(pickup >= train_end)
        if args.validation_cutoff:
            test = test.filter(pickup < F.to_timestamp(F.lit(args.cutoff)))

        # STEP 3: Compute training statistics
        stats = (
            train.rollup("PULocationID")
            .agg(
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
            overall = stats.filter(
                F.col("PULocationID").isNull()
            ).first()

            if overall is None or overall["n"] == 0:
                raise ValueError(
                    "No training rows; check the cutoff and data"
                )

            overall_mean = float(overall["mean_min"])
            overall_median = float(overall["median_min"])

            # STEP 4: Compute mean duration for each pickup zone
            zone_means = (
                stats.filter(
                    F.col("PULocationID").isNotNull()
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
                "cutoff": args.cutoff,
                "validation_cutoff": args.validation_cutoff,
                "evaluation_split": "validation" if args.validation_cutoff else "test",
                "sample_fraction": args.sample,
                "train_rows": int(overall["n"]),
                "test_rows": int(result["test_rows"]),
                "test_rows_with_unseen_pickup_zone":
                    int(result["unseen_zone_rows"]),
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

            print(
                "\n=== BASELINE RESULTS "

            )
            print(json.dumps(report, indent=2))

            # STEP 8: Save results locally
            output_path = Path(args.output)
            output_path.parent.mkdir(
                parents=True, exist_ok=True
            )

            output_path.write_text(
                json.dumps(report, indent=2) + "\n",
                encoding="utf-8"
            )

            print("Saved metrics:", output_path)

        finally:
            stats.unpersist()

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
