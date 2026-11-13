"""Train and evaluate ordinary linear regression or Ridge on prepared taxi trips."""

import argparse
import json
import math
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile

from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F

from yellow_taxi.linear_regression import (
    FEATURES, LABEL, PREDICTION, VECTOR_FEATURES, fit_linear_model, predict,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned", default="/yellow-taxi/prepared/train/",
                        help="Prepared Parquet input with engineered features")
    parser.add_argument("--sample", type=float, default=0.01)
    parser.add_argument("--reg-param", "--linear-reg-param", type=float, default=0.0,
                        help="0 for ordinary linear regression, positive for Ridge")
    parser.add_argument("--cutoff", type=date.fromisoformat, default=date(2025, 10, 1),
                        help="First pickup date in the final test period")
    parser.add_argument("--validation-cutoff", type=date.fromisoformat,
                        help="Train before this date and validate until --cutoff (exclusive)")
    parser.add_argument("--output", type=Path, help="Local JSON output path")
    args = parser.parse_args(argv)
    if not 0 < args.sample <= 1:
        parser.error("--sample must be in (0, 1]")
    if not math.isfinite(args.reg_param) or args.reg_param < 0:
        parser.error("--reg-param must be finite and nonnegative")
    if args.validation_cutoff and args.validation_cutoff >= args.cutoff:
        parser.error("--validation-cutoff must precede --cutoff")
    return args


def evaluate(predictions):
    error = F.col(PREDICTION) - F.col(LABEL)
    summary = predictions.agg(
        F.count("*").alias("rows"),
        F.avg(F.abs(error)).alias("mae"),
        F.avg(error * error).alias("mse"),
        F.var_pop(LABEL).alias("duration_variance"),
        F.sum((F.abs(error) <= 5.0).cast("long")).alias("within_five"),
    ).first()
    if summary.rows == 0:
        raise ValueError("No evaluation trips; check the dates and input data.")
    variance = summary.duration_variance
    return summary.rows, {
        "MAE_minutes": round(summary.mae, 4),
        "RMSE_minutes": round(math.sqrt(summary.mse), 4),
        "R2": round(1.0 - summary.mse / variance, 6) if variance > 0 else None,
        "within_5_minutes_count": summary.within_five,
        "within_5_minutes_pct": round(100.0 * summary.within_five / summary.rows, 4),
    }


def run(spark, args):
    trips = spark.read.parquet(args.cleaned)
    required = ["tpep_pickup_datetime", "PULocationID", LABEL]
    missing = sorted(set([*required, *FEATURES]) - set(trips.columns))
    if missing:
        raise ValueError(f"Missing prepared columns: {', '.join(missing)}")
    if args.sample < 1:
        trips = trips.sample(False, args.sample, seed=42)
    trips = (trips.select(*required, *FEATURES)
             .dropna(subset=required)
             .filter(F.col(LABEL).between(1, 240)))
    start = (args.validation_cutoff or args.cutoff).isoformat()
    cutoff = args.cutoff.isoformat()
    pickup = F.col("tpep_pickup_datetime")
    start_ts = F.to_timestamp(F.lit(start))
    evaluation = trips.filter(pickup >= start_ts)
    if args.validation_cutoff:
        evaluation = evaluation.filter(pickup < F.to_timestamp(F.lit(cutoff)))

    train = trips.filter(pickup < start_ts).persist(StorageLevel.MEMORY_AND_DISK)
    try:
        train_rows = train.count()
        if not train_rows:
            raise ValueError("No training trips; check the dates and input data.")
        model = fit_linear_model(train, args.reg_param)
    finally:
        train.unpersist()

    rows, metrics = evaluate(predict(model, evaluation))
    return {
        "model": "ordinary_linear_regression" if args.reg_param == 0 else "ridge_regression",
        "reg_param": args.reg_param,
        "input": args.cleaned,
        "sample_fraction": args.sample,
        "sampling_seed": 42,
        "train_rows": train_rows,
        "evaluation_rows": rows,
        "evaluation_split": "validation" if args.validation_cutoff else "test",
        "train_before": start,
        "evaluation_from": start,
        "evaluation_before": cutoff if args.validation_cutoff else None,
        "cutoff": cutoff,
        "metrics": metrics,
        "features": list(FEATURES),
        "intercept": float(model.intercept),
        "coefficients_by_feature": dict(zip(VECTOR_FEATURES, map(float, model.coefficients))),
    }


def main(argv=None):
    args = parse_args(argv)
    spark = SparkSession.builder.appName("nyc-taxi-linear-regression").getOrCreate()
    try:
        report = run(spark, args)
    finally:
        spark.stop()
    output = args.output or Path("results") / (
        f"{report['model']}_{report['evaluation_split']}_lambda{args.reg_param:g}.json"
    )
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=output.parent, delete=False, encoding="utf-8") as stream:
        stream.write(serialized)
    Path(stream.name).replace(output)
    print(f"\n=== {report['model'].replace('_', ' ').upper()} RESULTS ===")
    print(serialized, end="")
    print("Saved metrics:", output)


if __name__ == "__main__":
    main()
