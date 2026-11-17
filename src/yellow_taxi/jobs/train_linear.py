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
    LABEL, PREDICTION, linear_estimator, model_feature_inputs, model_feature_names, model_vectors,
)

DEFAULT_SAMPLE = 0.01
DEFAULT_LAMBDA = 0.0
DEFAULT_WEATHER = "on"
DEFAULT_CUTOFF = "2025-10-01"
DEFAULT_VALIDATION_CUTOFF = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned", default="/yellow-taxi/prepared/train/",
                        help="Prepared Parquet input with engineered features")
    parser.add_argument("--sample", type=float, default=DEFAULT_SAMPLE)
    parser.add_argument("--reg-param", "--linear-reg-param", "--lambda", type=float, default=DEFAULT_LAMBDA,
                        help="Lambda: 0 for ordinary linear regression, positive for Ridge; train_models.py tunes a grid")
    parser.add_argument("--weather", choices=("on", "off"), default=DEFAULT_WEATHER)
    parser.add_argument("--cutoff", type=date.fromisoformat, default=DEFAULT_CUTOFF,
                        help="First pickup date in the final test period")
    parser.add_argument("--validation-cutoff", type=date.fromisoformat, default=DEFAULT_VALIDATION_CUTOFF,
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


def evaluate(predictions, by_rush_hour=False):
    error = F.col(PREDICTION) - F.col(LABEL)
    aggregates = (
        F.count("*").alias("rows"),
        F.avg(F.abs(error)).alias("mae"),
        F.avg(error * error).alias("mse"),
        F.var_pop(LABEL).alias("duration_variance"),
        F.sum((F.abs(error) <= 5.0).cast("long")).alias("within_five"),
    )
    if by_rush_hour:
        groups = (predictions.rollup("is_rush_hour")
                  .agg(F.grouping("is_rush_hour").alias("overall"), *aggregates).collect())
        overall = next(row for row in groups if row.overall)
        rows, metrics = _metrics(overall)
        metrics["by_rush_hour"] = {
            {0: "off_peak", 1: "rush_hour"}.get(row.is_rush_hour, "unknown"):
                {"rows": row.rows, **_metrics(row)[1]}
            for row in groups if not row.overall
        }
        return rows, metrics
    return _metrics(predictions.agg(*aggregates).first())


def _metrics(summary):
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
    weather = args.weather == "on"
    required = ["tpep_pickup_datetime", "PULocationID", LABEL]
    columns = list(dict.fromkeys([*required, *model_feature_inputs(weather)]))
    missing = sorted(set(columns) - set(trips.columns))
    if missing:
        raise ValueError(f"Missing prepared columns: {', '.join(missing)}. "
                         "Use current prepared data; --weather off needs only taxi features.")
    if args.sample < 1:
        trips = trips.sample(False, args.sample, seed=42)
    trips = (trips.select(*columns)
             .dropna(subset=required)
             .filter(F.col(LABEL).between(1, 240)))
    start = (args.validation_cutoff or args.cutoff).isoformat()
    cutoff = args.cutoff.isoformat()
    pickup = F.col("tpep_pickup_datetime")
    start_ts = F.to_timestamp(F.lit(start))
    evaluation = trips.filter(pickup >= start_ts)
    if args.validation_cutoff:
        evaluation = evaluation.filter(pickup < F.to_timestamp(F.lit(cutoff)))

    train = model_vectors(trips.filter(pickup < start_ts), weather).persist(StorageLevel.MEMORY_AND_DISK)
    try:
        train_rows = train.count()
        if not train_rows:
            raise ValueError("No training trips; check the dates and input data.")
        model = linear_estimator(args.reg_param).fit(train)
    finally:
        train.unpersist()

    rows, metrics = evaluate(model.transform(model_vectors(evaluation, weather)), by_rush_hour=True)
    return {
        "model": "ordinary_linear_regression" if args.reg_param == 0 else "ridge_regression",
        "reg_param": args.reg_param,
        "lambda": args.reg_param,
        "input": args.cleaned,
        "sample_fraction": args.sample,
        "sampling_seed": 42,
        "weather": args.weather,
        "train_rows": train_rows,
        "evaluation_rows": rows,
        "evaluation_split": "validation" if args.validation_cutoff else "test",
        "train_before": start,
        "evaluation_from": start,
        "evaluation_before": cutoff if args.validation_cutoff else None,
        "cutoff": cutoff,
        "metrics": metrics,
        "features": list(model_feature_inputs(weather)),
        "intercept": float(model.intercept),
        "coefficients_by_feature": dict(zip(model_feature_names(weather), map(float, model.coefficients), strict=True)),
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
    write_report(report, output)


def write_report(report, output):
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=output.parent, delete=False, encoding="utf-8") as stream:
        stream.write(serialized)
    Path(stream.name).replace(output)
    print(serialized, end="")
    print("Saved metrics:", output)


if __name__ == "__main__":
    main()
