"""Train and evaluate ordinary linear regression or Ridge on prepared taxi trips."""

import argparse
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F

from yellow_taxi.jobs.baseline_models import (
    SEED, add_data_arguments, read_model_trips, validate_data_arguments,
)
from yellow_taxi.linear_regression import (
    LABEL, PREDICTION, linear_estimator, model_feature_inputs, model_feature_names, model_vectors,
)

DEFAULT_LAMBDA = 0.0
DEFAULT_WEATHER = "on"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--reg-param", "--linear-reg-param", "--lambda", type=float, default=DEFAULT_LAMBDA,
                        help="Lambda: 0 for ordinary linear regression, positive for Ridge; train_models.py tunes a grid")
    parser.add_argument("--weather", choices=("on", "off"), default=DEFAULT_WEATHER)
    parser.add_argument("--output", type=Path, help="Local JSON output path")
    args = parser.parse_args(argv)
    validate_data_arguments(parser, args)
    if not math.isfinite(args.reg_param) or args.reg_param < 0:
        parser.error("--reg-param must be finite and nonnegative")
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
        raise ValueError("No evaluation trips; check the prepared test root and sample size.")
    variance = summary.duration_variance
    return summary.rows, {
        "MAE_minutes": round(summary.mae, 4),
        "RMSE_minutes": round(math.sqrt(summary.mse), 4),
        "R2": round(1.0 - summary.mse / variance, 6) if variance > 0 else None,
        "within_5_minutes_count": summary.within_five,
        "within_5_minutes_pct": round(100.0 * summary.within_five / summary.rows, 4),
    }


def vectorize(trips, weather, keep=()):
    columns = list(dict.fromkeys([LABEL, *model_feature_inputs(weather), *keep]))
    missing = sorted(set(columns) - set(trips.columns))
    if missing:
        raise ValueError(f"Missing prepared columns: {', '.join(missing)}. "
                         "Use current prepared data; --weather off needs only taxi features.")
    return model_vectors(trips.select(*columns), weather)


def run(spark, args):
    weather = args.weather == "on"
    train = vectorize(read_model_trips(spark, args, "train"), weather).persist(StorageLevel.MEMORY_AND_DISK)
    try:
        train_rows = train.count()
        if not train_rows:
            raise ValueError("No training trips; check the prepared training root and sample size.")
        model = linear_estimator(args.reg_param).fit(train)
    finally:
        train.unpersist()

    evaluation = vectorize(read_model_trips(spark, args, "test"), weather)
    rows, metrics = evaluate(model.transform(evaluation), by_rush_hour=True)
    return {
        "model": "ordinary_linear_regression" if args.reg_param == 0 else "ridge_regression",
        "reg_param": args.reg_param,
        "lambda": args.reg_param,
        "input": args.cleaned,
        "sample_fraction": args.sample,
        "sampling_seed": SEED,
        "weather": args.weather,
        "train_rows": train_rows,
        "evaluation_rows": rows,
        "evaluation_split": "external_test",
        "test_path": args.test_path,
        "test_year": args.test_year,
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
    period = f"prepared_test_{args.test_year or 'all'}"
    output = args.output or Path("results") / f"{report['model']}_{period}_lambda{args.reg_param:g}.json"
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
