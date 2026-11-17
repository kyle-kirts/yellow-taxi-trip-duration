"""Tune Ridge and Random Forest on prepared folds and compare six predictors on later trips."""

import argparse
import math
from contextlib import ExitStack
from datetime import date
from itertools import product
from pathlib import Path
from statistics import mean
from time import perf_counter

from pyspark import StorageLevel
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession, functions as F

from yellow_taxi.dataset import LABEL, N_FOLDS, TRAIN_PATH, load_prepared
from yellow_taxi.jobs.baseline_models import evaluate_baselines
from yellow_taxi.jobs.train_linear import evaluate, write_report
from yellow_taxi.linear_regression import (
    PREDICTION, VECTOR, linear_estimator, model_feature_inputs, model_feature_names, model_vectors,
)

DEFAULT_SAMPLE = 0.01
DEFAULT_TUNE_SAMPLE = 1.0
DEFAULT_WEATHER = "on"
DEFAULT_CUTOFF = "2025-10-01"
DEFAULT_TEST_PATH = None
DEFAULT_TEST_YEAR = 2026
RIDGE_LAMBDAS = [0.01, 0.1, 1.0, 10.0]
RF_TREES = [20, 50]
RF_DEPTHS = [5, 10]
SEED = 42
PICKUP = "tpep_pickup_datetime"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned", default=TRAIN_PATH, help="Prepared train root containing fold partitions")
    holdout = parser.add_mutually_exclusive_group()
    holdout.add_argument("--cutoff", type=date.fromisoformat, help="First held-out date (default 2025-10-01)")
    holdout.add_argument("--test-path", help="Separate prepared test root partitioned by year")
    parser.add_argument("--test-year", type=int, help="Year to score from --test-path (default 2026)")
    parser.add_argument("--sample", type=float, default=DEFAULT_SAMPLE, help="Fraction of trips for this run")
    parser.add_argument("--tune-sample", type=float, default=DEFAULT_TUNE_SAMPLE, help="Fraction of sampled training trips used in CV")
    parser.add_argument("--weather", choices=("on", "off"), default=DEFAULT_WEATHER)
    parser.add_argument("--ridge-lambdas", type=float, nargs="+", default=RIDGE_LAMBDAS)
    parser.add_argument("--rf-trees", type=int, nargs="+", default=RF_TREES)
    parser.add_argument("--rf-depths", type=int, nargs="+", default=RF_DEPTHS)
    parser.add_argument("--output", type=Path, help="Local JSON results file")
    args = parser.parse_args(argv)
    if args.cutoff is None and args.test_path is None:
        args.test_path = DEFAULT_TEST_PATH
    if not (0 < args.sample <= 1 and 0 < args.tune_sample <= 1):
        parser.error("--sample and --tune-sample must be in (0, 1]")
    if any(not math.isfinite(x) or x <= 0 for x in args.ridge_lambdas):
        parser.error("--ridge-lambdas must be finite and positive; OLS already uses zero")
    if min(args.rf_trees) < 1 or any(not 1 <= x <= 30 for x in args.rf_depths):
        parser.error("--rf-trees must be positive and --rf-depths between 1 and 30")
    if args.test_year is not None and (not args.test_path or not 1 <= args.test_year <= 9999):
        parser.error("--test-year requires --test-path and a valid year")
    if args.test_path:
        args.test_year = args.test_year or DEFAULT_TEST_YEAR
    else:
        args.cutoff = args.cutoff or date.fromisoformat(DEFAULT_CUTOFF)
    return args


def split_trips(spark, args):
    trips = load_prepared(spark, path=args.cleaned)
    if args.sample < 1:
        trips = trips.sample(False, args.sample, seed=SEED)
    pickup = F.col(PICKUP)
    if args.test_path:
        evaluation = load_prepared(spark, mode="test", path=args.test_path)
        if "year" in evaluation.columns:
            evaluation = evaluation.filter(F.col("year") == args.test_year)
        if args.sample < 1:
            evaluation = evaluation.sample(False, args.sample, seed=SEED)
        return (trips.filter(pickup < F.lit(date(args.test_year, 1, 1))),
                evaluation.filter(F.year(pickup) == args.test_year))
    return trips.filter(pickup < F.lit(args.cutoff)), trips.filter(pickup >= F.lit(args.cutoff))


def prepare_vectors(trips, args, training):
    use_weather = args.weather == "on"
    keys = ["fold"] if training else []
    required = list(dict.fromkeys([*keys, PICKUP, LABEL, *model_feature_inputs(use_weather)]))
    missing = sorted(set(required) - set(trips.columns))
    if missing:
        raise ValueError(f"Missing prepared columns: {', '.join(missing)}. Check the prepared root, "
                         "rebuild it with prepare.py, or use --weather off for older taxi-only data.")
    eligible = (trips.select(*required).dropna(subset=[PICKUP, LABEL, "PULocationID"])
                .filter(F.col(LABEL).between(1, 240)))
    weather_flags = [F.col("temp_c").isNull().cast("int").alias("weather_missing")] if use_weather else []
    return model_vectors(eligible, use_weather).select(*keys, "PULocationID", "is_rush_hour", LABEL,
                                                       *weather_flags, VECTOR)


def fold_counts(trips):
    missing = [F.sum("weather_missing").alias("n_missing")] if "weather_missing" in trips.columns else []
    rows = trips.groupBy("fold").agg(F.count("*").alias("n"), *missing).collect()
    counts = {r.fold: r.n for r in rows}
    if set(counts) != set(range(N_FOLDS)):
        raise ValueError(f"Expected folds 0 through {N_FOLDS - 1}, found {counts}. "
                         "Check the prepared train root, dates and sample size.")
    return counts, sum(r.n_missing for r in rows) if missing else None


def estimator(name, params):
    if name == "random_forest":
        return RandomForestRegressor(featuresCol=VECTOR, labelCol=LABEL, predictionCol=PREDICTION,
                                     seed=SEED, **params)
    return linear_estimator(params["lambda"])


def tune(training, args):
    trials = [{"model": "ridge_regression", "params": {"lambda": x}, "fold_metrics": []}
              for x in sorted(set(args.ridge_lambdas))]
    for trees, depth in product(sorted(set(args.rf_trees)), sorted(set(args.rf_depths))):
        trials.append({"model": "random_forest", "params": {"numTrees": trees, "maxDepth": depth},
                       "fold_metrics": []})

    error = F.col(PREDICTION) - F.col(LABEL)
    for fold in range(N_FOLDS):
        train = training.filter(F.col("fold") != fold)
        valid = training.filter(F.col("fold") == fold)
        for trial in trials:
            started = perf_counter()
            model = estimator(trial["model"], trial["params"]).fit(train)
            fit_seconds = perf_counter() - started
            row = model.transform(valid).agg(F.avg(F.abs(error)), F.sqrt(F.avg(error * error))).first()
            trial["fold_metrics"].append({"fold": fold, "MAE_minutes": row[0], "RMSE_minutes": row[1],
                                          "fit_seconds": round(fit_seconds, 3)})
            print(f"fold={fold} {trial['model']} {trial['params']} MAE={row[0]:.4f}", flush=True)
    for trial in trials:
        for metric in ("MAE_minutes", "RMSE_minutes"):
            trial[f"mean_{metric}"] = mean(fold[metric] for fold in trial["fold_metrics"])
    selected = {}
    for name in ("ridge_regression", "random_forest"):
        candidates = (trial for trial in trials if trial["model"] == name)
        selected[name] = min(candidates, key=lambda trial: trial["mean_MAE_minutes"])["params"]
    return trials, selected


def forest_importance(model, weather):
    grouped = dict.fromkeys(model_feature_inputs(weather), 0.0)
    for name, weight in zip(model_feature_names(weather), model.featureImportances, strict=True):
        source = name.split("=")[0]
        if source not in grouped:
            source = source.rsplit("_", 1)[0]
        grouped[source] += float(weight)
    return dict(sorted(grouped.items(), key=lambda item: item[1], reverse=True))


def run(spark, args):
    print(f"Reading {args.cleaned}; sample={args.sample:g}; weather={args.weather}", flush=True)
    training, evaluation = split_trips(spark, args)
    with ExitStack() as stack:
        def cache(frame):
            frame.persist(StorageLevel.MEMORY_AND_DISK)
            stack.callback(frame.unpersist)
            return frame

        training = cache(prepare_vectors(training, args, training=True))
        counts, train_missing = fold_counts(training)
        cv_trips = training
        cv_counts = counts
        if args.tune_sample < 1:
            cv_trips = cache(training.sample(False, args.tune_sample, seed=SEED + 1))
            cv_counts, _ = fold_counts(cv_trips)
        evaluation = cache(prepare_vectors(evaluation, args, training=False))
        missing = [F.sum("weather_missing")] if args.weather == "on" else []
        eval_stats = evaluation.agg(F.count("*"), *missing).first()
        if not eval_stats[0]:
            raise ValueError("No evaluation trips; check the holdout dates, test path and sample size.")
        print(f"Training rows by fold: {counts}; evaluation rows: {eval_stats[0]}", flush=True)
        print("Tuning Ridge and Random Forest on four prepared folds", flush=True)
        trials, selected = tune(cv_trips, args)
        print(f"Selected parameters: {selected}", flush=True)
        if args.tune_sample < 1:
            cv_trips.unpersist()
        baseline = evaluate_baselines(training, evaluation)
        scores = baseline["baselines"]
        fit_seconds = {"three_baselines_together": baseline["baselines_joint_fit_seconds"]}
        for name, params in {"ordinary_linear_regression": {"lambda": 0.0}, **selected}.items():
            print(f"Fitting final {name}: {params}", flush=True)
            started = perf_counter()
            model = estimator(name, params).fit(training)
            fit_seconds[name] = round(perf_counter() - started, 3)
            rows, scores[name] = evaluate(model.transform(evaluation), by_rush_hour=True)
            if name == "random_forest":
                scores[name]["importance_by_input"] = forest_importance(model, args.weather == "on")
            if rows != baseline["test_rows"]:
                raise RuntimeError(f"{name} scored a different number of evaluation rows")
    return {
        "training_input": args.cleaned,
        "holdout": {"test_path": args.test_path, "test_year": args.test_year,
                    "cutoff": args.cutoff.isoformat() if args.cutoff else None},
        "evaluation_split": "external_test" if args.test_path else "development_holdout",
        "sample_fraction": args.sample, "sampling_seed": SEED, "tune_sample_fraction": args.tune_sample,
        "weather": args.weather, "feature_columns": model_feature_inputs(args.weather == "on"),
        "weather_missing_rows": ({"training": train_missing, "evaluation": eval_stats[1]}
                                 if args.weather == "on" else None),
        "selection_metric": "Lowest mean validation MAE across prepared month folds",
        "cv_rows_by_fold": cv_counts, "cv_trials": trials, "selected_params": selected,
        "evaluation": {"train_rows": baseline["train_rows"], "evaluation_rows": baseline["test_rows"],
                       "unseen_pickup_zone_rows": baseline["test_rows_with_unseen_pickup_zone"],
                       "fit_seconds": fit_seconds, "models": scores},
    }


def main(argv=None):
    args = parse_args(argv)
    spark = SparkSession.builder.appName("nyc-taxi-model-comparison").getOrCreate()
    try:
        report = run(spark, args)
    finally:
        spark.stop()
    period = args.test_year or args.cutoff.isoformat()
    write_report(report, args.output or Path(f"results/model_comparison_{period}_weather_{args.weather}.json"))
    print("\nModel                         MAE (min)  RMSE (min)")
    for name, score in report["evaluation"]["models"].items():
        print(f"{name:30} {score['MAE_minutes']:9.4f} {score['RMSE_minutes']:11.4f}")


if __name__ == "__main__":
    main()
