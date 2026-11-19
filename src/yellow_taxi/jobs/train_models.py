"""Compare models on prepared month folds, then score the separate prepared test data."""

import argparse
import math
from contextlib import ExitStack
from itertools import product
from pathlib import Path
from statistics import mean
from time import perf_counter

from pyspark import StorageLevel
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession, functions as F

from yellow_taxi.dataset import LABEL, N_FOLDS
from yellow_taxi.jobs.baseline_models import (
    SEED, add_data_arguments, evaluate_baselines, read_model_trips, validate_data_arguments,
)
from yellow_taxi.jobs.train_linear import evaluate, vectorize, write_report
from yellow_taxi.linear_regression import (
    PREDICTION, VECTOR, linear_estimator, model_feature_inputs, model_feature_names,
)

DEFAULT_TUNE_SAMPLE = 1.0
DEFAULT_WEATHER = "on"
RIDGE_LAMBDAS = [0.01, 0.1, 1.0, 10.0]
RF_TREES = [20, 50]
RF_DEPTHS = [5, 10]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--cv-only", action="store_true", help="Stop after four-fold validation, before opening test data")
    parser.add_argument("--tune-sample", type=float, default=DEFAULT_TUNE_SAMPLE, help="Fraction of sampled training trips used in CV")
    parser.add_argument("--weather", choices=("on", "off"), default=DEFAULT_WEATHER)
    parser.add_argument("--ridge-lambdas", type=float, nargs="+", default=RIDGE_LAMBDAS)
    parser.add_argument("--rf-trees", type=int, nargs="+", default=RF_TREES)
    parser.add_argument("--rf-depths", type=int, nargs="+", default=RF_DEPTHS)
    parser.add_argument("--output", type=Path, help="Local JSON results file")
    args = parser.parse_args(argv)
    validate_data_arguments(parser, args)
    if not 0 < args.tune_sample <= 1:
        parser.error("--tune-sample must be in (0, 1]")
    if any(not math.isfinite(x) or x <= 0 for x in args.ridge_lambdas):
        parser.error("--ridge-lambdas must be finite and positive; OLS already uses zero")
    if min(args.rf_trees) < 1 or any(not 1 <= x <= 30 for x in args.rf_depths):
        parser.error("--rf-trees must be positive and --rf-depths between 1 and 30")
    return args


def prepare_vectors(trips, args, training):
    use_weather = args.weather == "on"
    keys = ["fold"] if training else []
    frame = vectorize(trips, use_weather, keep=keys)
    columns = [*keys, "PULocationID", "DOLocationID", "is_rush_hour", LABEL, VECTOR]
    if use_weather:
        columns.append("weather_missing")
    return frame.select(*columns)


def fold_counts(trips):
    missing = [F.sum("weather_missing").alias("n_missing")] if "weather_missing" in trips.columns else []
    rows = trips.groupBy("fold").agg(F.count("*").alias("n"), *missing).collect()
    counts = {r.fold: r.n for r in rows}
    if set(counts) != set(range(N_FOLDS)):
        raise ValueError(f"Expected folds 0 through {N_FOLDS - 1}, found {counts}. "
                         "Check the prepared train root and sample size.")
    return counts, sum(r.n_missing for r in rows) if missing else None


def estimator(name, params):
    if name == "random_forest":
        return RandomForestRegressor(featuresCol=VECTOR, labelCol=LABEL, predictionCol=PREDICTION,
                                     seed=SEED, **params)
    return linear_estimator(params["lambda"])


def tune(training, args, fixed_models):
    trials = [{"model": name, "params": params, "fold_metrics": []} for name, params in fixed_models.items()]
    trials.extend({"model": "ridge_regression", "params": {"lambda": x}, "fold_metrics": []}
                  for x in sorted(set(args.ridge_lambdas)))
    for trees, depth in product(sorted(set(args.rf_trees)), sorted(set(args.rf_depths))):
        trials.append({"model": "random_forest", "params": {"numTrees": trees, "maxDepth": depth},
                       "fold_metrics": []})

    baseline_folds = {}
    error = F.col(PREDICTION) - F.col(LABEL)
    for fold in range(N_FOLDS):
        train = training.filter(F.col("fold") != fold)
        valid = training.filter(F.col("fold") == fold)
        baselines = evaluate_baselines(train, valid, round_errors=False)["baselines"]
        for name, metrics in baselines.items():
            baseline_folds.setdefault(name, []).append({
                "fold": fold, "MAE_minutes": metrics["MAE_minutes"], "RMSE_minutes": metrics["RMSE_minutes"],
            })
        for trial in trials:
            started = perf_counter()
            model = estimator(trial["model"], trial["params"]).fit(train)
            fit_seconds = perf_counter() - started
            row = model.transform(valid).agg(F.avg(F.abs(error)), F.sqrt(F.avg(error * error))).first()
            trial["fold_metrics"].append({"fold": fold, "MAE_minutes": row[0], "RMSE_minutes": row[1],
                                          "fit_seconds": round(fit_seconds, 3)})
            print(f"fold={fold} {trial['model']} {trial['params']} MAE={row[0]:.4f}", flush=True)
    trials.extend({"model": name, "params": {}, "fold_metrics": scores} for name, scores in baseline_folds.items())
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
    with ExitStack() as stack:
        def cache(frame):
            frame.persist(StorageLevel.MEMORY_AND_DISK)
            stack.callback(frame.unpersist)
            return frame

        training = cache(prepare_vectors(read_model_trips(spark, args, "train"), args, training=True))
        counts, train_missing = fold_counts(training)
        cv_trips = training
        cv_counts = counts
        if args.tune_sample < 1:
            cv_trips = cache(training.sample(False, args.tune_sample, seed=SEED + 1))
            cv_counts, _ = fold_counts(cv_trips)
        fixed = {"ordinary_linear_regression": {"lambda": 0.0}}
        print(f"Training rows by fold: {counts}; CV rows by fold: {cv_counts}", flush=True)
        trials, selected = tune(cv_trips, args, fixed)
        print(f"Selected parameters: {selected}", flush=True)
        if args.tune_sample < 1:
            cv_trips.unpersist()
        best = min(trials, key=lambda trial: trial["mean_MAE_minutes"])
        report = {
            "training_input": args.cleaned,
            "evaluation_split": "cross_validation" if args.cv_only else "external_test",
            "sample_fraction": args.sample, "sampling_seed": SEED, "tune_sample_fraction": args.tune_sample,
            "weather": args.weather, "feature_columns": model_feature_inputs(args.weather == "on"),
            "selection_metric": "Lowest mean validation MAE across prepared month folds",
            "cv_rows_by_fold": cv_counts, "cv_trials": trials, "selected_params": selected,
            "lowest_cv_mae": {key: best[key] for key in ("model", "params", "mean_MAE_minutes")},
        }
        if args.cv_only:
            return report

        evaluation = cache(prepare_vectors(read_model_trips(spark, args, "test"), args, training=False))
        missing = [F.sum("weather_missing")] if args.weather == "on" else []
        eval_stats = evaluation.agg(F.count("*"), *missing).first()
        if not eval_stats[0]:
            raise ValueError("No evaluation trips; check the prepared test root, test year and sample size.")
        print(f"Evaluation rows: {eval_stats[0]}", flush=True)
        baseline = evaluate_baselines(training, evaluation)
        scores = baseline["baselines"]
        fit_seconds = {"three_baselines_together": baseline["baselines_joint_fit_seconds"]}
        for name, params in {**fixed, **selected}.items():
            print(f"Fitting final {name}: {params}", flush=True)
            started = perf_counter()
            model = estimator(name, params).fit(training)
            fit_seconds[name] = round(perf_counter() - started, 3)
            rows, scores[name] = evaluate(model.transform(evaluation), by_rush_hour=True)
            if name == "random_forest":
                scores[name]["importance_by_input"] = forest_importance(model, args.weather == "on")
            if rows != baseline["test_rows"]:
                raise RuntimeError(f"{name} scored a different number of evaluation rows")
    report.update({
        "holdout": {"test_path": args.test_path, "test_year": args.test_year},
        "weather_missing_rows": ({"training": train_missing, "evaluation": eval_stats[1]}
                                 if args.weather == "on" else None),
        "evaluation": {"train_rows": baseline["train_rows"], "evaluation_rows": baseline["test_rows"],
                       "unseen_pickup_zone_rows": baseline["test_rows_with_unseen_pickup_zone"],
                       "fit_seconds": fit_seconds, "models": scores},
    })
    return report


def main(argv=None):
    args = parse_args(argv)
    spark = SparkSession.builder.appName("nyc-taxi-model-comparison").getOrCreate()
    try:
        report = run(spark, args)
    finally:
        spark.stop()
    period = "cv" if args.cv_only else f"prepared_test_{args.test_year or 'all'}"
    write_report(report, args.output or Path(f"results/model_comparison_{period}_weather_{args.weather}.json"))
    if args.cv_only:
        return
    print("\nModel                         MAE (min)  RMSE (min)")
    for name, score in report["evaluation"]["models"].items():
        print(f"{name:30} {score['MAE_minutes']:9.4f} {score['RMSE_minutes']:11.4f}")


if __name__ == "__main__":
    main()
