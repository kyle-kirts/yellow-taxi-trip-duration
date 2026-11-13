import math

import pytest

from yellow_taxi.jobs.train_linear import evaluate, parse_args, run
from yellow_taxi.linear_regression import FEATURES, LABEL, PREDICTION


def test_error_metrics_count_both_five_minute_boundaries(spark):
    # Signed errors are -5, +5, +6 and 0 minutes.
    predictions = spark.createDataFrame(
        [(10.0, 5.0), (20.0, 25.0), (30.0, 36.0), (40.0, 40.0)],
        [LABEL, PREDICTION],
    )
    rows, metrics = evaluate(predictions)
    assert rows == 4
    assert metrics["MAE_minutes"] == 4.0
    assert metrics["RMSE_minutes"] == pytest.approx(math.sqrt(86 / 4), abs=0.00005)
    assert metrics["R2"] == pytest.approx(1 - 86 / 500)
    assert metrics["within_5_minutes_count"] == 3
    assert metrics["within_5_minutes_pct"] == 75.0


@pytest.fixture(scope="module")
def prepared_trips(spark, tmp_path_factory):
    def trip(pickup, distance, minutes):
        features = dict.fromkeys(FEATURES, 0.0)
        features["zone_distance_mi"] = distance
        return {"tpep_pickup_datetime": pickup, "PULocationID": 1, LABEL: minutes, **features}

    # Before July the relationship is exactly duration = 6 + 2 * distance.
    rows = [trip("2025-06-30 23:59:59", float(i), 6.0 + 2 * i) for i in range(12)]
    rows += [
        trip("2025-07-01 00:00:00", 1.0, 8.0),
        trip("2025-09-30 23:59:59", None, 6.0),
        trip("2025-10-01 00:00:00", 1.0, 200.0),
    ]
    path = (tmp_path_factory.mktemp("linear") / "trips").as_uri()
    spark.createDataFrame(rows).selectExpr(
        "cast(tpep_pickup_datetime as timestamp) as tpep_pickup_datetime",
        "PULocationID", LABEL, *FEATURES,
    ).write.parquet(path)
    return path


@pytest.mark.parametrize("strength,model_name", [
    (0.0, "ordinary_linear_regression"),
    (0.1, "ridge_regression"),
])
def test_fit_on_earlier_trips_and_validate_before_october(spark, prepared_trips, strength, model_name):
    args = parse_args([
        "--cleaned", prepared_trips, "--sample", "1", "--reg-param", str(strength),
        "--validation-cutoff", "2025-07-01", "--cutoff", "2025-10-01",
    ])
    report = run(spark, args)
    assert report["model"] == model_name
    assert report["evaluation_split"] == "validation"
    assert report["train_rows"] == 12
    assert report["evaluation_rows"] == 2
    slope = report["coefficients_by_feature"]["zone_distance_mi"]
    if strength == 0:
        assert slope == pytest.approx(2.0, abs=0.0001)
        assert report["metrics"]["MAE_minutes"] == pytest.approx(0, abs=0.0001)
    else:
        assert 0 < slope < 2.0
