from datetime import datetime, timedelta

import pytest

from yellow_taxi.taxi_clean import add_label, clean, cleaning_rules, rule_counts

PICKUP = datetime(2024, 1, 15, 8, 0)


def raw_row(case="valid", duration_minutes=20, **overrides):
    row = {
        "case": case,
        "VendorID": 1,
        "tpep_pickup_datetime": PICKUP,
        "tpep_dropoff_datetime": PICKUP + timedelta(minutes=duration_minutes),
        "passenger_count": 1,
        "trip_distance": 5.0,
        "RatecodeID": 1,
        "PULocationID": 161,
        "DOLocationID": 161,
        "payment_type": 1,
        "fare_amount": 15.0,
        "total_amount": 20.0,
        "source_path": "yellow_tripdata_2024-01.parquet",
    }
    row.update(overrides)
    return row


def passing_cases(spark, rule_name, rows):
    keep = dict(cleaning_rules())[rule_name]
    df = add_label(spark.createDataFrame(rows))
    return {row.case for row in df.filter(keep).select("case").collect()}


@pytest.mark.parametrize(
    ("rule_name", "column", "values", "expected"),
    [
        ("passenger_count", "passenger_count", [0, 1, 6, 7], {"1", "6"}),
        ("RatecodeID", "RatecodeID", [0, 1, 6, 7], {"1", "6"}),
        ("VendorID", "VendorID", [0, 1, 2, 3, 6, 7], {"1", "2", "6", "7"}),
        (
            "PULocationID",
            "PULocationID",
            [0, 1, 263, 264, 265, 999],
            {"1", "263"},
        ),
        (
            "DOLocationID",
            "DOLocationID",
            [0, 1, 263, 264, 265, 999],
            {"1", "263"},
        ),
        (
            "payment_type",
            "payment_type",
            [0, 1, 2, 3, 4, 5, 6, 99],
            {"1", "2", "5"},
        ),
        ("fare_amount", "fare_amount", [0.0, 0.01], {"0.01"}),
        ("total_amount", "total_amount", [0.0, 0.01], {"0.01"}),
    ],
)
def test_cleaning_rule_value_ranges(spark, rule_name, column, values, expected):
    rows = [raw_row(case=str(value), **{column: value}) for value in values]
    assert passing_cases(spark, rule_name, rows) == expected


def test_duration_rule_includes_one_to_240_minutes(spark):
    durations = [0.99, 1.0, 240.0, 240.1]
    rows = [raw_row(case=str(value), duration_minutes=value) for value in durations]
    assert passing_cases(spark, "duration", rows) == {"1.0", "240.0"}


def test_distance_rule_excludes_zero_and_over_100_miles(spark):
    distances = [0.0, 0.01, 100.0, 100.01]
    rows = [raw_row(case=str(value), trip_distance=value) for value in distances]
    assert passing_cases(spark, "trip_distance", rows) == {"0.01", "100.0"}


def test_speed_rule_allows_80_mph_but_not_more(spark):
    speeds = [80.0, 80.01]
    rows = [
        raw_row(case=str(speed), duration_minutes=60, trip_distance=speed)
        for speed in speeds
    ]
    assert passing_cases(spark, "speed", rows) == {"80.0"}


def test_month_rule_rejects_mismatched_parseable_file_month(spark):
    rows = [
        raw_row(case="matching"),
        raw_row(case="wrong", source_path="yellow_tripdata_2024-02.parquet"),
        raw_row(case="unparseable", source_path=""),
    ]
    assert passing_cases(spark, "wrong_month", rows) == {"matching", "unparseable"}


def test_clean_removes_nulls_and_post_trip_columns_but_keeps_same_zone(spark):
    df = spark.createDataFrame([
        raw_row(case="valid"),
        raw_row(case="missing_passenger", passenger_count=None),
    ])

    result = clean(df)
    rows = result.collect()

    assert len(rows) == 1
    assert rows[0].case == "valid"
    assert rows[0].trip_duration_min == 20.0
    assert rows[0].PULocationID == rows[0].DOLocationID
    assert {
        "tpep_dropoff_datetime",
        "payment_type",
        "fare_amount",
        "total_amount",
        "source_path",
    }.isdisjoint(result.columns)


def test_rule_counts_only_count_rows_rejected_at_that_step(spark):
    df = add_label(spark.createDataFrame([
        raw_row(case="valid"),
        raw_row(case="bad_duration_and_distance", duration_minutes=0, trip_distance=0.0),
        raw_row(case="bad_distance", trip_distance=0.0),
    ]))

    counts = rule_counts(df, cleaning_rules())

    assert counts["raw"] == 3
    assert counts["duration"] == 1
    assert counts["trip_distance"] == 1
    assert counts["speed"] == 0
