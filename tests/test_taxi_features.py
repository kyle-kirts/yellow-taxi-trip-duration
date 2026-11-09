import math
from datetime import datetime

import pytest

from yellow_taxi.taxi_features import (add_time_features, add_zone_attributes, add_zone_distance,
                                                load_zone_centroids, load_zone_lookup)

EWR, JFK, LGA, MIDTOWN_CENTER, TIMES_SQ = 1, 132, 138, 161, 230


@pytest.fixture(scope="session")
def centroids(spark):
    return load_zone_centroids(spark)


def distances(spark, centroids, pairs):
    df = spark.createDataFrame(pairs, "PULocationID int, DOLocationID int")
    out = add_zone_distance(df, centroids)
    return out, {(r.PULocationID, r.DOLocationID): r.zone_distance_mi for r in out.collect()}


def test_same_zone_is_zero(spark, centroids):
    _, d = distances(spark, centroids, [(TIMES_SQ, TIMES_SQ)])
    assert d[(TIMES_SQ, TIMES_SQ)] == 0


def test_symmetric(spark, centroids):
    _, d = distances(spark, centroids, [(JFK, LGA), (LGA, JFK)])
    assert d[(JFK, LGA)] == pytest.approx(d[(LGA, JFK)])


def test_known_pairs(spark, centroids):
    _, d = distances(spark, centroids, [(JFK, LGA), (MIDTOWN_CENTER, TIMES_SQ)])
    assert 9 <= d[(JFK, LGA)] <= 11
    assert d[(MIDTOWN_CENTER, TIMES_SQ)] < 1


def test_no_helper_columns_and_no_nulls(spark, centroids):
    out, d = distances(spark, centroids, [(1, 263), (263, 1), (100, 200)])
    assert set(out.columns) == {"PULocationID", "DOLocationID", "zone_distance_mi"}
    assert all(v is not None for v in d.values())


@pytest.fixture(scope="session")
def lookup(spark):
    return load_zone_lookup(spark)


def time_rows(spark, stamps):
    df = spark.createDataFrame([(datetime.fromisoformat(t),) for t in stamps], "tpep_pickup_datetime timestamp_ntz")
    out = add_time_features(df)
    return out, {r.tpep_pickup_datetime.isoformat(" ", "minutes"): r for r in out.collect()}


def test_weekday_rush_hour(spark):
    _, r = time_rows(spark, ["2024-01-08 08:30"])  # Monday
    row = r["2024-01-08 08:30"]
    assert (row.pickup_dow, row.is_weekend, row.is_rush_hour, row.pickup_month) == (2, 0, 1, 1)


def test_weekend_is_never_rush_hour(spark):
    _, r = time_rows(spark, ["2024-01-07 08:30"])  # Sunday
    row = r["2024-01-07 08:30"]
    assert (row.pickup_dow, row.is_weekend, row.is_rush_hour) == (1, 1, 0)


def test_rush_hour_boundaries(spark):
    stamps = ["2024-01-08 06:59", "2024-01-08 07:00", "2024-01-08 09:59", "2024-01-08 10:00",
              "2024-01-08 16:00", "2024-01-08 18:59", "2024-01-08 19:00"]
    _, r = time_rows(spark, stamps)
    assert [r[t].is_rush_hour for t in stamps] == [0, 1, 1, 0, 1, 1, 0]


def test_hour_is_cyclical(spark):
    out, r = time_rows(spark, ["2024-01-08 23:00", "2024-01-09 00:00", "2024-01-08 12:00"])
    late, midnight, noon = r["2024-01-08 23:00"], r["2024-01-09 00:00"], r["2024-01-08 12:00"]
    assert math.dist((late.pickup_hour_sin, late.pickup_hour_cos),
                     (midnight.pickup_hour_sin, midnight.pickup_hour_cos)) < 0.27
    assert midnight.pickup_hour_cos == pytest.approx(1)
    assert noon.pickup_hour_cos == pytest.approx(-1)
    assert "_hour" not in out.columns


def zone_rows(spark, lookup, pairs):
    df = spark.createDataFrame(pairs, "PULocationID int, DOLocationID int")
    out = add_zone_attributes(df, lookup)
    return out, {(r.PULocationID, r.DOLocationID): r for r in out.collect()}


def test_airport_to_manhattan(spark, lookup):
    _, r = zone_rows(spark, lookup, [(JFK, MIDTOWN_CENTER)])
    row = r[(JFK, MIDTOWN_CENTER)]
    assert (row.pu_borough, row.pu_service_zone, row.do_borough) == ("Queens", "Airports", "Manhattan")
    assert (row.is_airport_trip, row.same_borough, row.same_zone) == (1, 0, 0)


def test_same_zone_flags(spark, lookup):
    _, r = zone_rows(spark, lookup, [(MIDTOWN_CENTER, MIDTOWN_CENTER)])
    row = r[(MIDTOWN_CENTER, MIDTOWN_CENTER)]
    assert (row.same_zone, row.same_borough, row.is_airport_trip) == (1, 1, 0)


def test_ewr_is_airport(spark, lookup):
    _, r = zone_rows(spark, lookup, [(EWR, TIMES_SQ)])
    row = r[(EWR, TIMES_SQ)]
    assert (row.pu_service_zone, row.is_airport_trip) == ("EWR", 1)


def test_all_zones_have_attributes(spark, lookup):
    out, _ = zone_rows(spark, lookup, [(i, 264 - i) for i in range(1, 264)])
    for c in ("pu_borough", "do_borough", "pu_service_zone", "do_service_zone", "same_borough"):
        assert out.filter(out[c].isNull()).count() == 0, c
