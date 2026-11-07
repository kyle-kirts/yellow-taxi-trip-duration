import pytest
from pyspark.sql import SparkSession

from yellow_taxi_trip_duration.features import add_zone_distance, load_zone_centroids

JFK, LGA, MIDTOWN_CENTER, TIMES_SQ = 132, 138, 161, 230


@pytest.fixture(scope="session")
def spark():
    s = SparkSession.builder.master("local[1]").appName("tests").getOrCreate()
    yield s
    s.stop()


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


def test_log_variant(spark, centroids):
    df = spark.createDataFrame([(JFK, LGA)], "PULocationID int, DOLocationID int")
    assert "log_zone_distance_mi" in add_zone_distance(df, centroids, log=True).columns
