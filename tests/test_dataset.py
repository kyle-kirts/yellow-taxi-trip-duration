from datetime import datetime

import pytest
from pyspark.sql.types import (DoubleType, IntegerType, LongType, StringType, StructField,
                               StructType, TimestampNTZType)

from yellow_taxi.dataset import RAW_COLUMNS, build_prepared, read_raw

PICKUP = datetime(2022, 1, 5, 8, 0)
DROPOFF = datetime(2022, 1, 5, 8, 20)
COMMON = [
    StructField("tpep_pickup_datetime", TimestampNTZType()),
    StructField("tpep_dropoff_datetime", TimestampNTZType()),
    StructField("trip_distance", DoubleType()),
    StructField("payment_type", LongType()),
    StructField("fare_amount", DoubleType()),
    StructField("total_amount", DoubleType()),
]


def old_schema():
    # 2022-01 .. 2023-01 raw schema: bigint IDs, double counts, lowercase airport_fee
    return StructType([
        StructField("VendorID", LongType()), StructField("passenger_count", DoubleType()),
        StructField("RatecodeID", DoubleType()), StructField("PULocationID", LongType()),
        StructField("DOLocationID", LongType()), StructField("airport_fee", DoubleType()), *COMMON,
    ])


def new_schema():
    # 2026-07+ raw schema: int IDs, bigint counts, extra request_source column
    return StructType([
        StructField("VendorID", IntegerType()), StructField("passenger_count", LongType()),
        StructField("RatecodeID", LongType()), StructField("PULocationID", IntegerType()),
        StructField("DOLocationID", IntegerType()), StructField("request_source", StringType()), *COMMON,
    ])


@pytest.fixture(scope="module")
def raw_dirs(spark, tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    old = root / "old"
    new = root / "new"
    spark.createDataFrame(
        [(1, 1.0, 1.0, 161, 236, 0.0, PICKUP, DROPOFF, 2.5, 1, 15.0, 20.0),
         (2, 2.0, 1.0, 132, 161, 1.75, PICKUP, DROPOFF, 17.0, 1, 50.0, 70.0)],
        old_schema()).write.parquet(old.as_uri())
    spark.createDataFrame(
        [(2, 1, 1, 236, 161, "app", PICKUP.replace(year=2026), DROPOFF.replace(year=2026), 2.5, 1, 15.0, 20.0)],
        new_schema()).write.parquet(new.as_uri())
    return old.as_uri(), new.as_uri()


def test_read_raw_mixed_schemas(spark, raw_dirs):
    df = read_raw(spark, raw_dirs)
    assert df.count() == 3
    types = dict((f.name, f.dataType.simpleString()) for f in df.schema)
    assert types == {**RAW_COLUMNS, "source_path": "string"}


def test_test_mode_adds_year(spark, raw_dirs):
    df = build_prepared(spark, "test", raw_paths=raw_dirs)
    assert "year" in df.columns and "fold" not in df.columns
    assert sorted(r.year for r in df.select("year").distinct().collect()) == [2022, 2026]
