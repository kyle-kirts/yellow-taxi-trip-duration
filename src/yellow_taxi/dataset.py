from functools import reduce

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from yellow_taxi.taxi_features import add_features, load_zone_centroids, load_zone_lookup
from yellow_taxi.taxi_clean import clean
from yellow_taxi.log import log_schema

RAW_ROOT = "/yellow-taxi/raw/"
RAW_TRAIN_PATHS = (RAW_ROOT + "2024/", RAW_ROOT + "2025/")
RAW_TEST_PATHS = (RAW_ROOT + "2022/", RAW_ROOT + "2023/", RAW_ROOT + "2026/")
PREPARED_ROOT = "/yellow-taxi/prepared/"
TRAIN_PATH = PREPARED_ROOT + "train/"
TEST_PATH = PREPARED_ROOT + "test/"
MODES = ("train", "test")
RAW = {"train": RAW_TRAIN_PATHS, "test": RAW_TEST_PATHS}
OUTPUT = {"train": (TRAIN_PATH, "fold"), "test": (TEST_PATH, "year")}
N_FOLDS = 4
LABEL = "trip_duration_min"
RAW_COLUMNS = {
    "VendorID": "int", "tpep_pickup_datetime": "timestamp_ntz", "tpep_dropoff_datetime": "timestamp_ntz",
    "passenger_count": "bigint", "trip_distance": "double", "RatecodeID": "bigint",
    "PULocationID": "int", "DOLocationID": "int", "payment_type": "bigint",
    "fare_amount": "double", "total_amount": "double",
}


def check_mode(mode):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def read_raw_file(spark, path):
    return (spark.read.parquet(path)
                 .select(*[F.col(c).try_cast(t).alias(c) for c, t in RAW_COLUMNS.items()])
                 .withColumn("source_path", F.input_file_name()))


def read_raw(spark, paths=RAW_TRAIN_PATHS):
    files = sorted(spark.read.parquet(*paths).inputFiles())
    return reduce(DataFrame.unionByName, [read_raw_file(spark, f) for f in files])


def build_features(spark, raw_paths=RAW_TRAIN_PATHS):
    df = clean(read_raw(spark, raw_paths))
    return add_features(df, load_zone_centroids(spark), load_zone_lookup(spark))


def add_fold(df):
    # CV fold by pickup month: 0 = Jan/May/Sep, 1 = Feb/Jun/Oct, 2 = Mar/Jul/Nov, 3 = Apr/Aug/Dec
    df = df.withColumn("fold", ((F.month("tpep_pickup_datetime") - 1) % N_FOLDS).cast("int"))
    log_schema(df, "add fold")
    return df


def add_year(df):
    df = df.withColumn("year", F.year("tpep_pickup_datetime").cast("int"))
    log_schema(df, "add year")
    return df


def build_prepared(spark, mode="train", raw_paths=None):
    check_mode(mode)
    df = build_features(spark, raw_paths or RAW[mode])
    return add_fold(df) if mode == "train" else add_year(df)


def write_prepared(df, mode="train", out_path=None):
    check_mode(mode)
    default_path, partition_col = OUTPUT[mode]
    writer = df.write.mode("overwrite").partitionBy(partition_col)
    if mode == "test":
        writer = writer.option("partitionOverwriteMode", "dynamic")
    writer.parquet(out_path or default_path)
    return out_path or default_path


def load_prepared(spark, mode="train", path=None):
    check_mode(mode)
    return spark.read.parquet(path or OUTPUT[mode][0])
