from pyspark.sql import functions as F

from yellow_taxi.taxi_features import add_features, load_zone_centroids, load_zone_lookup
from yellow_taxi.taxi_clean import clean
from yellow_taxi.log import log_schema

RAW_ROOT = "/yellow-taxi/raw/"
RAW_TRAIN_PATHS = (RAW_ROOT + "2024/", RAW_ROOT + "2025/")
# TODO: replace with real test years once raw data is added to hdfs (e.g. 2026 + pre-2024)
RAW_TEST_PATHS = (RAW_ROOT + "test-placeholder/",)
PREPARED_ROOT = "/yellow-taxi/prepared/"
TRAIN_PATH = PREPARED_ROOT + "train/"
TEST_PATH = PREPARED_ROOT + "test/"
MODES = ("train", "test")
RAW = {"train": RAW_TRAIN_PATHS, "test": RAW_TEST_PATHS}
OUTPUT = {"train": (TRAIN_PATH, "fold"), "test": (TEST_PATH, "year")}
N_FOLDS = 4
LABEL = "trip_duration_min"


def check_mode(mode):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def read_raw(spark, paths=RAW_TRAIN_PATHS):
    return (spark.read.option("mergeSchema", "true")
                 .parquet(*paths)
                 .withColumn("source_path", F.input_file_name()))


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
