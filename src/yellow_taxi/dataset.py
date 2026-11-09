from pyspark.sql import functions as F

from yellow_taxi.taxi_features import add_features, load_zone_centroids, load_zone_lookup
from yellow_taxi.taxi_clean import clean
from yellow_taxi.log import log_schema

RAW_PATHS = ("/yellow-taxi/2024/", "/yellow-taxi/2025/")
PREPARED_PATH = "/yellow-taxi/prepared/"
N_FOLDS = 4
LABEL = "trip_duration_min"


def read_raw(spark, paths=RAW_PATHS):
    return (spark.read.option("mergeSchema", "true")
                 .parquet(*paths)
                 .withColumn("source_path", F.input_file_name()))


def build_prepared(spark, raw_paths=RAW_PATHS):
    df = clean(read_raw(spark, raw_paths))
    df = add_features(df, load_zone_centroids(spark), load_zone_lookup(spark))
    # CV fold by pickup month: 0 = Jan/May/Sep, 1 = Feb/Jun/Oct, 2 = Mar/Jul/Nov, 3 = Apr/Aug/Dec
    df = df.withColumn("fold", ((F.month("tpep_pickup_datetime") - 1) % N_FOLDS).cast("int"))
    log_schema(df, "add fold")
    return df


def write_prepared(df, out_path=PREPARED_PATH):
    df.write.mode("overwrite").partitionBy("fold").parquet(out_path)


def load_prepared(spark, path=PREPARED_PATH):
    return spark.read.parquet(path)
