import csv
import math
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CENTROIDS_PATH = DATA_DIR / "taxi_zone_centroids.csv"
LOOKUP_PATH = DATA_DIR / "taxi_zone_lookup.csv"
FEET_PER_MILE = 5280
AIRPORT_ZONES = (1, 132, 138)  # EWR, JFK, LaGuardia

CENTROID_SCHEMA = StructType([
    StructField("LocationID", IntegerType(), False),
    StructField("centroid_x_ft", DoubleType(), False),
    StructField("centroid_y_ft", DoubleType(), False),
])

LOOKUP_SCHEMA = StructType([
    StructField("LocationID", IntegerType(), False),
    StructField("Borough", StringType(), False),
    StructField("service_zone", StringType(), False),
])


def load_zone_centroids(spark, path=CENTROIDS_PATH):
    with open(path, newline="") as f:
        rows = [(int(r["LocationID"]), float(r["centroid_x_ft"]), float(r["centroid_y_ft"]))
                for r in csv.DictReader(f)]
    return spark.createDataFrame(rows, CENTROID_SCHEMA)


def add_zone_distance(df, centroids):
    """
    Straight-line miles between pickup and dropoff zone centroids (zone_distance_mi)
    """
    pu = centroids.select(F.col("LocationID").alias("PULocationID"),
                          F.col("centroid_x_ft").alias("_pu_x"), F.col("centroid_y_ft").alias("_pu_y"))
    do = centroids.select(F.col("LocationID").alias("DOLocationID"),
                          F.col("centroid_x_ft").alias("_do_x"), F.col("centroid_y_ft").alias("_do_y"))
    df = (df.join(F.broadcast(pu), "PULocationID", "left")
            .join(F.broadcast(do), "DOLocationID", "left")
            .withColumn("zone_distance_mi",
                        F.hypot(F.col("_pu_x") - F.col("_do_x"), F.col("_pu_y") - F.col("_do_y")) / FEET_PER_MILE)
            .drop("_pu_x", "_pu_y", "_do_x", "_do_y"))

    return df


def load_zone_lookup(spark, path=LOOKUP_PATH):
    with open(path, newline="") as f:
        rows = [(int(r["LocationID"]), r["Borough"], r["service_zone"]) for r in csv.DictReader(f)]
    return spark.createDataFrame(rows, LOOKUP_SCHEMA)


def add_zone_attributes(df, lookup):
    pu = lookup.select(F.col("LocationID").alias("PULocationID"),
                       F.col("Borough").alias("pu_borough"), F.col("service_zone").alias("pu_service_zone"))
    do = lookup.select(F.col("LocationID").alias("DOLocationID"),
                       F.col("Borough").alias("do_borough"), F.col("service_zone").alias("do_service_zone"))
    return (df.join(F.broadcast(pu), "PULocationID", "left")
              .join(F.broadcast(do), "DOLocationID", "left")
              .withColumn("same_zone", (F.col("PULocationID") == F.col("DOLocationID")).cast("int"))
              .withColumn("same_borough", (F.col("pu_borough") == F.col("do_borough")).cast("int"))
              .withColumn("is_airport_trip", (F.col("PULocationID").isin(*AIRPORT_ZONES)
                                              | F.col("DOLocationID").isin(*AIRPORT_ZONES)).cast("int")))


def add_time_features(df, ts="tpep_pickup_datetime"):
    hour = F.col("_hour")
    angle = 2 * math.pi * hour / 24
    weekend = F.col("pickup_dow").isin(1, 7)
    rush = hour.between(7, 9) | hour.between(16, 18)
    return (df.withColumn("_hour", F.hour(ts))
              .withColumn("pickup_hour_sin", F.sin(angle))
              .withColumn("pickup_hour_cos", F.cos(angle))
              .withColumn("pickup_dow", F.dayofweek(ts))
              .withColumn("pickup_month", F.month(ts))
              .withColumn("is_weekend", weekend.cast("int"))
              .withColumn("is_rush_hour", (~weekend & rush).cast("int"))
              .drop("_hour"))


def add_features(df, centroids, lookup):
    df = add_zone_distance(df, centroids)
    df = add_zone_attributes(df, lookup)
    return add_time_features(df)
