import csv
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

CENTROIDS_PATH = Path(__file__).resolve().parents[2] / "data" / "taxi_zone_centroids.csv"
FEET_PER_MILE = 5280

CENTROID_SCHEMA = StructType([
    StructField("LocationID", IntegerType(), False),
    StructField("centroid_x_ft", DoubleType(), False),
    StructField("centroid_y_ft", DoubleType(), False),
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
