from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from yellow_taxi_trip_duration.features import add_zone_distance, load_zone_centroids
from yellow_taxi_trip_duration.taxi_clean import clean

def main():
    spark = SparkSession.builder.appName("clean").getOrCreate()

    df = (spark.read.option("mergeSchema", "true")
            .parquet("/yellow-taxi/2024/", "/yellow-taxi/2025/")
            .withColumn("source_path", F.input_file_name()))

    df = clean(df)
    df = add_zone_distance(df, load_zone_centroids(spark))
    df.show()

    spark.stop()

if __name__ == "__main__":
    main()