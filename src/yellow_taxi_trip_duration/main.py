from pyspark.sql import SparkSession

from yellow_taxi_trip_duration.taxi_clean import clean

def main():
    spark = SparkSession.builder.appName("clean").getOrCreate()

    df = (spark.read.option("mergeSchema", "true")
            .parquet("/yellow-taxi/2024/", "/yellow-taxi/2025/")
            .withColumn("source_path", F.input_file_name()))

    df = clean(df)

    spark.stop()

if __name__ == "__main__":
    main()