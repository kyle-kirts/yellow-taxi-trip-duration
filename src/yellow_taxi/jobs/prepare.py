from pyspark.sql import SparkSession

from yellow_taxi.dataset import build_prepared, write_prepared


def main():
    spark = SparkSession.builder.appName("prepare").getOrCreate()
    write_prepared(build_prepared(spark))
    spark.stop()


if __name__ == "__main__":
    main()
