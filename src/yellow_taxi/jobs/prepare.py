from datetime import datetime
from pathlib import Path

from pyspark.sql import SparkSession

from yellow_taxi.dataset import PREPARED_PATH, RAW_PATHS, build_prepared, write_prepared
from yellow_taxi.log import logger, setup_file_logging

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"


def main():
    setup_file_logging(RESULTS_DIR / f"prepare_{datetime.now():%Y%m%d_%H%M%S}.log")
    logger.info("prepare: reading %s", ", ".join(RAW_PATHS))
    spark = SparkSession.builder.appName("prepare").getOrCreate()
    write_prepared(build_prepared(spark))
    logger.info("prepare: wrote %s", PREPARED_PATH)
    spark.stop()


if __name__ == "__main__":
    main()
