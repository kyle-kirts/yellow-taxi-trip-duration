import argparse
from datetime import datetime
from pathlib import Path

from pyspark.sql import SparkSession

from yellow_taxi.dataset import MODES, OUTPUT, RAW, build_prepared, write_prepared
from yellow_taxi.log import logger, setup_file_logging

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Clean and feature engineering of raw taxi parquet files. "
                    f"train: {' '.join(RAW['train'])} -> {OUTPUT['train'][0]} (by fold); "
                    f"test: {' '.join(RAW['test'])} -> {OUTPUT['test'][0]} (by year)")
    parser.add_argument("--mode", choices=MODES, default="train",
                        help="train adds CV folds; test does not (default: train)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    setup_file_logging(RESULTS_DIR / f"prepare_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}.log")
    logger.info("prepare (%s): reading %s", args.mode, ", ".join(RAW[args.mode]))
    spark = SparkSession.builder.appName(f"prepare-{args.mode}").getOrCreate()
    try:
        out = write_prepared(build_prepared(spark, args.mode), args.mode)
        logger.info("prepare (%s): wrote %s", args.mode, out)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
