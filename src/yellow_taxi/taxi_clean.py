from functools import reduce

from pyspark.sql import functions as F

from yellow_taxi.log import log_schema, logger

DROPPED = (
    'Airport_fee', 'congestion_surcharge', 'improvement_surcharge',
    'tolls_amount', 'extra', 'mta_tax', 'tip_amount', 'store_and_fwd_flag',
    'cbd_congestion_fee',
)
CLEAN_DROP = ('total_amount', 'payment_type', 'fare_amount', 'tpep_dropoff_datetime', 'source_path')
FEATURE_COLS = (
    'tpep_pickup_datetime', 'tpep_dropoff_datetime',
    'PULocationID', 'DOLocationID',
    'trip_distance', 'passenger_count', 'RatecodeID', 'VendorID',
)


def add_label(df):
    return df.withColumn(
        "trip_duration_min",
        (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")) / 60,
    )


def cleaning_rules():
    return [
        ("wrong_month",     F.date_format("tpep_pickup_datetime", "yyyy-MM") == F.regexp_extract("source_path", r"(\d{4}-\d{2})\.parquet$", 1)),
        ("duration",        F.col("trip_duration_min").between(1, 240)),
        ("trip_distance",   (F.col("trip_distance") > 0) & (F.col("trip_distance") <= 100)),
        ("speed",           F.try_divide(F.col("trip_distance"), F.col("trip_duration_min") / 60) <= 80),
        ("passenger_count", F.col("passenger_count").between(1, 6)),
        ("RatecodeID",      F.col("RatecodeID").between(1, 6)),
        ("VendorID",        F.col("VendorID").isin(1, 2, 6, 7)),
        ("PULocationID",    ~F.col("PULocationID").isin(264, 265)),
        ("DOLocationID",    ~F.col("DOLocationID").isin(264, 265)),
        ("payment_type",    ~F.col("payment_type").isin(3, 4, 6)),
        ("fare_amount",     F.col("fare_amount") > 0),
        ("total_amount",    F.col("total_amount") > 0),
    ]


def not_null_rule():
    return ("dropna", reduce(lambda a, b: a & b, [F.col(c).isNotNull() for c in (*FEATURE_COLS, *CLEAN_DROP)]))


def rule_counts(df, rules):
    passed = F.lit(True)
    exprs = [F.count(F.lit(1)).alias("raw")]
    for name, keep in rules:
        keep = F.coalesce(keep, F.lit(False))
        exprs.append(F.sum(F.when(passed & ~keep, 1).otherwise(0)).alias(name))
        passed = passed & keep
    return df.agg(*exprs).first().asDict()


def log_metrics(raw_rows, metrics):
    lines = [
        f"{'step':<18}{'removed':>14}{'remaining':>14}{'% of raw':>10}",
        f"{'raw':<18}{'':>14}{raw_rows:>14,}",
    ]
    for name, removed, remaining in metrics:
        lines.append(f"{name:<18}{removed:>14,}{remaining:>14,}{removed / raw_rows:>10.2%}")
    final_rows = metrics[-1][2]
    total_removed = raw_rows - final_rows
    lines.append(f"{'TOTAL':<18}{total_removed:>14,}{final_rows:>14,}{total_removed / raw_rows:>10.2%}")
    logger.info("--- cleaning rules ---\n%s", "\n".join(lines))


def clean(df):
    log_schema(df, "raw")
    df = df.drop(*DROPPED)
    df = add_label(df)
    log_schema(df, "drop unused columns + add label")
    rules = [not_null_rule(), *cleaning_rules()]

    counts = rule_counts(df, rules)
    raw_rows = counts["raw"]
    metrics = []
    rows = raw_rows
    for name, _ in rules:
        rows -= counts[name]
        metrics.append((name, counts[name], rows))
    log_metrics(raw_rows, metrics)

    for name, keep in rules:
        df = df.filter(keep)

    df = df.drop(*CLEAN_DROP)
    log_schema(df, "clean")
    return df

