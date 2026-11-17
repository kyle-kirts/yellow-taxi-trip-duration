from pyspark.sql import functions as F

from yellow_taxi.log import log_schema, logger
from yellow_taxi.taxi_clean import log_metrics, rule_counts

STATION_ID = "USW00094728"
VALUE_COLS = ("temperature", "precipitation", "visibility", "wind_speed")
PRES_WX_COLS = ("pres_wx_MW1", "pres_wx_MW2", "pres_wx_AU1", "pres_wx_AU2", "pres_wx_AW1", "pres_wx_AW2")
# Guards against impossible values; units are degrees C, mm, km, m/s.
VALUE_RANGES = {"temperature": (-40, 50), "precipitation": (0, 150), "visibility": (0, 20), "wind_speed": (0, 50)}
BAD_QUALITY = ("2", "3", "6", "7")  # suspect / erroneous
REMOVED_QUALITY_SOURCES = ("220", "221", "222", "223", "347", "348")  # for these sources quality 5 = removed
# Sources whose routine reports omit precipitation when it is dry (GHCNh from Sep 2025 on).
DRY_FILL_SOURCES = ("223", "413")
DRY_FILL_REPORTS = ("FM15", "FM16")
ATTRIBUTES = ("Quality_Code", "Source_Code", "Report_Type")
HELPER_COLS = (
    "STATION", "DATE", "precipitation_Measurement_Code", "source_path",
    *[f"{c}_{a}" for c in VALUE_COLS for a in ATTRIBUTES],
)
RAW_COLUMNS = ("STATION", "DATE", "precipitation_Measurement_Code", *VALUE_COLS,
               *[f"{c}_{a}" for c in VALUE_COLS for a in ATTRIBUTES], *PRES_WX_COLS)
CLEAN_COLUMNS = ("obs_utc", "report_type", "temperature", "precipitation", "precip_trace",
                 "visibility", "wind_speed", *PRES_WX_COLS)


def read_raw_weather(spark, path):
    # quote="" disables quoting so stray quotes in the free-text REM column cannot merge fields.
    df = spark.read.csv(path, sep="|", header=True, quote="")
    return (df.select(*RAW_COLUMNS)
              .withColumn("source_path", F.input_file_name())
              .withColumn("obs_utc", F.col("DATE").try_cast("timestamp_ntz"))
              .withColumns({c: F.col(c).try_cast("double") for c in VALUE_COLS})
              .withColumn("report_type", F.coalesce(*[F.col(f"{c}_Report_Type") for c in VALUE_COLS])))


def file_year():
    return F.regexp_extract("source_path", r"_(\d{4})\.psv$", 1)


def bad_value(col):
    quality = F.col(f"{col}_Quality_Code")
    source = F.col(f"{col}_Source_Code")
    low, high = VALUE_RANGES[col]
    flagged = (quality.isin(*BAD_QUALITY) | quality.rlike("[a-z]")
               | ((quality == "5") & source.isin(*REMOVED_QUALITY_SOURCES)))
    return F.col(col).isNotNull() & (F.coalesce(flagged, F.lit(False)) | ~F.col(col).between(low, high))


def dry_precipitation():
    return (F.col("precipitation").isNull()
            & F.col("report_type").isin(*DRY_FILL_REPORTS)
            & F.col("temperature_Source_Code").isin(*DRY_FILL_SOURCES))


def log_value_fixes(df):
    exprs = [F.sum(F.when(bad_value(c), 1).otherwise(0)).alias(c) for c in VALUE_COLS]
    exprs.append(F.sum(F.when(dry_precipitation(), 1).otherwise(0)).alias("dry_fill"))
    counts = df.agg(*exprs).first().asDict()
    lines = [f"{c + ' -> null':<28}{counts[c] or 0:>10,}" for c in VALUE_COLS]
    lines.append(f"{'precipitation null -> 0.0':<28}{counts['dry_fill'] or 0:>10,}")
    logger.info("--- weather value fixes ---\n%s", "\n".join(lines))


def null_bad_values(df):
    return df.withColumns({c: F.when(~bad_value(c), F.col(c)) for c in VALUE_COLS})


def fill_dry_precipitation(df):
    # Must run on the raw precipitation column: a value nulled for bad quality stays null.
    return df.withColumn("precipitation", F.when(dry_precipitation(), F.lit(0.0)).otherwise(F.col("precipitation")))


def add_precip_trace(df):
    return df.withColumn("precip_trace", F.coalesce(F.col("precipitation_Measurement_Code") == "T", F.lit(False)))


def cleaning_rules():
    has_value = F.lit(False)
    for c in (*VALUE_COLS, *PRES_WX_COLS):
        has_value = has_value | F.col(c).isNotNull()
    return [
        ("dropna",     F.col("obs_utc").isNotNull()),
        ("station",    F.col("STATION") == STATION_ID),
        ("wrong_year", (file_year() == "") | (F.date_format("obs_utc", "yyyy") == file_year())),
        ("no_values",  has_value),
    ]


def clean_weather(df):
    log_schema(df, "raw weather")
    log_value_fixes(df)
    df = add_precip_trace(null_bad_values(fill_dry_precipitation(df)))

    rules = cleaning_rules()
    counts = rule_counts(df, rules)
    raw_rows = counts["raw"]
    metrics = []
    rows = raw_rows
    for name, _ in rules:
        rows -= counts[name] or 0
        metrics.append((name, counts[name] or 0, rows))
    if raw_rows:
        log_metrics(raw_rows, metrics)

    for _, keep in rules:
        df = df.filter(keep)

    df = df.drop(*HELPER_COLS).select(*CLEAN_COLUMNS)
    log_schema(df, "clean weather")
    return df
