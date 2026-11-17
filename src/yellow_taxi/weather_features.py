from pyspark.sql import Window
from pyspark.sql import functions as F

from yellow_taxi.log import log_schema
from yellow_taxi.weather_clean import PRES_WX_COLS

NYC_TZ = "America/New_York"
HOUR = "weather_hour"
LAST_VALUE = {"temp_c": "temperature", "visibility_km": "visibility", "wind_speed_ms": "wind_speed"}
FFILL_HOURS = 2
HEAVY_RAIN_MM = 7.6      # 0.3 in/h
LOW_VISIBILITY_KM = 1.6  # 1 mile
RAIN_CODES, SNOW_CODES, FOG_CODES = "RA|DZ", "SN", "FG|BR"
WEATHER_FEATURES = (
    "temp_c", "visibility_km", "wind_speed_ms", "precip_mm_last_hr", "precip_mm_last_3hr",
    "is_raining", "heavy_rain", "is_snowing", "is_fog", "low_visibility",
)


def has_code(pattern):
    """True when any weather column of the report contains one of the codes."""
    hit = F.lit(False)
    for c in PRES_WX_COLS:
        hit = hit | F.coalesce(F.col(c).rlike(pattern), F.lit(False))
    return hit


def last_non_null(col):
    return F.max(F.when(F.col(col).isNotNull(), F.struct("obs_utc", col)))[col]


def collapse_to_hours(df):
    wet = (F.coalesce(F.col("precipitation") > 0, F.lit(False)) | F.col("precip_trace") | has_code(RAIN_CODES))
    return (df.withColumn(HOUR, F.date_trunc("hour", "obs_utc") + F.expr("INTERVAL 1 HOUR"))
              .groupBy(HOUR)
              .agg(*[last_non_null(src).alias(out) for out, src in LAST_VALUE.items()],
                   # reports within an hour carry running totals, so the max is the hour's total
                   F.max("precipitation").alias("precip_mm_last_hr"),
                   F.max(wet.cast("int")).alias("is_raining"),
                   F.max(has_code(SNOW_CODES).cast("int")).alias("is_snowing"),
                   F.max(has_code(FOG_CODES).cast("int")).alias("is_fog")))


def fill_hours(hourly):
    """One row for every hour between the first and last, so the row windows below span real time."""
    calendar = (hourly.agg(F.min(HOUR).alias("lo"), F.max(HOUR).alias("hi"))
                      .select(F.explode(F.sequence("lo", "hi", F.expr("INTERVAL 1 HOUR"))).alias(HOUR)))
    return calendar.join(hourly, HOUR, "left")


def hourly_weather(df):
    """Cleaned weather observations -> one row per weather_hour (UTC) with WEATHER_FEATURES."""
    hourly = fill_hours(collapse_to_hours(df))
    recent = Window.orderBy(HOUR).rowsBetween(-FFILL_HOURS, 0)
    hourly = (hourly
              .withColumns({c: F.last(c, ignorenulls=True).over(recent) for c in LAST_VALUE})
              .withColumn("precip_mm_last_3hr", F.sum(F.coalesce("precip_mm_last_hr", F.lit(0.0))).over(recent))
              .withColumn("heavy_rain", (F.col("precip_mm_last_hr") >= HEAVY_RAIN_MM).cast("int"))
              .withColumn("low_visibility", (F.col("visibility_km") < LOW_VISIBILITY_KM).cast("int"))
              .select(HOUR, *WEATHER_FEATURES))
    log_schema(hourly, "hourly weather")
    return hourly


def add_weather(trips, weather, ts="tpep_pickup_datetime"):
    pickup_utc = F.convert_timezone(F.lit(NYC_TZ), F.lit("UTC"), F.col(ts))
    df = (trips.withColumn(HOUR, F.date_trunc("hour", pickup_utc))
               .join(F.broadcast(weather), HOUR, "left")
               .drop(HOUR)
               .withColumn("weather_missing", F.col("temp_c").isNull().cast("int")))
    log_schema(df, "add_weather")
    return df
