from datetime import datetime

import pytest
from pyspark.sql.types import StructField, StructType, TimestampNTZType

from conftest import weather_report, write_weather_psv
from yellow_taxi.weather_clean import clean_weather, read_raw_weather
from yellow_taxi.weather_features import HOUR, WEATHER_FEATURES, add_weather, hourly_weather

REPORTS = [
    # 10:00 UTC hour: a special report then the routine one; precipitation is a running total
    weather_report("2024-07-15T10:20:00", report_type="FM16", temperature=20.0, precipitation=3.0,
                   visibility=1.0, pres_wx_AU1="+RA:02"),
    weather_report("2024-07-15T10:51:00", temperature=21.0, precipitation=8.0, visibility=5.0, wind_speed=4.0),
    # 11:00 hour: dry, snow and mist codes only
    weather_report("2024-07-15T11:51:00", temperature=22.0, precipitation=0.0, visibility=16.0,
                   pres_wx_AW1="SN:71", pres_wx_AW2="BR:10"),
    # 12:00 hour: trace precipitation, no temperature
    weather_report("2024-07-15T12:51:00", precipitation=0.0, precipitation_Measurement_Code="T"),
    # 13:00, 14:00 and 15:00 hours: nothing reported
    weather_report("2024-07-15T16:51:00", temperature=30.0, precipitation=0.0, visibility=16.0),
]


def utc(hour):
    return datetime(2024, 7, 15, hour)


@pytest.fixture(scope="module")
def weather(spark, tmp_path_factory):
    path = write_weather_psv(tmp_path_factory.mktemp("weather"), REPORTS)
    return hourly_weather(clean_weather(read_raw_weather(spark, path))).cache()


@pytest.fixture(scope="module")
def hours(weather):
    return {r[HOUR]: r for r in weather.collect()}


def test_one_row_per_hour_labelled_by_hour_end(weather, hours):
    assert tuple(weather.columns) == (HOUR, *WEATHER_FEATURES)
    # first report 10:20 -> hour ending 11:00; last report 16:51 -> hour ending 17:00
    assert sorted(hours) == [utc(h) for h in range(11, 18)]


def test_collapse_takes_last_value_and_max_precipitation(hours):
    row = hours[utc(11)]
    assert (row.temp_c, row.visibility_km, row.wind_speed_ms) == (21.0, 5.0, 4.0)
    assert row.precip_mm_last_hr == 8.0
    assert (row.is_raining, row.heavy_rain, row.is_snowing, row.is_fog, row.low_visibility) == (1, 1, 0, 0, 0)


def test_present_weather_codes(hours):
    row = hours[utc(12)]
    assert (row.is_raining, row.is_snowing, row.is_fog, row.heavy_rain) == (0, 1, 1, 0)


def test_trace_counts_as_rain(hours):
    assert hours[utc(13)].is_raining == 1
    assert hours[utc(13)].precip_mm_last_hr == 0.0


def test_forward_fill_stops_after_two_hours(hours):
    # temperature was last reported in the hour ending 12:00
    assert hours[utc(13)].temp_c == 22.0
    assert hours[utc(14)].temp_c == 22.0
    assert hours[utc(15)].temp_c is None
    # precipitation and flags are never carried forward
    assert hours[utc(14)].precip_mm_last_hr is None
    assert hours[utc(14)].is_raining is None


def test_three_hour_precipitation(hours):
    assert hours[utc(11)].precip_mm_last_3hr == 8.0
    assert hours[utc(13)].precip_mm_last_3hr == 8.0   # 8.0 + 0.0 + 0.0
    assert hours[utc(14)].precip_mm_last_3hr == 0.0   # the 8.0 has rolled out of the window


def test_join_converts_pickup_to_utc(spark, weather):
    schema = StructType([StructField("tpep_pickup_datetime", TimestampNTZType())])
    pickups = [
        datetime(2024, 7, 15, 7, 15),   # EDT = 11:15 UTC -> row for 10:00-10:59 UTC
        datetime(2024, 7, 15, 6, 59),   # 10:59 UTC -> before the first weather hour
        datetime(2024, 7, 15, 11, 30),  # 15:30 UTC -> gap hour, past the forward-fill limit
    ]
    trips = spark.createDataFrame([(p,) for p in pickups], schema)
    df = add_weather(trips, weather)
    assert HOUR not in df.columns
    rows = {r.tpep_pickup_datetime: r for r in df.collect()}
    assert len(rows) == 3
    assert (rows[pickups[0]].temp_c, rows[pickups[0]].precip_mm_last_hr, rows[pickups[0]].weather_missing) == (21.0, 8.0, 0)
    assert (rows[pickups[1]].temp_c, rows[pickups[1]].weather_missing) == (None, 1)
    assert (rows[pickups[2]].temp_c, rows[pickups[2]].weather_missing) == (None, 1)


def test_join_uses_standard_time_in_winter(spark, tmp_path):
    path = write_weather_psv(tmp_path, [weather_report("2024-01-10T12:51:00", temperature=1.0)])
    winter = hourly_weather(clean_weather(read_raw_weather(spark, path)))
    schema = StructType([StructField("tpep_pickup_datetime", TimestampNTZType())])
    trips = spark.createDataFrame([(datetime(2024, 1, 10, 8, 15),)], schema)  # EST = 13:15 UTC
    assert add_weather(trips, winter).first().temp_c == 1.0
