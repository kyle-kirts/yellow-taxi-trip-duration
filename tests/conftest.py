import pytest
from pyspark.sql import SparkSession

from yellow_taxi.weather_clean import RAW_COLUMNS, STATION_ID, VALUE_COLS


@pytest.fixture(scope="session")
def spark():
    s = SparkSession.builder.master("local[1]").appName("tests").getOrCreate()
    yield s
    s.stop()


def weather_report(date, report_type="FM15", source="343", **values):
    """One raw GHCNh row as a dict; measurements get passing quality/source/report-type attributes."""
    row = {"STATION": STATION_ID, "DATE": date}
    for col, value in values.items():
        row[col] = str(value)
        if col in VALUE_COLS:
            row[f"{col}_Quality_Code"] = "5" if source == "343" else "1"
            row[f"{col}_Source_Code"] = source
            row[f"{col}_Report_Type"] = report_type
    return row


def write_weather_psv(directory, rows, year=2024):
    """Write rows to a GHCNh-named .psv in `directory` and return the directory as a file:// URI."""
    lines = ["|".join(RAW_COLUMNS)] + ["|".join(row.get(c, "") for c in RAW_COLUMNS) for row in rows]
    (directory / f"GHCNh_{STATION_ID}_{year}.psv").write_text("\n".join(lines) + "\n")
    return directory.as_uri()
