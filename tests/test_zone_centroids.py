import csv

from yellow_taxi_trip_duration.features import CENTROIDS_PATH


def read_rows():
    with open(CENTROIDS_PATH, newline="") as f:
        return list(csv.DictReader(f))


def test_one_row_per_zone():
    ids = [int(r["LocationID"]) for r in read_rows()]
    assert len(ids) == 263
    assert set(ids) == set(range(1, 264))


def test_coordinates_in_nyc_state_plane_bounds():
    for r in read_rows():
        assert 900_000 <= float(r["centroid_x_ft"]) <= 1_070_000, r
        assert 110_000 <= float(r["centroid_y_ft"]) <= 275_000, r
