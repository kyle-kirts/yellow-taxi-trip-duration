"""Converts the TLC taxi-zone shapefile to a centroid CSV.

The shapefile is in NY State Plane Long Island (EPSG:2263, US survey feet), so
centroids are planar coordinates in feet.

"""
import csv
from pathlib import Path

import shapefile
from shapely.geometry import shape

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SHP_PATH = DATA_DIR / "taxi_zones" / "taxi_zones.shp"
OUT_PATH = DATA_DIR / "taxi_zone_centroids.csv"
EXPECTED_IDS = set(range(1, 264))


def zone_centroids(shp_path=SHP_PATH):
    rows = {}
    with shapefile.Reader(str(shp_path)) as reader:
        for sr in reader.iterShapeRecords():
            loc_id = int(sr.record["LocationID"])
            if loc_id in rows:
                raise ValueError(f"duplicate LocationID {loc_id}")
            c = shape(sr.shape.__geo_interface__).centroid
            # -> input interface .__geo_interface__ {'type': 'Polygon',
            #     'coordinates': [[(986752.6, 210853.7), (986627.9, 210628.5), ...]]}
            # -> shape() a shapely Polygon (or MultiPolygon) object, call .centroid to find its center
            rows[loc_id] = (round(c.x, 1), round(c.y, 1))
    if set(rows) != EXPECTED_IDS:
        raise ValueError(f"LocationIDs mismatch: missing={sorted(EXPECTED_IDS - set(rows))}, "
                         f"extra={sorted(set(rows) - EXPECTED_IDS)}")
    return rows


def main():
    rows = zone_centroids()
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["LocationID", "centroid_x_ft", "centroid_y_ft"])
        for loc_id in sorted(rows):
            w.writerow([loc_id, *rows[loc_id]])
    print(f"wrote {len(rows)} centroids to {OUT_PATH}")


if __name__ == "__main__":
    main()
