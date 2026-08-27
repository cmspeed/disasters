import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from osgeo import ogr, osr

logger = logging.getLogger(__name__)


def parse_bbox_input(bbox_string: str) -> list[float] | str:
    """
    Parses a string input (KML, GeoJSON, WKT, or 4 coordinates)
    into a standardized [South, North, West, East] bounding box list,
    OR returns the original string (WKT/filepath) to preserve precise AOI shapes downstream.
    """
    logger = logging.getLogger(__name__)

    # Check if it's a WKT string
    if bbox_string.upper().startswith(("POLYGON", "MULTIPOLYGON", "BBOX")):
        logger.info("Detected WKT string. Preserving complex geometry...")
        return bbox_string

    # Check if it's a web-hosted AOI file
    if bbox_string.lower().startswith(("http://", "https://")):
        logger.info(f"Detected URL for AOI. Preserving geometry from: {bbox_string}")
        return bbox_string

    # Check if it's a geospatial file-type (KML, GeoJSON, SHP)
    if os.path.isfile(bbox_string):
        logger.info(
            f"Detected file path for AOI. Preserving geometry from: {bbox_string}"
        )
        return bbox_string

    # Assume it's a raw coordinate string
    logger.info("Parsing raw coordinates...")
    coords = [float(x) for x in bbox_string.replace(",", " ").split()]
    if len(coords) != 4:
        raise ValueError(
            "Bounding box must be a valid file, WKT, or 4 space/comma separated coordinates."
        )

    s, n, w, e = coords

    swapped = False

    # Auto-swap S/N if flipped
    if s > n:
        logger.warning("South coordinate is greater than North. Auto-swapping...")
        s, n = n, s
        swapped = True

    # Auto-swap W/E if flipped (protect Antimeridian crossings)
    if w > e:
        # A true antimeridian box has a positive West, negative East, and a large numerical gap
        if w > 0 and e < 0 and (w - e) > 180:
            logger.info(
                "Detected valid bounding box crossing the Antimeridian. Preserving coordinates."
            )
        else:
            logger.warning("West coordinate is greater than East. Auto-swapping...")
            w, e = e, w
            swapped = True

    if swapped:
        logger.info(f"Corrected bounding box to [S, N, W, E]: {[s, n, w, e]}")

    return [s, n, w, e]


def ensure_directory(output_dir: Path) -> Path:
    """
    Create the output directory if it does not exist.

    Args:
        output_dir (Path): Path to the output directory.

    Returns:
        Path: The validated directory path.

    Raises:
        Exception: If the directory cannot be created.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Could not create output directory: {e}")
        raise
    return output_dir


def scan_local_directory(local_dir: Path) -> pd.DataFrame:
    """
    Scans a local directory for OPERA Geotiffs, parses their filenames,
    and constructs a DataFrame mimicking the structure of 'opera_products_metadata.xls'.

    Args:
        local_dir (Path): Path to the local directory containing valid OPERA GeoTIFF products.

    Returns:
        pd.DataFrame: DataFrame containing metadata extracted from file names.
    """
    # Scan for all TIF files recursively
    tif_files = list(local_dir.rglob("*.tif"))

    # Check if directory is empty or has no TIFs
    if not tif_files:
        logger.error(f"No .tif files found in {local_dir}.")
        logger.info(
            "Please ensure your local directory contains valid OPERA GeoTIFF products."
        )
        logger.info("The script expects files like: OPERA_L3_DSWx-HLS_..._WTR.tif")
        return pd.DataFrame()

    logger.info(f"Scanning {len(tif_files)} local files in {local_dir}...")

    # Dictionary to hold grouped granule data
    granules = defaultdict(dict)

    # Map filename prefixes to OPERA Dataset IDs
    product_map = {
        "OPERA_L3_DSWX-HLS": "OPERA_L3_DSWX-HLS_V1",
        "OPERA_L3_DSWx-HLS": "OPERA_L3_DSWX-HLS_V1",
        "OPERA_L3_DSWX-S1": "OPERA_L3_DSWX-S1_V1",
        "OPERA_L3_DSWx-S1": "OPERA_L3_DSWX-S1_V1",
        "OPERA_L3_DIST-ALERT-HLS": "OPERA_L3_DIST-ALERT-HLS_V1",
        "OPERA_L3_DIST-ALERT-S1": "OPERA_L3_DIST-ALERT-S1_V1",
        "OPERA_L2_RTC-S1": "OPERA_L2_RTC-S1_V1",
        "OPERA_L3_DSWX-NI": "OPERA_L3_DSWX-NI_V1",
        "OPERA_L3_DSWx-NI": "OPERA_L3_DSWX-NI_V1",
    }

    files_processed_count = 0

    for f in tif_files:
        name = f.name

        # Identify product type
        prod_key = None
        for key in product_map.keys():
            if name.startswith(key):
                prod_key = key
                break

        if not prod_key:
            # Skip non-OPERA files
            continue

        dataset_name = product_map[prod_key]

        # Extract Date and Tile ID
        parts = name.split("_")
        date_str = None
        tile_id = "UNKNOWN"

        for i, part in enumerate(parts):
            if re.match(r"\d{8}T\d{6}Z", part):
                date_str = part
                if i > 0:
                    tile_id = parts[i - 1]
                break

        if not date_str:
            continue

        # Identify layer type
        layer_col = None

        # DSWx layers
        if name.endswith("WTR.tif") and "BWTR" not in name:
            layer_col = "WTR"
        elif name.endswith("BWTR.tif"):
            layer_col = "BWTR"
        elif name.endswith("CONF.tif") and "VEG-DIST" not in name:
            layer_col = "CONF"

        # DIST layers
        elif "VEG-ANOM-MAX" in name:
            layer_col = "VEG-ANOM-MAX"
        elif "VEG-DIST-STATUS" in name:
            layer_col = "VEG-DIST-STATUS"
        elif "VEG-DIST-DATE" in name:
            layer_col = "VEG-DIST-DATE"
        elif "VEG-DIST-CONF" in name:
            layer_col = "VEG-DIST-CONF"

        # RTC layers
        elif name.endswith("_VV.tif"):
            layer_col = "RTC-VV"
        elif name.endswith("_VH.tif"):
            layer_col = "RTC-VH"

        # Fallback
        else:
            suffix = parts[-1].replace(".tif", "")
            if suffix.isupper():
                layer_col = suffix

        if not layer_col:
            continue

        # Group by Unique Key (Dataset, Date, Tile)
        group_key = (dataset_name, date_str, tile_id)

        # Determine column name expected by generate_products()
        col_name = f"Download URL {layer_col}"

        granules[group_key][col_name] = str(f.absolute())
        granules[group_key]["Start Time"] = date_str
        granules[group_key]["Dataset"] = dataset_name

        files_processed_count += 1

    # Final check
    if not granules:
        logger.error(
            f"Found {len(tif_files)} files in {local_dir}, but none matched expected OPERA filename patterns."
        )
        return pd.DataFrame()

    # Convert to DataFrame
    rows = []
    for key, data in granules.items():
        rows.append(data)

    df = pd.DataFrame(rows)
    df["Start Time"] = pd.to_datetime(
        df["Start Time"], format="%Y%m%dT%H%M%SZ", errors="coerce"
    )

    logger.info(
        f"Constructed local metadata DataFrame with {len(df)} unique granules (from {files_processed_count} files)."
    )
    return df


def cleanup_temp_file(filepath: Path) -> None:
    """
    Safely remove the temporary file.

    Args:
        filepath (Path): Path to the temporary file to be removed.
    """
    if filepath.exists():
        try:
            os.remove(filepath)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file {filepath}: {e}")


def write_json(data: dict, filepath: Path) -> None:
    """
    Write a dictionary to a JSON file.

    Args:
        data (dict): Data to be written.
        filepath (Path): Output path for the JSON file.
    """
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def export_aoi(bbox: list[float], output_dir: Path) -> None:
    """
    Export the Area of Interest (AOI) bounding box as a GeoJSON and Shapefile.

    Args:
        bbox (list[float]): Bounding box in [S, N, W, E] format (miny, maxy, minx, maxx).
        output_dir (Path): Directory to save the exported files.
    """
    logger = logging.getLogger(__name__)

    miny, maxy, minx, maxx = bbox

    # Validate EPSG:4326 coordinate ranges
    if not (-90 <= miny <= 90 and -90 <= maxy <= 90):
        raise ValueError(
            f"Latitude values must be in [-90, 90]. Got miny={miny}, maxy={maxy}"
        )
    if not (-180 <= minx <= 180 and -180 <= maxx <= 180):
        raise ValueError(
            f"Longitude values must be in [-180, 180]. Got minx={minx}, maxx={maxx}"
        )
    if miny > maxy:
        raise ValueError(f"miny ({miny}) cannot be greater than maxy ({maxy})")

    # Filename formatting
    def format_coord(val, pos_dir, neg_dir):
        direction = pos_dir if val >= 0 else neg_dir
        return f"{abs(val):.6f}{direction}".replace(".", "p")

    lat_s = format_coord(miny, "N", "S")
    lat_n = format_coord(maxy, "N", "S")
    lon_w = format_coord(minx, "E", "W")
    lon_e = format_coord(maxx, "E", "W")

    name = f"AOI_{lat_s}_{lat_n}_{lon_w}_{lon_e}"

    # Export as GeoJSON into /geojson
    geojson_dir = output_dir / "geojson"
    geojson_path = geojson_dir / f"{name}.geojson"

    if minx > maxx:
        geometry = {
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [
                        [minx, miny],
                        [180.0, miny],
                        [180.0, maxy],
                        [minx, maxy],
                        [minx, miny],
                    ]
                ],
                [
                    [
                        [-180.0, miny],
                        [maxx, miny],
                        [maxx, maxy],
                        [-180.0, maxy],
                        [-180.0, miny],
                    ]
                ],
            ],
        }
    else:
        geometry = {
            "type": "Polygon",
            "coordinates": [
                [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]
            ],
        }

    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Area of Interest"},
                "geometry": geometry,
            }
        ],
    }

    try:
        geojson_dir.mkdir(parents=True, exist_ok=True)
        with open(geojson_path, "w") as f:
            json.dump(geojson_data, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to export AOI GeoJSON: {e}")

    # Export as Shapefile into /shp using GDAL/OGR natively
    shp_dir = output_dir / "shp"
    shp_path = shp_dir / f"{name}.shp"
    ds = None
    feature = None
    try:
        shp_dir.mkdir(parents=True, exist_ok=True)
        driver = ogr.GetDriverByName("ESRI Shapefile")
        if shp_path.exists():
            driver.DeleteDataSource(str(shp_path))

        ds = driver.CreateDataSource(str(shp_path))
        if ds is None:
            logger.warning(f"Could not create shapefile at {shp_path}")
            return

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)

        geom_type = ogr.wkbMultiPolygon if minx > maxx else ogr.wkbPolygon
        layer = ds.CreateLayer("AOI", srs, geom_type)

        field_defn = ogr.FieldDefn("Name", ogr.OFTString)
        field_defn.SetWidth(50)
        layer.CreateField(field_defn)

        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("Name", "Area of Interest")

        if minx > maxx:
            multi = ogr.Geometry(ogr.wkbMultiPolygon)

            # Western half
            ring1 = ogr.Geometry(ogr.wkbLinearRing)
            for pt in [
                (minx, miny),
                (180.0, miny),
                (180.0, maxy),
                (minx, maxy),
                (minx, miny),
            ]:
                ring1.AddPoint_2D(*pt)
            poly1 = ogr.Geometry(ogr.wkbPolygon)
            poly1.AddGeometry(ring1)
            multi.AddGeometry(poly1)

            # Eastern half
            ring2 = ogr.Geometry(ogr.wkbLinearRing)
            for pt in [
                (-180.0, miny),
                (maxx, miny),
                (maxx, maxy),
                (-180.0, maxy),
                (-180.0, miny),
            ]:
                ring2.AddPoint_2D(*pt)
            poly2 = ogr.Geometry(ogr.wkbPolygon)
            poly2.AddGeometry(ring2)
            multi.AddGeometry(poly2)

            feature.SetGeometry(multi)
        else:
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for pt in [
                (minx, miny),
                (maxx, miny),
                (maxx, maxy),
                (minx, maxy),
                (minx, miny),
            ]:
                ring.AddPoint_2D(*pt)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            feature.SetGeometry(poly)

        layer.CreateFeature(feature)
    except Exception as e:
        logger.warning(f"Failed to export AOI shapefile: {e}")
    finally:
        feature = None
        ds = None
