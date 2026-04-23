import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def parse_bbox_input(bbox_string: str) -> list[float]:
    """
    Parses a string input (KML, GeoJSON, WKT, or 4 coordinates) 
    into a standardized [South, North, West, East] bounding box list.
    Args:
        bbox_string (str): Input string representing a bounding box. 
            Can be a file path to KML/GeoJSON/SHP, a WKT string, or raw coordinates.
    Returns:
        list[float]: Bounding box in the format [South, North, West, East].
    """
    import os
    import geopandas as gpd
    from shapely import wkt
    import logging

    logger = logging.getLogger(__name__)

    # Check if it's a geospatial file-type (KML, GeoJSON, SHP)
    if os.path.isfile(bbox_string):
        if bbox_string.lower().endswith('.kml'):
            import fiona
            # Enable KML driver for geopandas/fiona
            fiona.drvsupport.supported_drivers['KML'] = 'rw'
            fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
            
        logger.info(f"Extracting bounding box from file: {bbox_string}")
        gdf = gpd.read_file(bbox_string)
        # GeoPandas total_bounds returns [minx, miny, maxx, maxy] -> [West, South, East, North]
        w, s, e, n = gdf.total_bounds
        return [s, n, w, e]
        
    # Check if it's a WKT string
    if bbox_string.upper().startswith(('POLYGON', 'MULTIPOLYGON', 'BBOX')):
        logger.info("Extracting bounding box from WKT string...")
        geom = wkt.loads(bbox_string)
        w, s, e, n = geom.bounds
        return [s, n, w, e]
        
    # Assume it's a raw coordinate string
    logger.info("Parsing raw coordinates...")
    coords = [float(x) for x in bbox_string.replace(',', ' ').split()]
    if len(coords) != 4:
        raise ValueError("Bounding box must be a valid file, WKT, or 4 space/comma separated coordinates.")
    
    return coords


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
        logger.info("Please ensure your local directory contains valid OPERA GeoTIFF products.")
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
        parts = name.split('_')
        date_str = None
        tile_id = "UNKNOWN"
        
        for i, part in enumerate(parts):
            if re.match(r'\d{8}T\d{6}Z', part):
                date_str = part
                if i > 0:
                    tile_id = parts[i-1]
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
            suffix = parts[-1].replace('.tif', '')
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
        logger.error(f"Found {len(tif_files)} files in {local_dir}, but none matched expected OPERA filename patterns.")
        return pd.DataFrame()

    # Convert to DataFrame
    rows = []
    for key, data in granules.items():
        rows.append(data)

    df = pd.DataFrame(rows)
    df['Start Time'] = pd.to_datetime(df['Start Time'], format='%Y%m%dT%H%M%SZ', errors='coerce')
    
    logger.info(f"Constructed local metadata DataFrame with {len(df)} unique granules (from {files_processed_count} files).")
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