import logging
import math
import os
import re
import textwrap
import uuid
from importlib.resources import files
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pygmt
import rioxarray
from osgeo import gdal
from pygmt.params import Box
from pyproj import Geod
import rasterio

from .io import cleanup_temp_file

logger = logging.getLogger(__name__)


def expand_region(region: list, width_deg: float = 15, height_deg: float = 10) -> list:
    """
    Return a new region [xmin, xmax, ymin, ymax] of fixed size,
    centered on the centroid of the input region, with coordinates rounded
    to 0 decimal places.

    Args:
        region (list): Input region in the form [xmin, xmax, ymin, ymax].
        width_deg (float): Desired width in degrees.
        height_deg (float): Desired height in degrees.

    Returns:
        list: New region with fixed size, centered on the input region.
    """
    xmin, xmax, ymin, ymax = region
    center_lon = (xmin + xmax) / 2
    center_lat = (ymin + ymax) / 2

    half_width = width_deg / 2
    half_height = height_deg / 2

    expanded_region = [
        round(center_lon - half_width),
        round(center_lon + half_width),
        round(center_lat - half_height),
        round(center_lat + half_height),
    ]
    return expanded_region


def expand_region_to_aspect(region: list, target_aspect: float) -> list:
    """
    Expand the input region to match a target aspect ratio.

    Args:
        region (list): Input region in the form [xmin, xmax, ymin, ymax].
        target_aspect (float): Desired aspect ratio (width / height).

    Returns:
        list: New region with adjusted aspect ratio.
    """
    xmin, xmax, ymin, ymax = map(float, region)
    width = xmax - xmin
    height = ymax - ymin
    current_aspect = width / height

    if current_aspect > target_aspect:
        # Too wide → expand height
        new_height = width / target_aspect
        pad = (new_height - height) / 2
        ymin -= pad
        ymax += pad
    else:
        # Too tall → expand width
        new_width = height * target_aspect
        pad = (new_width - width) / 2
        xmin -= pad
        xmax += pad

    return [xmin, xmax, ymin, ymax]


def make_map(
    maps_dir: Path,
    mosaic_path: Path,
    short_name: str,
    layer: str,
    date: str,
    bbox: list,
    zoom_bbox: list = None,
    is_difference: bool = False,
    utm_suffix: str = ""
) -> Path:
    """
    Create a map using PyGMT from the provided mosaic path.

    Args:
        maps_dir (Path): Directory where the map will be saved.
        mosaic_path (Path): Path to the mosaic file.
        short_name (str): Short name of the product.
        layer (str): Layer name to be used in the map.
        date (str): Date/PassID string.
        bbox (list): Bounding box in the form [South, North, West, East].
        zoom_bbox (list, optional): Bounding box for the zoom-in inset map, in the form [South, North, West, East].
        is_difference (bool, optional): Flag to indicate if the mosaic is a difference product. Defaults to False.
        utm_suffix (str, optional): Suffix string corresponding to the UTM Zone logic.

    Returns:
        map_name (Path): Path to the saved map image.

    Raises:
        ImportError: If required libraries are not installed.
    """
    # Helper to prettify pass IDs (YYYYMMDDtHHMM -> YYYY-MM-DD HH:MM)
    def format_pass_id(pid):
        # If it matches YYYYMMDDtHHMM
        if re.match(r"\d{8}t\d{4}", pid):
             return f"{pid[:4]}-{pid[4:6]}-{pid[6:8]} {pid[9:11]}:{pid[11:13]}"
        # If it matches YYYY-MM-DD
        if re.match(r"\d{4}-\d{2}-\d{2}", pid):
            return pid
        return pid

    # Determine date string for filename
    if is_difference:
        # Update regex to support standard dates (YYYY-MM-DD) AND pass IDs (YYYYMMDDtHHMM)
        # Groups: 1=Later, 2=Earlier
        match = re.search(
            r"((?:\d{4}-\d{2}-\d{2})|(?:\d{8}t\d{4}))_((?:\d{4}-\d{2}-\d{2})|(?:\d{8}t\d{4}))_(\_\d+N|\_\d+S|\_EPSG\d+|\_Hash\d+)?(?:log-)?diff",
            str(mosaic_path),
        )
        if match:
            date_later = match.group(1)
            date_earlier = match.group(2)
            date_str = f"{date_later}_{date_earlier}_diff"
        else:
            date_str = date
    else:
        date_str = date

    # Create a unique temporary path for the WGS84 reprojected file
    unique_id = uuid.uuid4().hex
    mosaic_wgs84 = Path(str(mosaic_path).replace(".tif", f"_WGS84_TMP_{unique_id}.tif"))

    try:
        # Reproject to WGS84 (into the temp file)
        gdal.Warp(
            str(mosaic_wgs84),
            str(mosaic_path),
            dstSRS="EPSG:4326",
            resampleAlg="near",
            creationOptions=["COMPRESS=DEFLATE"],
        )

        # Load mosaic from the temporary file
        grd = rioxarray.open_rasterio(mosaic_wgs84).squeeze()

        # Handle Nodata
        try:
            nodata_value = grd.rio.nodata
        except AttributeError:
            nodata_value = 255
            
        if nodata_value is not None:
            # Mask out nodata
            grd = grd.where(grd != nodata_value)

        # Define region
        region = [bbox[2], bbox[3], bbox[0], bbox[1]]  # [xmin, xmax, ymin, ymax]

        # Define target aspect ratio
        target_aspect = 60 / 100
        region_padded = expand_region_to_aspect(region, target_aspect)

        # Define projection
        center_lon = (region_padded[0] + region_padded[1]) / 2
        projection_width_cm = 15
        projection = f"M{center_lon}/{projection_width_cm}c"

        # Create PyGMT figure
        fig = pygmt.Figure()
        pygmt.config(FONT_ANNOT="6p")

        # Base coast layer
        fig.coast(
            region=region_padded,
            projection=projection,
            borders="2/thin",
            shorelines="thin",
            land="grey",
            water="lightblue",
        )

        palette_dir = files('disasters.assets.palettes')

        # 'landslide' mode (RTC)
        if is_difference:
            is_log_diff = "_log-diff" in str(mosaic_path)
            
            # --- CASE A: CONTINUOUS SCALE (RTC / Landslide) ---
            if is_log_diff:
                data_values = grd.values[~np.isnan(grd.values)]
                if len(data_values) == 0:
                    logger.warning(f"No valid data in {mosaic_path}")
                    cleanup_temp_file(mosaic_wgs84)
                    return None

                p2, p98 = np.percentile(data_values, [2, 98])
                symmetric_limit = max(abs(p2), abs(p98))
                if symmetric_limit == 0: symmetric_limit = 1 
                
                p_min = -symmetric_limit
                p_max = symmetric_limit
                inc = (p_max - p_min) / 1000.0

                cpt_name = f"difference_cpt_{unique_id}" # Unique Name
                pygmt.makecpt(
                    cmap="vik", series=[p_min, p_max, inc], output=cpt_name, continuous=True
                )
                
                fig.grdimage(
                    grid=grd, region=region_padded, projection=projection,
                    cmap=cpt_name, frame=["WSne", "xaf", "yaf"], nan_transparent=True
                )
                fig.colorbar(cmap=cpt_name, frame=["x+lNormalized backscatter difference (dB)"])

            # 'flood' mode (DSWx)
            else:
                # Check data range to determine if this is Binary Gain or Full Categorical
                valid_vals = grd.values[~np.isnan(grd.values)]
                max_val = valid_vals.max() if valid_vals.size > 0 else 0

                # --- Sub-Case B1: Binary Positive Change (Max value is 1) ---
                if max_val <= 1:
                    cpt_path = maps_dir / f"binary_gain_{unique_id}.cpt"
                    
                    # Create Simple Blue/White CPT
                    with open(cpt_path, "w") as f:
                        # 0 -> White (Fully Transparent 100)
                        f.write("0 255/255/255@100 1 255/255/255@100\n")
                        # 1 -> Blue (Opaque 0)
                        f.write("1 0/0/200@0 2 0/0/200@0\n")
                        # Background/NaN
                        f.write("B 255/255/255@100\nF 255/255/255@100\nN 255/255/255@100\n")

                    fig.grdimage(
                        grid=grd, region=region_padded, projection=projection,
                        cmap=str(cpt_path), frame=["WSne", "xaf", "yaf"], nan_transparent=True
                    )

                    # Simple Legend (Placed Inside Top Left to prevent layout squish)
                    legend_path = maps_dir / f"binary_legend_{unique_id}.txt"
                    with open(legend_path, "w") as f:
                        f.write("H 10p,Helvetica-Bold Water Change\n")
                        f.write("D 0.2c 1p\n") 
                        f.write("S 0.3c s 0.4c 0/0/200 0.25p 0.8c Water Gain\n")
                    
                    fig.legend(spec=str(legend_path), position="jTL+o0.2c/0.2c+w4.5c", box="+gwhite+p1p")

                # --- Sub-Case B2: Full Categorical (Max value > 1) ---
                else:
                    cpt_path = maps_dir / f"categorical_diff_{unique_id}.cpt"
                    
                    # Define color map for full 0-15 classes
                    color_map = {
                        # No Change (Black / Transparent for 0)
                        0:  (255, 255, 255, 0),    5:  (0, 0, 0, 255),
                        10: (0, 0, 0, 255),        15: (0, 0, 0, 255),
                        
                        # Losses (Red/Orange)
                        1:  (200, 0, 0, 255),      2:  (255, 127, 80, 255),
                        3:  (255, 165, 0, 255),    9:  (255, 200, 100, 255),
                        13: (255, 200, 100, 255),

                        # Gains (Blues)
                        4:  (0, 0, 200, 255),      8:  (100, 149, 237, 255),
                        12: (60, 179, 113, 255),   6:  (30, 144, 255, 255),
                        7:  (30, 144, 255, 255)
                    }

                    # Build valid CPT
                    with open(cpt_path, "w") as f:
                        for i in range(16):
                            if i in color_map:
                                r, g, b, a = color_map[i]
                                transparency = int(100 * (1 - (a / 255.0)))
                                f.write(f"{i} {r}/{g}/{b}@{transparency} {i+1} {r}/{g}/{b}@{transparency}\n")
                            else:
                                f.write(f"{i} 255/255/255@100 {i+1} 255/255/255@100\n")
                        f.write("B 255/255/255@100\nF 255/255/255@100\nN 255/255/255@100\n")
                    
                    fig.grdimage(
                        grid=grd, region=region_padded, projection=projection,
                        cmap=str(cpt_path), frame=["WSne", "xaf", "yaf"], nan_transparent=True
                    )

                    # Full Legend (Placed Inside Top Left to prevent layout squish)
                    legend_path = maps_dir / f"categorical_legend_{unique_id}.txt"
                    with open(legend_path, "w") as f:
                        f.write("H 10p,Helvetica-Bold Water Change Classes\n")
                        f.write("D 0.2c 1p\n")
                        f.write("S 0.3c s 0.4c 0/0/200 0.25p 0.8c Water Gain (Inundation)\n")
                        f.write("S 0.3c s 0.4c 30/144/255 0.25p 0.8c Water Gain (Partial)\n")
                        f.write("S 0.3c s 0.4c 200/0/0 0.25p 0.8c Water Loss (Drying)\n")
                        f.write("S 0.3c s 0.4c 255/127/80 0.25p 0.8c Water Loss (Partial)\n")
                        f.write("S 0.3c s 0.4c 0/0/0 0.25p 0.8c Stable Water\n")
                        
                    fig.legend(spec=str(legend_path), position="jTL+o0.2c/0.2c+w6.5c", box="+gwhite+p1p")

        # Add grid image (based on product/layer)
        elif not is_difference and layer in ["WTR", "BWTR", "CONF", "VEG-DIST-STATUS"]:
            
            # 1. Extract embedded colormap dynamically from the GeoTIFF
            cmap_dict = None
            with rasterio.open(mosaic_path) as src:
                try:
                    cmap_dict = src.colormap(1)
                except ValueError:
                    logger.warning(f"No embedded colormap found in {mosaic_path}")

            # 2. Build the CPT on-the-fly directly from the GeoTIFF
            color_palette = str(maps_dir / f"dynamic_cpt_{unique_id}.cpt")
            with open(color_palette, "w") as f:
                if cmap_dict:
                    for val in range(256):
                        if val in cmap_dict:
                            r, g, b, a = cmap_dict[val]
                            transparency = int(100 * (1 - (a / 255.0)))
                            # Force Nodatas or Alpha-0s to be fully transparent
                            if val == 255 or a == 0:
                                f.write(f"{val} {r}/{g}/{b}@100 {val+1} {r}/{g}/{b}@100\n")
                            else:
                                f.write(f"{val} {r}/{g}/{b}@{transparency} {val+1} {r}/{g}/{b}@{transparency}\n")
                        else:
                            f.write(f"{val} 0/0/0@100 {val+1} 0/0/0@100\n")
                else:
                    # Minimal failsafe
                    f.write("0 0/0/0@100 256 0/0/0@100\n")
                f.write("B 255/255/255@100\nF 255/255/255@100\nN 255/255/255@100\n")

            # 3. Draw the Map (interpolation="n" stops the fuzzy "rainbow" edge colors)
            fig.grdimage(
                grid=grd,
                region=region_padded,
                projection=projection,
                cmap=color_palette,
                frame=["WSne", "xaf", "yaf"],
                nan_transparent=True,
                interpolation="n"
            )

            # 4. Helper function to extract exact RGB string for the legend
            def get_rgb(v):
                if cmap_dict and v in cmap_dict:
                    return f"{cmap_dict[v][0]}/{cmap_dict[v][1]}/{cmap_dict[v][2]}"
                return "255/255/255"

            # 5. Build the corresponding Custom Legend in Upper Right (jTR)
            legend_path = maps_dir / f"custom_legend_{unique_id}.txt"
            
            if layer == "WTR":
                with open(legend_path, "w") as f:
                    f.write(f"H 10p,Helvetica-Bold {short_name.split('_')[2]} Water\n")
                    f.write("D 0.2c 1p\n")
                    if "HLS" in short_name:
                        f.write(f"S 0.3c s 0.4c {get_rgb(1)} 0.25p 0.8c 1: Open Water\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(2)} 0.25p 0.8c 2: Partial Surface Water\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(252)} 0.25p 0.8c 252: Snow/Ice Mask\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(253)} 0.25p 0.8c 253: Cloud/Cloud Shadow\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(254)} 0.25p 0.8c 254: Ocean Masked\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(0)} 0.25p 0.8c 0: Not Water\n")
                    else: # S1
                        f.write(f"S 0.3c s 0.4c {get_rgb(1)} 0.25p 0.8c 1: Open Water\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(3)} 0.25p 0.8c 3: Inundated Vegetation\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(250)} 0.25p 0.8c 250: HAND Masked\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(251)} 0.25p 0.8c 251: Layover/Shadow Masked\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(0)} 0.25p 0.8c 0: Not Water\n")
                fig.legend(spec=str(legend_path), position="jTL+o0.2c/0.2c+w6.0c", box="+gwhite+p1p")

            elif layer == "BWTR":
                with open(legend_path, "w") as f:
                    f.write(f"H 10p,Helvetica-Bold {short_name.split('_')[2]} Binary Water\n")
                    f.write("D 0.2c 1p\n")
                    if "HLS" in short_name:
                        f.write(f"S 0.3c s 0.4c {get_rgb(1)} 0.25p 0.8c 1: Water\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(252)} 0.25p 0.8c 252: Snow/Ice Mask\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(253)} 0.25p 0.8c 253: Cloud/Cloud Shadow\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(254)} 0.25p 0.8c 254: Ocean Masked\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(0)} 0.25p 0.8c 0: Not Water\n")
                    else: # S1
                        f.write(f"S 0.3c s 0.4c {get_rgb(1)} 0.25p 0.8c 1: Water\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(250)} 0.25p 0.8c 250: HAND Masked\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(251)} 0.25p 0.8c 251: Layover/Shadow Masked\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(0)} 0.25p 0.8c 0: Not Water\n")
                fig.legend(spec=str(legend_path), position="jTL+o0.2c/0.2c+w6.0c", box="+gwhite+p1p")

            elif layer == "CONF":
                with open(legend_path, "w") as f:
                    f.write("H 10p,Helvetica-Bold Confidence Classes\n")
                    f.write("D 0.2c 1p\n")
                    if "HLS" in short_name:
                        f.write(f"S 0.3c s 0.4c {get_rgb(0)} 0.25p 0.8c 0: Not Water\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(1)} 0.25p 0.8c 1: High Confidence\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(2)} 0.25p 0.8c 2: Moderate Confidence\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(3)} 0.25p 0.8c 3: Partial (Conservative)\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(4)} 0.25p 0.8c 4: Partial (Aggressive)\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(10)} 0.25p 0.8c 10: Not Water (Cloud)\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(11)} 0.25p 0.8c 11: High Conf (Cloud)\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(12)} 0.25p 0.8c 12: Mod Conf (Cloud)\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(13)} 0.25p 0.8c 13: Partial Cons (Cloud)\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(14)} 0.25p 0.8c 14: Partial Agg (Cloud)\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(20)} 0.25p 0.8c 20-24: Snow/Ice\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(254)} 0.25p 0.8c 254: Ocean Mask\n")
                    else: # S1
                        f.write(f"S 0.3c s 0.4c {get_rgb(0)} 0.25p 0.8c 0: Not Water\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(1)} 0.25p 0.8c 1: High Confidence\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(2)} 0.25p 0.8c 2: Moderate Confidence\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(5)} 0.25p 0.8c 5: Inundated Vegetation\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(6)} 0.25p 0.8c 6, 7: Low Backscatter\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(30)} 0.25p 0.8c 30: Not Water (Wetland)\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(31)} 0.25p 0.8c 31: High Conf (Wetland)\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(32)} 0.25p 0.8c 32: Mod Conf (Wetland)\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(35)} 0.25p 0.8c 35: Inundated Veg (Wetland)\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(36)} 0.25p 0.8c 36, 37: Low Backscr (Wetland)\n")
                        f.write(f"S 0.3c s 0.4c {get_rgb(250)} 0.25p 0.8c 250, 251: Topo/Layover\n")
                fig.legend(spec=str(legend_path), position="jTL+o0.2c/0.2c+w6.5c", box="+gwhite+p1p")

            elif layer == "VEG-DIST-STATUS":
                with open(legend_path, "w") as f:
                    f.write("H 10p,Helvetica-Bold Vegetation Disturbance Status\n")
                    f.write("D 0.2c 1p\n")
                    f.write(f"S 0.3c s 0.4c {get_rgb(0)} 0.25p 0.8c 0: No disturbance\n")
                    f.write(f"S 0.3c s 0.4c {get_rgb(1)} 0.25p 0.8c 1: First <50%\n")
                    f.write(f"S 0.3c s 0.4c {get_rgb(2)} 0.25p 0.8c 2: Provisional <50%\n")
                    f.write(f"S 0.3c s 0.4c {get_rgb(3)} 0.25p 0.8c 3: Confirmed <50%\n")
                    f.write(f"S 0.3c s 0.4c {get_rgb(4)} 0.25p 0.8c 4: First >=50%\n")
                    f.write(f"S 0.3c s 0.4c {get_rgb(5)} 0.25p 0.8c 5: Provisional >=50%\n")
                    f.write(f"S 0.3c s 0.4c {get_rgb(6)} 0.25p 0.8c 6: Confirmed >=50%\n")
                    f.write(f"S 0.3c s 0.4c {get_rgb(7)} 0.25p 0.8c 7: Confirmed <50%, finished\n")
                    f.write(f"S 0.3c s 0.4c {get_rgb(8)} 0.25p 0.8c 8: Confirmed >=50%, finished\n")
                fig.legend(spec=str(legend_path), position="jTL+o0.2c/0.2c+w6.5c", box="+gwhite+p1p")

        elif layer == "VEG-ANOM-MAX":
            color_palette = str(palette_dir / "VEG-ANOM-MAX.cpt")
            fig.grdimage(
                grid=grd,
                region=region_padded,
                projection=projection,
                cmap=color_palette,
                frame=["WSne", "xaf", "yaf"],
                nan_transparent=True,
            )
            fig.colorbar(
                cmap=color_palette,
                frame="xaf+lVEG-ANOM-MAX(%)",
            )

        elif short_name.startswith("OPERA_L2_RTC"):

            data_values = grd.values[~np.isnan(grd.values)]

            # Calculate the 2nd and 98th percentiles
            p2, p98 = np.percentile(data_values, [2, 98])

            # Ensure min is less than max
            if p2 >= p98:
                p2 -= 0.01
                p98 += 0.01

            # Calculate increment for 1000 steps
            inc = (p98 - p2) / 1000.0

            cpt_name = f"rtc_grayscale_{unique_id}"

            pygmt.makecpt(
                cmap="gray", series=[p2, p98, inc], output=cpt_name, continuous=True
            )

            fig.grdimage(
                grid=grd,
                region=region_padded,
                projection=projection,
                cmap=cpt_name,
                frame=["WSne", "xaf", "yaf"],
                nan_transparent=True,
            )

            fig.colorbar(
                cmap=cpt_name, frame=["x+lNormalized backscatter (@~g@~@-0@-)"]
            )

        # Add scalebar and compass rose
        xmin, xmax, ymin, ymax = region_padded
        center_lat = (ymin + ymax) / 2
        geod = Geod(ellps="WGS84")
        _, _, distance_m = geod.inv(xmin, center_lat, xmax, center_lat)

        # Set scalebar to ~25% of region width
        raw_length_km = distance_m * 0.25 / 1000
        exponent = math.floor(math.log10(raw_length_km))
        base = 10**exponent

        for factor in [1, 2, 5, 10]:
            scalebar_length_km = base * factor
            if scalebar_length_km >= raw_length_km:
                break

        fig.basemap(
            map_scale=f"jBR+o1c/0.6c+c-7+w{scalebar_length_km:.0f}k+f+lkm+ar",
            box=Box(fill="white@30", pen="0.5p,gray30,solid", radius="3p"),
        )

        fig.basemap(
            rose="jTR+o0.6c/0.2c+w1.5c",
            box=Box(fill="white@30", pen="0.5p,gray30,solid", radius="3p"),
        )

        bounds = grd.rio.bounds()
        region = [bounds[0], bounds[2], bounds[1], bounds[3]]
        region_expanded_main = expand_region(region, width_deg=25, height_deg=15)

        # Add inset map (regional context)
        with fig.inset(
            position="jBL+o0.2c/0.2c",
            box="+pblack",
            region=region_expanded_main,
            projection="M5c",
        ):
            # Use a plotting method to create a figure inside the inset.
            fig.coast(
                land="gray",
                borders=[1, 2],
                shorelines="1/thin",
                water="white",
            )
            # Coordinates for rectangular outline of the main region
            xmin, xmax, ymin, ymax = region_padded
            rectangle = [
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
                [xmin, ymin],
            ]

            # Plot the rectangle on the inset
            fig.plot(
                x=[pt[0] for pt in rectangle],
                y=[pt[1] for pt in rectangle],
                pen="2p,red",
            )

        # Optional inset for the zoom-in bbox (bottom right, include a scalebar)
        if zoom_bbox:
            zoom_region = [zoom_bbox[2], zoom_bbox[3], zoom_bbox[0], zoom_bbox[1]]

            # Calculate scale bar length for the zoom inset
            xmin_z, xmax_z, ymin_z, ymax_z = zoom_region
            center_lat_z = (ymin_z + ymax_z) / 2

            _, _, distance_m_z = geod.inv(xmin_z, center_lat_z, xmax_z, center_lat_z)
            raw_length_km_z = distance_m_z * 0.25 / 1000  # 25% of inset width in km

            scalebar_length_km_z = 1  # Default fallback
            if raw_length_km_z > 0:
                exponent_z = math.floor(math.log10(raw_length_km_z))
                base_z = 10**exponent_z

                for factor in [1, 2, 5, 10]:
                    scalebar_length_z = base_z * factor
                    if scalebar_length_z >= raw_length_km_z:
                        scalebar_length_km_z = scalebar_length_z
                        break

            with fig.inset(
                position="jBR+o0.5c/1.5c",
                box="+p1p,magenta",
                region=zoom_region,
                projection="M5c",
            ):

                # Add coastline to inset
                fig.coast(
                    region=zoom_region,
                    projection="M5c",
                    borders="1/thin",
                    shorelines="thin",
                    land="grey",
                    water="lightblue",
                )

                # Re-plot the data for the inset map
                if not is_difference and layer in ["WTR", "BWTR", "CONF", "VEG-DIST-STATUS"]:
                    fig.grdimage(
                        grid=grd,
                        region=zoom_region,
                        projection="M5c",
                        cmap=color_palette,
                        nan_transparent=True,
                        interpolation="n"
                    )
                elif not is_difference and layer == "VEG-ANOM-MAX":
                    fig.grdimage(
                        grid=grd,
                        region=zoom_region,
                        projection="M5c",
                        cmap=color_palette,
                        nan_transparent=True,
                    )
                elif not is_difference and short_name.startswith("OPERA_L2_RTC"):
                    fig.grdimage(
                        grid=grd,
                        region=zoom_region,
                        projection="M5c",
                        cmap=cpt_name,
                        nan_transparent=True,
                    )
                elif is_difference and "gain" in str(mosaic_path):
                    fig.grdimage(
                        grid=grd,
                        region=zoom_region,
                        projection="M5c",
                        cmap=str(cpt_path),
                        nan_transparent=True,
                        interpolation="n"
                    )
                elif is_difference:
                    fig.grdimage(
                        grid=grd,
                        region=zoom_region,
                        projection="M5c",
                        cmap=cpt_name,
                        nan_transparent=True
                    )

                # Add scale bar to the inset map. Use Bottom-Left (jBL) inside the inset frame.
                fig.basemap(
                    map_scale=f"jBL+o-0.5c/-0.5c+c-7+w{scalebar_length_km_z:.0f}k+f+lkm+ar",
                    box=Box(fill="white@30", pen="0.5p,gray30,solid", radius="3p"),
                )

            # Plot a rectangle on the main map to show the zoom area
            fig.plot(
                x=[
                    zoom_region[0],
                    zoom_region[1],
                    zoom_region[1],
                    zoom_region[0],
                    zoom_region[0],
                ],
                y=[
                    zoom_region[2],
                    zoom_region[2],
                    zoom_region[3],
                    zoom_region[3],
                    zoom_region[2],
                ],
                pen="1p,magenta",
            )

        # Export map
        map_name = maps_dir / f"{short_name}_{layer}_{date_str}{utm_suffix}_map.png"
        fig.savefig(map_name, dpi=900)
        cleanup_temp_file(mosaic_wgs84)

        # Remove all temp CPTs and Legends after all maps are drawn
        for tmp_file in [
            maps_dir / f"binary_gain_{unique_id}.cpt",
            maps_dir / f"binary_legend_{unique_id}.txt",
            maps_dir / f"categorical_diff_{unique_id}.cpt",
            maps_dir / f"categorical_legend_{unique_id}.txt",
            maps_dir / f"dynamic_cpt_{unique_id}.cpt",
            maps_dir / f"custom_legend_{unique_id}.txt",
            f"difference_cpt_{unique_id}",
            f"rtc_grayscale_{unique_id}"
        ]:
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except:
                pass

        return map_name

    except Exception as e:
        cleanup_temp_file(mosaic_wgs84)
        logger.error(f"An error occurred during map generation for {mosaic_path}: {e}")
        raise


def make_layout(
    layout_dir: Path,
    map_name: Path,
    short_name: str,
    layer: str,
    date: str,
    layout_date: str,
    layout_title: str,
    reclassify_snow_ice: bool = False,
    utm_suffix: str = ""
) -> None:
    """
    Create a layout using matplotlib for the provided map.

    Args:
        layout_dir (Path): Directory where the layout will be saved.
        map_name (Path): Path to the map image.
        short_name (str): Short name of the product.
        layer (str): Layer name to be used in the layout.
        date (str): Date/PassID string.
        layout_date (str): Date threshold in the format YYYY-MM-DD.
        layout_title (str): Title for the layout.
        reclassify_snow_ice (bool, optional): Flag indicating if snow/ice reclassification was applied. Defaults to False.
        utm_suffix (str, optional): Suffix string corresponding to the UTM Zone logic.
    """
    # Helper to prettify dates
    def format_display_date(pid):
        # Handle difference format: "YYYYMMDDtHHMM, YYYYMMDDtHHMM"
        if ',' in pid:
             parts = pid.split(',')
             return f"{format_display_date(parts[0].strip())}, {format_display_date(parts[1].strip())}"
        # Handle single PassID YYYYMMDDtHHMM
        if re.match(r"\d{8}t\d{4}", pid):
             return f"{pid[:4]}-{pid[4:6]}-{pid[6:8]} {pid[9:11]}:{pid[11:13]}"
        return pid

    # Create blank figure
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_axis_off()

    # Set background
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Add main map ===
    map_img = mpimg.imread(map_name)
    ax.imshow(map_img, extent=[0, 60, 0, 100])  # Main map on left 60% of layout

    # Add OPERA/ARIA logos
    logo_dir = files('disasters.assets.logos')
    logo_opera = mpimg.imread(logo_dir / "OPERA_logo.png")
    logo_new = mpimg.imread(logo_dir / "ARIA_logo.png")

    # Create a new axes for logos in the bottom-right corner
    logo_ax = fig.add_axes([0.82, 0.12, 0.06, 0.08], anchor="SE", zorder=10)
    logo_ax.imshow(logo_opera)
    logo_ax.axis("off")

    logo_ax2 = fig.add_axes([0.89, 0.12, 0.06, 0.08], anchor="SE", zorder=10)
    logo_ax2.imshow(logo_new)
    logo_ax2.axis("off")

    # Layout control
    x_pos = 0.65  # Just right of the map
    line_spacing = 0.04  # vertical spacing between blocks

    # Define wrap width (in characters)
    wrap_width = 50

    # Map text elements
    if short_name == "OPERA_L3_DSWX-S1_V1" and layer != "CONF":
        subtitle = "OPERA Dynamic Surface Water eXtent from Sentinel-1 (DSWx-S1)"
        map_information = (
            f"The ARIA/OPERA water extent map is derived from an OPERA DSWx-S1 mosaicked "
            f"product from Copernicus Sentinel-1 data."
            f"This map depicts regions of full surface water and inundated surface water. "
        )
        data_source = "Copernicus Sentinel-1"

    elif short_name == "OPERA_L3_DSWX-HLS_V1" and layer != "CONF":
        subtitle = "OPERA Dynamic Surface Water eXtent from HLS (DSWx-HLS)"
        if reclassify_snow_ice == True:
            map_information = textwrap.dedent(
                f"""\
                The ARIA/OPERA water extent map is derived from an OPERA DSWx-HLS mosaicked 
                product from Harmonized Landsat and Sentinel-2 data.

                Note: Cloud/cloud shadow and snow/ice layers are derived from HLS Fmask 
                quality assurance (QA) data, which sometimes misclassifies sediment-rich water as snow/ice. 
                Snow/ice pixels were reclassified to open water to capture the full inundated extent.
            """
            )
        else:
            map_information = (
                f"The ARIA/OPERA water extent map is derived from an OPERA DSWx-HLS mosaicked "
                f"product from Harmonized Landsat and Sentinel-2 data."
                f"This map depicts regions of full surface water and inundated surface water. "
            )
        data_source = "Copernicus Harmonized Landsat and Sentinel-2"

    elif layer == "CONF":
        product_label = "DSWx-HLS" if "HLS" in short_name else "DSWx-S1"
        subtitle = f"OPERA Dynamic Surface Water eXtent Confidence ({product_label})"
        map_information = (
            f"This map depicts the confidence layer associated with the ARIA/OPERA water extent map. "
            f"It represents the quality, probability, or classification confidence of the surface water."
        )
        data_source = "Copernicus Harmonized Landsat and Sentinel-2" if "HLS" in short_name else "Copernicus Sentinel-1"

    elif short_name == "OPERA_L3_DIST-ALERT-S1_V1":
        subtitle = "OPERA Surface Disturbance Alert from Sentinel-1 (DIST-ALERT-S1)"
        map_information = (
            f"The ARIA/OPERA surface disturbance alert map is derived from an OPERA DIST-ALERT-S1 mosaicked "
            f"product from Copernicus Sentinel-1 data."
            f"This map depicts regions of surface disturbance since "
            + layout_date
            + "."
        )
        data_source = "Copernicus Sentinel-1"

    elif short_name == "OPERA_L3_DIST-ALERT-HLS_V1":
        subtitle = "OPERA Surface Disturbance Alert from Harmonized Landsat and Sentinel-2 (DIST-ALERT-HLS)"
        map_information = (
            f"The ARIA/OPERA surface disturbance alert map is derived from an OPERA DIST-ALERT-HLS mosaicked "
            f"product from Harmonized Landsat and Sentinel-2 data. "
            f"This map depicts regions of vegetation disturbance since "
            + layout_date
            + "."
        )
        data_source = "Copernicus Harmonized Landsat and Sentinel-2"

    elif short_name == "OPERA_L2_RTC-S1_V1":
        subtitle = "OPERA Radiometrically Terrain Corrected Backscatter from Sentinel-1 (RTC-S1)"
        map_information = (
            f"The ARIA/OPERA backscatter map is derived from an OPERA RTC-S1 mosaicked product "
            f"from Copernicus Sentinel-1 data."
            f"This map depicts the radar backscatter intensity, which can be used to identify "
            f"surface features and changes."
        )
        data_source = "Copernicus Sentinel-1"

    # Format date
    acquisitions = format_display_date(str(date))

    data_sources = textwrap.dedent(
        f"""\
        Product: {short_name}

        Layer: {layer}

        Data Source: {data_source}

        Resolution: 30 meters
    """
    )

    data_availability = textwrap.dedent(
        f"""\
        This product is available at: https://aria-share.jpl.nasa.gov/

        Visit the OPERA website: https://www.jpl.nasa.gov/go/opera/
    """
    )

    disclaimer = (
        "The results posted here are preliminary and unvalidated, "
        "intended to aid field response and provide a first look at the disaster-affected region."
    )

    # Wrapping text
    title_wrp = textwrap.fill(layout_title, width=40)
    subtitle_wrp = textwrap.fill(subtitle, width=wrap_width)
    acquisitions_wrp = textwrap.fill(acquisitions, width=wrap_width)
    map_information_wrp = textwrap.fill(map_information, width=wrap_width)
    data_sources_wrp = textwrap.fill(data_sources, width=wrap_width)
    data_availability_wrp = textwrap.fill(data_availability, width=wrap_width)
    disclaimer_wrp = textwrap.fill(disclaimer, width=wrap_width)

    # Starting y-position (top of the figure)
    y_start = 0.98

    # Define title
    ax.text(
        x_pos,
        y_start,
        title_wrp,
        fontsize=14,
        weight="bold",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    # Define subtitle
    ax.text(
        x_pos,
        y_start - line_spacing * 1,
        subtitle_wrp,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    # Acquisition heading
    ax.text(
        x_pos,
        y_start - line_spacing * 3.5,
        "Data Acquisitions:",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    # Acquisition dates
    ax.text(
        x_pos,
        y_start - line_spacing * 4,
        acquisitions,
        fontsize=8,
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    # Map information heading
    ax.text(
        x_pos,
        y_start - line_spacing * 6,
        "Map Information:",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    # Map information text
    ax.text(
        x_pos,
        y_start - line_spacing * 6.5,
        map_information_wrp,
        fontsize=8,
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    # Data sources heading
    ax.text(
        x_pos,
        y_start - line_spacing * (10 + 1.5),
        "Data Sources:",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    # Data sources text
    ax.text(
        x_pos,
        y_start - line_spacing * (10.5 + 1.5),
        data_sources,
        fontsize=8,
        ha="left",
        va="top",
        transform=ax.transAxes,
        linespacing=1,
        wrap=True,
    )
    # Data availability heading
    ax.text(
        x_pos,
        y_start - line_spacing * (15 + 1.5),
        "Product Availability:",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    # Data availability text
    ax.text(
        x_pos,
        y_start - line_spacing * (15.5 + 1.5),
        data_availability,
        fontsize=8,
        ha="left",
        va="top",
        linespacing=1,
        transform=ax.transAxes,
        wrap=True,
    )

    # Disclaimer heading
    ax.text(
        x_pos,
        y_start - line_spacing * (18 + 1.5),
        "Disclaimer:",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    # Disclaimer
    ax.text(
        x_pos,
        y_start - line_spacing * (18.5 + 1.5),
        disclaimer_wrp,
        fontsize=8,
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    layout_name = layout_dir / f"{short_name}_{layer}_{date}{utm_suffix}_layout.pdf"
    plt.savefig(layout_name, format="pdf", bbox_inches="tight", pad_inches=0.2, dpi=400)
    plt.close(fig)
    return