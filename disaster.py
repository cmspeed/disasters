import argparse
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import next_pass
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import xarray as xr
from osgeo import gdal


def parse_arguments():
    """
    Parse command line arguments for the disaster analysis workflow.
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run disaster analysis workflow.")

    valid_short_names = [
        "OPERA_L3_DSWX-HLS_V1",
        "OPERA_L3_DSWX-S1_V1",
        "OPERA_L3_DIST-ALERT-HLS_V1",
        "OPERA_L3_DIST-ANN-HLS_V1",
        "OPERA_L2_RTC-S1_V1",
        "OPERA_L2_CSLC-S1_V1",
        "OPERA_L3_DISP-S1_V1",
    ]

    valid_layer_names = ["WTR", "BWTR", "VEG-ANOM-MAX", "VEG-DIST-STATUS"]

    valid_modes = ["flood", "fire", "landslide", "earthquake"]

    valid_functions = ["opera_search", "both"]

    parser.add_argument(
        "-b",
        "--bbox",
        nargs=4,
        type=float,
        metavar=("S", "N", "W", "E"),
        required=True,
        help="Bounding box in the form: South North West East",
    )

    parser.add_argument(
        "-zb",
        "--zoom_bbox",
        nargs=4,
        type=float,
        metavar=("S", "N", "W", "E"),
        required=False,
        default=None,
        help="Optional bounding box for the zoom-in inset map.",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        required=True,
        help="Path to the directory where results and metadata will be saved.",
    )

    parser.add_argument(
        "-sn",
        "--short_name",
        type=str,
        choices=valid_short_names,
        help="Short name to filter the DataFrame (must be one of the known OPERA products)",
    )

    parser.add_argument(
        "-l",
        "--layer_name",
        type=str,
        choices=valid_layer_names,
        help="Layer name to extract from metadata (e.g., 'WTR', 'BWTR', 'VEG-ANOM-MAX')",
    )

    parser.add_argument(
        "-d",
        "--date",
        type=str,
        required=False,
        help="Specifies the end date (YYYY-MM-DD) for the OPERA product search. "
        "The script will find the 'N' most recent products available on or before this date (where 'N' is set by --number-of-dates argument). "
        "Defaults to 'today'.",
    )

    parser.add_argument(
        "-n",
        "--number_of_dates",
        type=int,
        default=5,
        help="Number of most recent dates to consider for OPERA products",
    )

    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="flood",
        choices=valid_modes,
        help="Mode of operation: flood, fire, landslide, earthquake. Default is 'flood'.",
    )

    parser.add_argument(
        "-f",
        "--functionality",
        type=str,
        default="opera_search",
        choices=valid_functions,
        help="Functionality to run: 'opera_search' or 'both'. Default is 'opera_search'.",
    )

    parser.add_argument(
        "-lt",
        "--layout_title",
        type=str,
        required=True,
        default="Layout Title",
        help="Title for the PDF layout(s). Must be enclosed in double quotes and is required.",
    )

    parser.add_argument(
        "-fd",
        "--filter_date",
        type=str,
        required=False,
        default=None,
        help="Date string (YYYY-MM-DD) to filter by date in the date filtering step in 'fire' mode.",
    )

    parser.add_argument(
        "-rc",
        "--reclassify_snow_ice",
        action="store_true",
        required=False,
        help="Flag to reclassify false snow/ice positives as water in DSWx-HLS products ONLY. Default is False.",
    )

    return parser.parse_args()


def authenticate():
    """
    Authenticate with Earthdata and ASF for data access.
    Returns:
        tuple: (username, password) for Earthdata and ASF access.
    """
    import netrc

    import boto3
    import earthaccess
    import rasterio
    from rasterio.session import AWSSession

    temp_creds_req = earthaccess.get_s3_credentials(daac="PODAAC")
    session = boto3.Session(
        aws_access_key_id=temp_creds_req["accessKeyId"],
        aws_secret_access_key=temp_creds_req["secretAccessKey"],
        aws_session_token=temp_creds_req["sessionToken"],
        region_name="us-west-2",
    )
    rio_env = rasterio.Env(
        AWSSession(session),
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS="TIF, TIFF",
        GDAL_HTTP_COOKIEFILE=os.path.expanduser("~/cookies.txt"),
        GDAL_HTTP_COOKIEJAR=os.path.expanduser("~/cookies.txt"),
    )
    rio_env.__enter__()
    # Parse credentials from the netrc file for ASF access
    netrc_file = Path.home() / ".netrc"
    auths = netrc.netrc(netrc_file)
    username, _, password = auths.authenticators("urs.earthdata.nasa.gov")
    return username, password


def make_output_dir(output_dir: Path):
    """
    Create the output directory if it does not exist.
    Args:
        output_dir (Path): Path to the output directory.
    Raises:
        Exception: If the directory cannot be created.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Created or reused output directory: {output_dir}")
    except Exception as e:
        print(f"[ERROR] Could not create output directory: {e}")
        raise
    return


def read_opera_metadata_csv(output_dir):
    """
    Read the OPERA products metadata CSV file and clean the 'Start Time' column.

    Args:
        output_dir (Path): Path to the directory containing the CSV file.
    Returns:
        pd.DataFrame: DataFrame with 'Start Time' as datetime64[ns].
    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    csv_path = Path(output_dir) / "opera_products_metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_path)

    # Define the two format strings (this is necessary as RTC has slightly different format)
    FORMAT_MICROSECONDS = "%Y-%m-%dT%H:%M:%S.%fZ"  # For non-RTC data
    FORMAT_SECONDS_ONLY = "%Y-%m-%dT%H:%M:%SZ"  # For RTC data

    # Parse the non-RTC format (RTC dates become NaT)
    df_temp1 = pd.to_datetime(
        df["Start Time"], format=FORMAT_MICROSECONDS, errors="coerce"
    )

    # Parse the RTC format (Non-RTC dates become NaT)
    df_temp2 = pd.to_datetime(
        df["Start Time"], format=FORMAT_SECONDS_ONLY, errors="coerce"
    )

    # Combine differently parsed datetimes into single column
    df["Start Time"] = df_temp1.combine_first(df_temp2)

    print(f"[INFO] Loaded {len(df)} rows from {csv_path}")

    return df


def compile_and_load_data(
    data_layer_links, mode, conf_layer_links=None, date_layer_links=None
):
    """
    Compile and load data from the provided layer links for mosaicking. This modified version
    loads all datasets regardless of their CRS, but filters to the most common Sentinel-1
    unit (S1A or S1C) if applicable.

    Args:
        data_layer_links (list): List of URLs corresponding to the OPERA data layers to mosaic.
        mode (str): Mode of operation, e.g., "flood", "fire", "landslide", "earthquake".
        conf_layer_links (list, optional): List of URLs for additional layers to filter false positives.
        date_layer_links (list, optional): List of URLs for date layers to filter by date.
    Returns:
        DS: List of rioxarray datasets loaded from the provided links (in granule order).
        conf_DS: List of rioxarray datasets for confidence layers (if applicable, in granule order).
        date_DS: List of rioxarray datasets for date layers (if applicable, in granule order).
    Raises:
        Exception: If there is an error loading any of the datasets.
    """
    import logging
    from collections import Counter

    from opera_utils.disp._remote import open_file

    logging.getLogger().setLevel(logging.ERROR)

    # Authenticate to get username and password
    username, password = authenticate()

    # Ensure only S1A or S1C are used (not both) for a single date
    satellite_counts = Counter()
    for link in data_layer_links:
        if "S1A" in link:
            satellite_counts["S1A"] += 1
        elif "S1C" in link:
            satellite_counts["S1C"] += 1

    if satellite_counts:
        # Get the satellite type with the highest count
        most_common_satellite, _ = satellite_counts.most_common(1)[0]
        print(
            f"[INFO] Most common satellite type: {most_common_satellite}, keeping only those links."
        )

        # Create a boolean mask to filter all lists consistently
        is_most_common = [most_common_satellite in link for link in data_layer_links]

        data_layer_links = [
            link for i, link in enumerate(data_layer_links) if is_most_common[i]
        ]

        # Filter auxiliary links consistently if they exist
        if conf_layer_links:
            conf_layer_links = [
                link for i, link in enumerate(conf_layer_links) if is_most_common[i]
            ]
        if date_layer_links:
            date_layer_links = [
                link for i, link in enumerate(date_layer_links) if is_most_common[i]
            ]

    # Helper to load datasets
    def load_datasets(links):
        datasets = []
        for link in links:
            try:
                datasets.append(rioxarray.open_rasterio(link, masked=False))
            except Exception as e:
                f = open_file(
                    link,
                    earthdata_username=username,
                    earthdata_password=password,
                )
                datasets.append(rioxarray.open_rasterio(f, masked=False))
        return datasets

    # Load the primary data layer (DS)
    DS = load_datasets(data_layer_links)

    # If conf_layer_links AND mode == 'flood' compile and load layers to use in filtering
    if conf_layer_links and mode == "flood":
        conf_DS = load_datasets(conf_layer_links)
        return DS, conf_DS

    # If conf_layer_links AND date_layer_links AND mode == 'fire' or mode == 'landslide' compile and load layers to use in filtering
    if (
        conf_layer_links
        and date_layer_links
        and (mode == "fire" or mode == "landslide")
    ):
        conf_DS = load_datasets(conf_layer_links)
        date_DS = load_datasets(date_layer_links)
        return DS, date_DS, conf_DS
    else:
        return DS


def reclassify_snow_ice_as_water(DS, conf_DS):
    """
    Reclassify false snow/ice positives (value 252) as water (value 1) based on the confidence layers. Only applicable for DSWx-HLS.

    Args:
        DS (list): List of rioxarray datasets (BWTR layers).
        conf_DS (list): List of rioxarray datasets (CONF layers).

    Returns:
        list: List of updated rioxarray datasets with 252 reclassified as 1.
        colormap: Colormap from the original datasets (if available).
    """
    import opera_mosaic

    if conf_DS is None:
        raise ValueError("conf_DS must not be None when reclassifying snow/ice.")

    if len(DS) != len(conf_DS):
        raise ValueError("DS and conf_DS must be the same length.")

    values_to_reclassify = [1, 3, 4, 21, 23, 24]

    try:
        colormap = opera_mosaic.get_image_colormap(DS[0])
        print(
            f"[INFO] Colormap successfully retrieved and will be used in reclassified output"
        )
    except Exception:
        print("[INFO] Unable to get colormap")
        colormap = None

    updated_list = []

    for da_data, da_conf in zip(DS, conf_DS):
        # Get the original data values
        data_values = da_data.values.copy()
        conf_values = da_conf.values

        # Identify locations where DS == 252 and conf layer indicates water
        condition = (data_values == 252) & np.isin(conf_values, values_to_reclassify)

        # Reclassify those pixels to 1 (Water)
        data_values[condition] = 1

        # Create updated DataArray
        updated = xr.DataArray(
            data_values, coords=da_data.coords, dims=da_data.dims, attrs=da_data.attrs
        )

        # Preserve spatial metadata
        if hasattr(da_data, "rio"):
            updated = (
                updated.rio.write_nodata(da_data.rio.nodata)
                .rio.write_crs(da_data.rio.crs)
                .rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
                .rio.write_transform(da_data.rio.transform())
            )

        updated_list.append(updated)

    return updated_list, colormap


def filter_by_date_and_confidence(
    DS,
    DS_dates,
    date_threshold,
    DS_conf=None,
    confidence_threshold=None,
    fill_value=None,
):
    """
    Filters each data xarray in `DS` based on:
      - date threshold from `DS_dates`
      - optional confidence threshold from `DS_conf`

    Pixels not meeting the criteria are set to `fill_value`.
    If `fill_value` is None, defaults to da_data.rio.nodata or NaN.

    Parameters
    ----------
    DS : list of xr.DataArray
        List of data granules (e.g., VEG-DIST-STATUS tiles).
    DS_dates : list of xr.DataArray
        List of corresponding date granules.
    date_threshold : int or datetime-like
        Pixels with dates >= this value are retained.
    DS_conf : list of xr.DataArray, optional
        List of confidence rasters corresponding to `DS`. Default is None.
    confidence_threshold : float or int, optional
        Pixels with confidence >= this value are retained.
    fill_value : number, optional
        Value to fill where condition is not met. If None, uses nodata or NaN.

    Returns
    -------
    filtered_list: list of xr.DataArray filtered data granules.
    colormap: Colormap from the original datasets (if available).
    """
    import opera_mosaic

    assert len(DS) == len(DS_dates), "DS and DS_dates must be same length"
    if DS_conf is not None:
        assert len(DS_conf) == len(DS), "DS_conf must match DS in length"

    try:
        colormap = opera_mosaic.get_image_colormap(DS[0])
        print(
            "[INFO] Colormap successfully retrieved and will be used in reclassified output"
        )
    except Exception:
        print("[INFO] Unable to get colormap")
        colormap = None

    filtered_list = []

    for i, (da_data, da_date) in enumerate(zip(DS, DS_dates)):
        print(f"[INFO] Filtering granule {i + 1}/{len(DS)}")

        # Date mask
        date_mask = da_date >= date_threshold

        # Optional confidence mask
        if DS_conf is not None and confidence_threshold is not None:
            conf_layer = DS_conf[i]
            print(f"[INFO] Confidence layer shape: {conf_layer.shape}")
            total_pixels = conf_layer.size

            # Construct confidence mask based on confidence_threshold
            conf_mask = conf_layer >= confidence_threshold

            retained_pixels = conf_mask.sum().item()
            print(
                f"[INFO] Confidence retained: {retained_pixels} / {total_pixels} ({retained_pixels / total_pixels:.2%})"
            )

            max_retained_conf = conf_layer.where(conf_mask).max().item()
            print(f"[INFO] Max confidence among retained pixels: {max_retained_conf}")

            combined_mask = date_mask & conf_mask
        else:
            combined_mask = date_mask

        # Determine fill value
        default_nodata = (
            da_data.rio.nodata
            if hasattr(da_data, "rio") and da_data.rio.nodata is not None
            else da_data.attrs.get("_FillValue", np.nan)
        )
        replacement = fill_value if fill_value is not None else default_nodata

        # Apply mask
        filtered = xr.where(combined_mask, da_data, replacement)

        # Preserve metadata
        filtered.attrs.update(da_data.attrs)

        if hasattr(da_data, "rio"):
            filtered = (
                filtered.rio.write_nodata(replacement)
                .rio.write_crs(da_data.rio.crs)
                .rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
                .rio.write_transform(da_data.rio.transform())
            )

        filtered_list.append(filtered)

    return filtered_list, colormap


def compute_date_threshold(date_str):
    """
    Compute the date threshold in days from a reference date (2020-12-31).
    Args:
        date_str (str): Date string in the format YYYY-MM-DD.
    Returns:
        date_threshold (int): Number of days from the reference date to the target date.
    """
    # Define the reference date and the target date
    reference_date = datetime(2020, 12, 31)
    target_date = datetime.strptime(date_str, "%Y-%m-%d")

    # Calculate the difference between the two dates
    delta = target_date - reference_date

    # Get the number of days from the timedelta object
    date_threshold = delta.days

    return date_threshold


def same_utm_zone(crs_a: rasterio.crs.CRS, crs_b: rasterio.crs.CRS) -> bool:
    """
    Returns True if both are projected UTM with the same EPSG (e.g., 32611 vs 32611).
    """
    try:
        if not crs_a or not crs_b:
            return False
        epsg_a = crs_a.to_epsg()
        epsg_b = crs_b.to_epsg()
        if epsg_a is None or epsg_b is None:
            return False
        # UTM EPSGs are typically 326xx (north) / 327xx (south)
        return epsg_a == epsg_b and (
            str(epsg_a).startswith("326") or str(epsg_a).startswith("327")
        )
    except Exception:
        return False


def compute_and_write_difference(
    earlier_path: Path,
    later_path: Path,
    out_path: Path,
    nodata_value: float | int | None = None,
    log: bool = False,
):
    """
    Create a difference raster for 'flood' or 'landslide' mode.
    Writes a uint8 categorical raster with an embedded color palette for DSWx products.
    """
    import rioxarray
    import xarray as xr
    import numpy as np
    import rasterio

    # Open rasters and apply mask for nodata handling
    da_later = rioxarray.open_rasterio(later_path, masked=True)
    da_early = rioxarray.open_rasterio(earlier_path, masked=True)

    # Determine the nodata value to use (nd) from inputs
    nd = nodata_value
    if nd is None:
        nd = da_later.rio.nodata
        if nd is None:
            nd = da_early.rio.nodata

    # Landslide mode
    if log:
        # Compute log difference for RTC-S1
        diff = 10 * np.log10(da_later / da_early)
        diff = diff.astype("float32")
        print("[INFO] Computed log difference for RTC-S1.")
        
        diff.attrs.clear()
        diff.attrs["DESCRIPTION"] = "Log Ratio Difference (Later / Earlier) for RTC-S1"

        # Handle Nodata
        input_nodata_mask = xr.where(da_later.isnull() | da_early.isnull(), True, False)
        result_nan_mask = np.isnan(diff)
        final_nodata_mask = input_nodata_mask | result_nan_mask

        diff = xr.where(final_nodata_mask, nd, diff)
        diff.rio.write_nodata(nd, encoded=True, inplace=True)
        diff.rio.write_crs(da_later.rio.crs, inplace=True)

        # Write Temp
        tmp_gtiff = out_path.with_suffix(".tmp.tif")
        diff.rio.to_raster(tmp_gtiff, compress="DEFLATE", tiled=True, dtype="float32")
        save_gtiff_as_cog(tmp_gtiff, out_path)
        try: 
            tmp_gtiff.unlink(missing_ok=True)
        except: 
            pass
        return

    # Compute categorical difference for DSWx products
    else:
        print("[INFO] Computing categorical transition codes for DSWx...")
        
        # FIX: Ensure nd is a valid integer (255) if input nodata is None or NaN
        # This prevents the 'cannot convert float NaN to integer' error
        if nd is None or np.isnan(nd):
            nd = 255.0
        
        nd_float = float(nd)

        # Detect either DSWx-HLS or DSWx-S1 from filename
        filename = str(out_path.name)
        is_hls = "HLS" in filename
        is_s1 = "S1" in filename and not is_hls

        # Define the Color Palette (R, G, B, Alpha)
        # Colors: Blues for Gains, Reds for Losses, Transparent for No Change
        full_colormap = {
            # --- NO CHANGE (Transparent) ---
            0: (255, 255, 255, 255), # 0 -> 0: Not Water -> Not Water (White)
            5: (0, 0, 0, 255), # 1 -> 1: Open Water -> Open Water (Black)
            10: (0, 0, 0, 255), # 2 -> 2: Partial Surface Water -> Partial Surface Water (Black)
            15: (0, 0, 0, 255), # 3 -> 3: Inundated Vegetation -> Inundated Vegetation (Black)

            # --- WATER LOSS (Reds/Oranges) ---
            # "Recession" (Wet -> Dry)
            1: (200, 0, 0, 255), # 1->0: Open Water -> Not Water (Deep Red)
            2: (255, 127, 80, 255), # 2->0: Partial Surface Water -> Not Water (Lightest Red/Coral)
            # 3: (255, 127, 80, 255), # 3->0: Inundated Vegetation -> Not Water (Lightest Red/Coral)
            9: (255, 200, 100, 255), # 1->2: Open Water -> Partial Surface Water (Light Red)
            #13: (255, 200, 100, 255), # 1->3: Open Water -> Inundated Vegetation (Light Red)

            # --- WATER GAIN (Blues) ---
            # "Inundation" (Dry -> Wet)
            4: (0, 0, 200, 255), # 0->1: Not Water -> Open Water (Deepest Blue)
            8: (100, 149, 237, 255), # 0->2: Not Water -> Partial Surface Water (Lightest Blue)
            # 12: (100, 149, 237, 255), # 0->3: Not Water -> Inundated Vegetation (Lightest Blue)
            6: (30, 144, 255, 255), # 2->1: Partial Surface Water -> Open Water (Light Blue)
            # 7: (30, 144, 255, 255), # 3->1: Inundated Vegetation -> Open Water (Light Blue)
        }

        full_names = {
            0: "No Change: Not Water",
            5: "No Change: Open Water",
            1: "Loss: Open Water to Not Water",
            4: "Gain: Not Water to Open Water",
            # HLS Specific
            10: "No Change: Partial Surface Water",
            2: "Loss: Partial Surface Water to Not Water",
            9: "Loss: Open Water to Partial Surface Water",
            8: "Gain: Not Water to Partial Surface Water",
            6: "Gain: Partial Surface Water to Open Water",
            # S1 Specific
            15: "No Change: Inundated Vegetation",
            3: "Loss: Inundated Vegetation to Not Water",
            13: "Loss: Open Water to Inundated Vegetation",
            12: "Gain: Not Water to Inundated Vegetation",
            7: "Gain: Inundated Vegetation to Open Water",
        }

        # Filter colormap and names based on product type
        active_colormap = {}
        active_names = {}
        
        # Always include universal classes (0, 1, 4, 5)
        universal_keys = [0, 1, 4, 5]

        if is_hls:
            # HLS: Include Universal + Partial Water (2, 6, 8, 9, 10)
            hls_keys = universal_keys + [2, 6, 8, 9, 10]
            for k in hls_keys:
                if k in full_colormap: active_colormap[k] = full_colormap[k]
                if k in full_names: active_names[k] = full_names[k]
        
        elif is_s1:
            # S1: Include Universal + Inundated Veg (3, 7, 12, 13, 15)
            s1_keys = universal_keys + [3, 7, 12, 13, 15]
            for k in s1_keys:
                if k in full_colormap: active_colormap[k] = full_colormap[k]
                if k in full_names: active_names[k] = full_names[k]
        else:
            # Fallback (include everything if unknown)
            active_colormap = full_colormap
            active_names = full_names

        VALID_CLASSES = [0, 1, 2, 3]
        MAX_CLASS_VALUE = 4

        # Keep data as floats to avoid 'NaN to integer' errors
        L = da_later.fillna(0)
        E = da_early.fillna(0)
        transition_code = L * MAX_CLASS_VALUE + E

        # Create mask for impossible combinations (Partial <-> Veg)
        impossible_mask = (transition_code == 11) | (transition_code == 14)
        
        # Mask invalid inputs (np.isin works with floats)
        valid_input_mask = (np.isin(L, VALID_CLASSES) & np.isin(E, VALID_CLASSES))
        
        # Additional user-specified mask
        masked_classes_list = [3, 7, 12]
        user_mask = np.isin(transition_code, masked_classes_list)

        # Keep if (Valid Input) AND (Not Impossible) AND (Not User Masked)
        keep_pixel_mask = valid_input_mask & ~impossible_mask & ~user_mask

        # Apply Logic: Mask invalid/impossible pixels to 'nd_float'
        final_data_float = xr.where(
            keep_pixel_mask,
            transition_code,
            nd_float
        )

        # Fill NaNs with nd_float 
        final_data_float = final_data_float.fillna(nd_float)

        # Convert to uint8
        final_data = final_data_float.astype("uint8")

        # Inherit Georeferencing
        final_data.rio.write_crs(da_later.rio.crs, inplace=True)
        final_data.rio.write_nodata(int(nd_float), encoded=True, inplace=True)

        # Write Temp GeoTIFF
        tmp_gtiff = out_path.with_suffix(".tmp.tif")
        final_data.rio.to_raster(tmp_gtiff, compress="DEFLATE", tiled=True, dtype="uint8")

        # Inject Colormap and Metadata
        with rasterio.open(tmp_gtiff, 'r+') as dst:
            dst.write_colormap(1, active_colormap)
            tags = {f"CLASS_{k}": v for k, v in active_names.items()}
            dst.update_tags(**tags)

        # Convert to COG
        save_gtiff_as_cog(tmp_gtiff, out_path)
        try: 
            tmp_gtiff.unlink(missing_ok=True)
        except: 
            pass

        print(f"[INFO] Categorical difference written to {out_path}")


def compute_and_write_difference_positive_change_only(
    earlier_path: Path,
    later_path: Path,
    out_path: Path,
    nodata_value: int = 255
):
    """
    Computes a binary 'Positive Change' (Water Gain) raster.
    Agnostic to DSWx-S1 and DSWx-HLS (WTR and BWTR layers).

    Logic (Assign 1 - Blue):
      1. New Water: Not Water (0) -> Any Water Class (1, 2, 3)
      2. Intensification: Partial/Veg Water (2, 3) -> Open Water (1)
    
    Logic (Assign 0 - White):
      - All other valid transitions (e.g., 1->1, 1->0, 3->3).
      
    Logic (Assign 255 - NoData):
      - If either input is NoData (255) or Masked (>= 250).
      
    Metadata:
      - Embeds a colormap: 0=White, 1=Blue (0,0,200,255).
      - Embeds CLASS names for QGIS/GDAL.
    """
    import rioxarray
    import xarray as xr
    import numpy as np
    import rasterio

    print(f"[INFO] Computing generalized positive change (gain) layer for {out_path.name}...")

    # 1. Open rasters
    # Open unmasked to handle raw integer values directly
    da_later = rioxarray.open_rasterio(later_path, masked=False).squeeze()
    da_early = rioxarray.open_rasterio(earlier_path, masked=False).squeeze()

    # 2. Define Groups
    # Valid Water Classes across S1 and HLS WTR/BWTR:
    # 1: Open Water (S1/HLS)
    # 2: Partial Surface Water (HLS)
    # 3: Inundated Vegetation (S1)
    any_water = [1, 2, 3]
    partial_or_veg = [2, 3]
    open_water = 1
    not_water = 0

    # 3. Create Masks
    # A. Nodata/Masks: Values >= 250 are essentially invalid/masked in all DSWx products
    #    (250=HAND, 251=Layover, 252=Snow/Ice, 253=Cloud, 254=Ocean, 255=Fill)
    mask_invalid = (da_early >= 250) | (da_later >= 250)

    # B. Condition 1: New Water (0 -> 1, 2, 3)
    #    Use np.isin for checking membership in list
    cond_new_water = (da_early == not_water) & (np.isin(da_later, any_water))

    # C. Condition 2: Intensification (2, 3 -> 1)
    #    Partial/Veg becoming Open Water
    cond_intensification = (np.isin(da_early, partial_or_veg)) & (da_later == open_water)

    # Combine Gain Conditions
    mask_gain = cond_new_water | cond_intensification

    # 4. Create Output Array (uint8)
    # Initialize with 0 (No Positive Change)
    out_data = np.zeros_like(da_early.values, dtype="uint8")
    
    # Apply Gain (Set to 1)
    out_data[mask_gain.values] = 1
    
    # Apply Nodata (Set to 255) - Overwrites any previous assignment
    out_data[mask_invalid.values] = 255

    # 5. Wrap in xarray for CRS/Transform handling
    da_out = xr.DataArray(
        out_data,
        coords=da_later.coords,
        dims=da_later.dims,
        attrs=da_later.attrs
    )
    da_out.rio.write_crs(da_later.rio.crs, inplace=True)
    da_out.rio.write_nodata(255, inplace=True)

    # 6. Write to Temporary GeoTIFF
    tmp_gtiff = out_path.with_suffix(".tmp.tif")
    da_out.rio.to_raster(tmp_gtiff, compress="DEFLATE", tiled=True, dtype="uint8")

    # 7. Inject Colormap and Metadata
    # Color 0: White (255, 255, 255, 255)
    # Color 1: Blue  (0, 0, 200, 255)
    custom_colormap = {
        0: (255, 255, 255, 255),
        1: (0, 0, 200, 255)
    }

    # Define class names for QGIS Legend
    class_names = {
        0: "No Change or Water Loss",
        1: "Water Gain"
    }

    with rasterio.open(tmp_gtiff, 'r+') as dst:
        dst.write_colormap(1, custom_colormap)
        tags = {f"CLASS_{k}": v for k, v in class_names.items()}
        dst.update_tags(**tags)

    # 8. Convert to COG
    save_gtiff_as_cog(tmp_gtiff, out_path)

    # Cleanup
    try:
        tmp_gtiff.unlink(missing_ok=True)
    except:
        pass

    print(f"[INFO] Positive change difference written to {out_path}")


def generate_products(
    df_opera,
    mode,
    mode_dir,
    layout_title,
    bbox,
    zoom_bbox,
    filter_date=None,
    reclassify_snow_ice=False,
):
    """
    Generate mosaicked products, maps, and layouts based on the provided DataFrame and mode.
    The logic is updated to generate a separate mosaic for each unique UTM zone present
    in the intersecting granules for a given date, ensuring high-fidelity data processing.

    Args:
        df_opera (pd.DataFrame): DataFrame containing OPERA products metadata.
        mode (str): Mode of operation, e.g., "flood", "fire", "landslide", "earthquake".
        mode_dir (Path): Path to the directory where products will be saved.
        layout_title (str): Title for the PDF layout(s).
        bbox (list): Bounding box in the form [South, North, West, East].
        zoom_bbox (list): Optional bounding box for the zoom-in inset map in the form [South, North, West, East].
        filter_date (str, optional): Date string (YYYY-MM-DD) to filter by date in the date filtering step in 'fire' and 'landslide' mode.
        reclassify_snow_ice (bool, optional): Whether to reclassify false snow/ice positives as water in DSWx-HLS products ONLY. Default is False.
    Raises:
        Exception: If the mode is not recognized or if there are issues with data processing.
    """
    import re
    from collections import defaultdict

    from rasterio.shutil import copy

    import opera_mosaic

    # Create data directory
    data_dir = mode_dir / "data"
    make_output_dir(data_dir)

    # Create maps directory
    maps_dir = mode_dir / "maps"
    make_output_dir(maps_dir)

    # Create layouts directory
    layouts_dir = mode_dir / "layouts"
    make_output_dir(layouts_dir)

    if mode == "flood":
        short_names = ["OPERA_L3_DSWX-HLS_V1", "OPERA_L3_DSWX-S1_V1"]
        layer_names = ["WTR", "BWTR"]
    elif mode == "fire":
        short_names = ["OPERA_L3_DIST-ALERT-HLS_V1", "OPERA_L3_DIST-ALERT-S1_V1"]
        layer_names = ["VEG-ANOM-MAX", "VEG-DIST-STATUS"]
    elif mode == "landslide":
        short_names = ["OPERA_L3_DIST-ALERT-HLS_V1", "OPERA_L2_RTC-S1_V1"]
        layer_names = ["VEG-ANOM-MAX", "VEG-DIST-STATUS", "RTC-VV", "RTC-VH"]
    elif mode == "earthquake":
        print("[INFO] Earthquake mode coming soon. Exiting...")
        return

    # Extract and find unique dates, sort them
    df_opera["Start Date"] = df_opera["Start Time"].dt.date.astype(str)
    unique_dates = df_opera["Start Date"].dropna().unique()
    unique_dates.sort()

    # Create an index of mosaics created for use in pair-wise differencing
    mosaic_index = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for date in unique_dates:
        df_on_date = df_opera[df_opera["Start Date"] == date]

        for short_name in short_names:
            df_sn = df_on_date[df_on_date["Dataset"] == short_name]

            if df_sn.empty:
                continue

            for layer in layer_names:
                url_column = f"Download URL {layer}"
                if url_column not in df_sn.columns:
                    continue

                urls = df_sn[url_column].dropna().tolist()
                if not urls:
                    continue

                print(f"[INFO] Processing {short_name} - {layer} on {date}")
                print(f"[INFO] Found {len(urls)} URLs")

                layout_date = ""
                DS, conf_DS, date_DS = None, None, None

                if mode == "fire":
                    date_column = "Download URL VEG-DIST-DATE"
                    conf_column = "Download URL VEG-DIST-CONF"
                    date_layer_links = (
                        df_sn[date_column].dropna().tolist()
                        if date_column in df_sn.columns
                        else []
                    )
                    conf_layer_links = (
                        df_sn[conf_column].dropna().tolist()
                        if conf_column in df_sn.columns
                        else []
                    )
                    if not date_layer_links:
                        print(
                            f"[WARN] No VEG-DIST-DATE URLs found for {short_name} on {date}"
                        )
                    else:
                        print(
                            f"[INFO] Found {len(date_layer_links)} VEG-DIST-DATE URLs"
                        )
                    if not conf_layer_links:
                        print(
                            f"[WARN] No VEG-DIST-CONF URLs found for {short_name} on {date}"
                        )
                    else:
                        print(f"[INFO] Found {len(conf_layer_links)} CONF URLs")
                    DS, date_DS, conf_DS = compile_and_load_data(
                        urls,
                        mode,
                        conf_layer_links=conf_layer_links,
                        date_layer_links=date_layer_links,
                    )
                    if filter_date:
                        date_threshold = compute_date_threshold(filter_date)
                        layout_date = str(filter_date)
                        print(
                            f"[INFO] date_threshold set to {date_threshold} for filter_date {filter_date}"
                        )
                    else:
                        date_threshold = 0
                        layout_date = "All Dates"
                        print(
                            f"[INFO] date_threshold set to {date_threshold} for filter_date {filter_date}"
                        )

                elif mode == "landslide":
                    if short_name == "OPERA_L3_DIST-ALERT-HLS_V1":
                        date_column = "Download URL VEG-DIST-DATE"
                        conf_column = "Download URL VEG-DIST-CONF"
                        date_layer_links = (
                            df_sn[date_column].dropna().tolist()
                            if date_column in df_sn.columns
                            else []
                        )
                        conf_layer_links = (
                            df_sn[conf_column].dropna().tolist()
                            if conf_column in df_sn.columns
                            else []
                        )
                        if not date_layer_links:
                            print(
                                f"[WARN] No VEG-DIST-DATE URLs found for {short_name} on {date}"
                            )
                        else:
                            print(
                                f"[INFO] Found {len(date_layer_links)} VEG-DIST-DATE URLs"
                            )
                        if not conf_layer_links:
                            print(
                                f"[WARN] No VEG-DIST-CONF URLs found for {short_name} on {date}"
                            )
                        else:
                            print(f"[INFO] Found {len(conf_layer_links)} CONF URLs")
                        DS, date_DS, conf_DS = compile_and_load_data(
                            urls,
                            mode,
                            conf_layer_links=conf_layer_links,
                            date_layer_links=date_layer_links,
                        )

                        if filter_date:
                            date_threshold = compute_date_threshold(filter_date)
                            layout_date = str(filter_date)
                            print(
                                f"[INFO] date_threshold set to {date_threshold} for filter_date {filter_date}"
                            )
                        else:
                            date_threshold = 0
                            layout_date = "All Dates"
                            print(
                                f"[INFO] date_threshold set to {date_threshold} for filter_date {filter_date}"
                            )

                    elif short_name == "OPERA_L2_RTC-S1_V1":
                        DS = compile_and_load_data(urls, mode)

                elif mode == "flood":
                    conf_column = "Download URL CONF"
                    conf_layer_links = (
                        df_sn[conf_column].dropna().tolist()
                        if conf_column in df_sn.columns
                        else []
                    )
                    if not conf_layer_links:
                        print(f"[WARN] No CONF URLs found for {short_name} on {date}")
                        conf_DS = None
                    else:
                        print(f"[INFO] Found {len(conf_layer_links)} CONF URLs")

                    DS, conf_DS = compile_and_load_data(
                        urls, mode, conf_layer_links=conf_layer_links
                    )

                # Group loaded DataArrays by CRS (UTM Zone)
                crs_groups = defaultdict(list)
                conf_groups = defaultdict(list)
                date_groups = defaultdict(list)

                # Ensure all lists are non-empty before zipping
                if not DS:
                    print(
                        f"[WARN] No datasets loaded for {short_name} - {layer} on {date}. Skipping."
                    )
                    continue

                # Determine auxiliary list lengths for zipping
                aux_lists = []
                if conf_DS is not None and mode == "flood":
                    aux_lists.append(conf_DS)
                elif conf_DS is not None and mode in ["fire", "landslide"]:
                    aux_lists.extend([date_DS, conf_DS])

                if aux_lists:
                    # Zip DS with auxiliary layers (conf_DS, date_DS)
                    for i, (da_data, *aux_data) in enumerate(zip(DS, *aux_lists)):
                        try:
                            crs_str = str(da_data.rio.crs)
                        except AttributeError:
                            print(f"[WARN] Granule {i} missing CRS metadata. Skipping.")
                            continue

                        crs_groups[crs_str].append(da_data)
                        if mode == "flood":
                            conf_groups[crs_str].append(aux_data[0])
                        elif mode in ["fire", "landslide"] and short_name.startswith(
                            "OPERA_L3_DIST"
                        ):
                            date_groups[crs_str].append(aux_data[0])
                            conf_groups[crs_str].append(aux_data[1])
                else:
                    # Only DS is present (e.g., RTC-S1)
                    for i, da_data in enumerate(DS):
                        try:
                            crs_str = str(da_data.rio.crs)
                        except AttributeError:
                            print(f"[WARN] Granule {i} missing CRS metadata. Skipping.")
                            continue
                        crs_groups[crs_str].append(da_data)

                # Iterate through each CRS group to process and mosaic
                for crs_str, ds_group in crs_groups.items():

                    # Generate a descriptive UTM suffix (e.g., _18N, _17S)
                    utm_suffix = ""

                    # Try to extract the UTM Zone number and Hemisphere (N/S) from the CRS string
                    utm_match = re.search(r"UTM Zone (\d{1,2})([NS])", crs_str)

                    if utm_match:
                        zone_number = utm_match.group(1)
                        hemisphere = utm_match.group(2)
                        utm_suffix = f"_{zone_number}{hemisphere}"
                        print(f"[INFO] Detected UTM Zone: {utm_suffix}")
                    else:
                        # Fallback to EPSG if UTM info isn't explicitly in the WKT string (e.g., just "EPSG:32618")
                        crs_match = re.search(r"EPSG:(\d+)", crs_str)
                        if crs_match:
                            epsg_code = int(crs_match.group(1))

                            # Standard UTM EPSG codes are 326xx (North) or 327xx (South)
                            if 32600 <= epsg_code <= 32660:
                                zone_number = epsg_code - 32600
                                utm_suffix = f"_{zone_number}N"
                                print(
                                    f"[INFO] Deduced UTM Zone from N-EPSG: {utm_suffix}"
                                )
                            elif 32700 <= epsg_code <= 32760:
                                zone_number = epsg_code - 32700
                                utm_suffix = f"_{zone_number}S"
                                print(
                                    f"[INFO] Deduced UTM Zone from S-EPSG: {utm_suffix}"
                                )
                            else:
                                # Fallback for non-standard or unknown EPSG
                                utm_suffix = f"_EPSG{epsg_code}"
                                print(
                                    f"[WARN] Non-UTM EPSG code {epsg_code}. Using suffix: {utm_suffix}"
                                )
                        else:
                            # Last resort: use a hash of the full CRS string
                            print(
                                f"[WARN] Could not parse UTM or EPSG from CRS string: {crs_str}. Using raw hash."
                            )
                            utm_suffix = f"_Hash{hash(crs_str) % 10000}"

                    current_conf_DS = conf_groups.get(crs_str)
                    current_date_DS = date_groups.get(crs_str)

                    colormap = None  # Initialize colormap

                    # Filtering/Reclassification (Per CRS Group)
                    if mode == "fire" or (
                        mode == "landslide" and short_name.startswith("OPERA_L3_DIST")
                    ):
                        # Filter DIST layers by date and confidence
                        ds_group, colormap = filter_by_date_and_confidence(
                            ds_group,
                            current_date_DS,
                            date_threshold,
                            DS_conf=current_conf_DS,
                            confidence_threshold=0,
                            fill_value=None,
                        )

                    elif mode == "flood":
                        if (
                            reclassify_snow_ice == True
                            and short_name == "OPERA_L3_DSWX-HLS_V1"
                            and layer in ["BWTR", "WTR"]
                        ):
                            # Reclassify false snow/ice positives in DSWX-HLS only (if user-specified --reclassify_snow_ice True)
                            if current_conf_DS is None:
                                print(
                                    f"[WARN] CONF layers not available; skipping snow/ice reclassification for {short_name} on {date}"
                                )
                            else:
                                print(
                                    f"[INFO] Reclassifying false snow/ice positives as water based on CONF layers for CRS {utm_suffix}"
                                )
                                ds_group, colormap = reclassify_snow_ice_as_water(
                                    ds_group, current_conf_DS
                                )
                        else:
                            if (
                                reclassify_snow_ice == True
                                and short_name != "OPERA_L3_DSWX-HLS_V1"
                            ):
                                print(
                                    f"[INFO] Snow/ice reclassification is only applicable to DSWx-HLS. Skipping for {short_name}."
                                )
                            else:
                                print(
                                    "[INFO] Snow/ice reclassification not requested; proceeding without reclassification."
                                )

                    # Use pre-determined colormap or make another attempt to get it. If still None, proceed without it.
                    if colormap is None:
                        try:
                            colormap = opera_mosaic.get_image_colormap(ds_group[0])
                        except Exception:
                            colormap = None

                    # Mosaic the datasets using the appropriate method/rule
                    mosaic, _, nodata = opera_mosaic.mosaic_opera(
                        ds_group, product=short_name, merge_args={}
                    )
                    image = opera_mosaic.array_to_image(
                        mosaic, colormap=colormap, nodata=nodata
                    )

                    # Create filename and full paths
                    mosaic_name = f"{short_name}_{layer}_{date}{utm_suffix}_mosaic.tif"
                    mosaic_path = data_dir / mosaic_name
                    tmp_path = data_dir / f"tmp_{mosaic_name}"

                    # Save the mosaic to a temporary GeoTIFF
                    copy(image, tmp_path, driver="GTiff")

                    warp_args = {
                        "xRes": 30,
                        "yRes": 30,
                        "creationOptions": ["COMPRESS=DEFLATE"],
                    }

                    if short_name.startswith("OPERA_L2_RTC"):
                        warp_args["outputType"] = gdal.GDT_Float32

                    # Reproject/compress using GDAL directly into the final GeoTIFF
                    gdal.Warp(mosaic_path, tmp_path, **warp_args)

                    # Convert to COG (writes back into mosaic_path)
                    save_gtiff_as_cog(mosaic_path, mosaic_path)

                    print(f"[INFO] Mosaic written as COG: {mosaic_path}")

                    # Clean up tmp file
                    if tmp_path.exists():
                        tmp_path.unlink()

                    # Add info to the mosiac index for pair-wise differencing
                    with rasterio.open(mosaic_path) as ds:
                        mosaic_crs = ds.crs

                    # UPDATED STRUCTURE: [short_name][layer][date] = {utm_suffix: info_dict}
                    mosaic_index[short_name][layer][str(date)][utm_suffix] = {
                        "path": mosaic_path,
                        "crs": mosaic_crs,
                    }
                    # Make a map with PyGMT
                    map_name = make_map(
                        maps_dir,
                        mosaic_path,
                        short_name,
                        layer,
                        date,
                        bbox,
                        zoom_bbox,
                        is_difference=False,
                        utm_suffix=utm_suffix,
                    )

                    # Make a PDF layout
                    make_layout(
                        layouts_dir,
                        map_name,
                        short_name,
                        layer,
                        date,
                        layout_date,
                        layout_title,
                        reclassify_snow_ice,
                        utm_suffix=utm_suffix,
                    )

    # Pair-wise differencing for 'flood' mode
    if mode == "flood":
        print("[INFO] Computing pairwise differences between water products...")
        skipped = []

        for short_name_k, layers_dict in mosaic_index.items():
            for layer_k, date_map in layers_dict.items():

                # Restructure by UTM zone for differencing
                utm_date_map = defaultdict(lambda: defaultdict(dict))
                for d, utm_dict in date_map.items():
                    for utm, info in utm_dict.items():
                        utm_date_map[utm][d] = info

                for utm_suffix_k, utm_map in utm_date_map.items():
                    dates = sorted(utm_map.keys())

                    # Generate difference for all possible pairs within this UTM zone
                    for i in range(len(dates)):
                        for j in range(i + 1, len(dates)):
                            d_early = dates[i]
                            d_later = dates[j]

                            early_info = utm_map[d_early]
                            later_info = utm_map[d_later]

                            crs_a = early_info["crs"]
                            crs_b = later_info["crs"]

                            # The check will always pass if utm_map was created correctly.
                            if not same_utm_zone(crs_a, crs_b):
                                # This block should rarely be hit if the grouping is correct
                                skipped.append(
                                    {
                                        "short_name": short_name_k,
                                        "layer": layer_k,
                                        "date_earlier": d_early,
                                        "date_later": d_later,
                                        "utm_suffix": utm_suffix_k,
                                        "crs_earlier": (
                                            crs_a.to_string() if crs_a else None
                                        ),
                                        "crs_later": (
                                            crs_b.to_string() if crs_b else None
                                        ),
                                        "reason": "internal CRS mismatch after grouping (error)",
                                    }
                                )
                                continue

                            # Name and path: {short}_{layer}_{LATER}_{EARLIER}{UTM}_diff.tif
                            diff_name = f"{short_name_k}_{layer_k}_{d_later}_{d_early}{utm_suffix_k}_water_change.tif"
                            diff_path = (mode_dir / "data") / diff_name

                        try:
                            compute_and_write_difference_positive_change_only(
                                earlier_path=early_info["path"],
                                later_path=later_info["path"],
                                out_path=diff_path,
                                nodata_value=None,
                            )
                            print(f"[INFO] Wrote diff COG: {diff_path}")
                            
                            # Make a map with PyGMT
                            diff_date_str = f"{d_later}_{d_early}"
                            map_name = make_map(maps_dir, diff_path, short_name_k, layer_k, diff_date_str, bbox, zoom_bbox, is_difference=True)

                            # Make a PDF layout
                            if map_name:
                                diff_date_str_layout = f"{d_early}, {d_later}"
                                make_layout(layouts_dir, map_name, short_name_k, layer_k, diff_date_str, diff_date_str_layout, layout_title, reclassify_snow_ice)
                        
                        except Exception as e:
                            skipped.append({
                                "short_name": short_name_k,
                                "layer": layer_k,
                                "date_earlier": d_early,
                                "date_later": d_later,
                                "error": str(e),
                                "reason": "no overlapping data values; both rasters contain only nodata in the overlap region."
                            })

        # Report skipped pairs due to CRS/UTM differences or errors
        report_path = (mode_dir / "data") / "difference_skipped_pairs.json"
        with open(report_path, "w") as f:
            json.dump(skipped, f, indent=2)
        print(f"[INFO] Difference skip report: {report_path} ({len(skipped)} skipped)")


    # Pair-wise differencing for 'landslide' mode (RTC-S1 log difference)
    if mode == "landslide":
        print(
            "[INFO] Computing pairwise log difference between RTC backscatter products..."
        )
        skipped = []

        for short_name_k, layers_dict in mosaic_index.items():
            for layer_k, date_map in layers_dict.items():

                utm_date_map = defaultdict(lambda: defaultdict(dict))
                for d, utm_dict in date_map.items():
                    for utm, info in utm_dict.items():
                        utm_date_map[utm][d] = info

                for utm_suffix_k, utm_map in utm_date_map.items():
                    # Only compute log-diff for RTC products
                    if short_name_k != "OPERA_L2_RTC-S1_V1":
                        continue

                    dates = sorted(utm_map.keys())

                    # Generate difference for all possible pairs within this UTM zone
                    for i in range(len(dates)):
                        for j in range(i + 1, len(dates)):
                            d_early = dates[i]
                            d_later = dates[j]

                            early_info = utm_map[d_early]
                            later_info = utm_map[d_later]

                            # The UTM/CRS check is now mostly redundant since they are grouped.
                            crs_a = early_info["crs"]
                            crs_b = later_info["crs"]
                            if not same_utm_zone(crs_a, crs_b):
                                skipped.append(
                                    {
                                        "short_name": short_name_k,
                                        "layer": layer_k,
                                        "date_earlier": d_early,
                                        "date_later": d_later,
                                        "utm_suffix": utm_suffix_k,
                                        "crs_earlier": (
                                            crs_a.to_string() if crs_a else None
                                        ),
                                        "crs_later": (
                                            crs_b.to_string() if crs_b else None
                                        ),
                                        "reason": "internal CRS mismatch after grouping (error)",
                                    }
                                )
                                continue

                            # Name and path: {short}_{layer}_{LATER}_{EARLIER}{UTM}_log_diff.tif
                            diff_name = f"{short_name_k}_{layer_k}_{d_later}_{d_early}{utm_suffix_k}_log-diff.tif"
                            diff_path = (mode_dir / "data") / diff_name

                            try:
                                compute_and_write_difference(
                                    earlier_path=early_info["path"],
                                    later_path=later_info["path"],
                                    out_path=diff_path,
                                    nodata_value=None,
                                    log=True,
                                )
                                print(f"[INFO] Wrote diff COG: {diff_path}")

                                # Make a map with PyGMT
                                diff_date_str = f"{d_later}_{d_early}"
                                map_name = make_map(
                                    maps_dir,
                                    diff_path,
                                    short_name_k,
                                    layer_k,
                                    diff_date_str,
                                    bbox,
                                    zoom_bbox,
                                    is_difference=True,
                                    utm_suffix=utm_suffix_k,
                                )

                                # Make a PDF layout
                                if map_name:
                                    diff_date_str_layout = f"{d_early}, {d_later}"
                                    make_layout(
                                        layouts_dir,
                                        map_name,
                                        short_name_k,
                                        layer_k,
                                        diff_date_str,
                                        diff_date_str_layout,
                                        layout_title,
                                        reclassify_snow_ice,
                                        utm_suffix=utm_suffix_k,
                                    )

                            except Exception as e:
                                skipped.append(
                                    {
                                        "short_name": short_name_k,
                                        "layer": layer_k,
                                        "date_earlier": d_early,
                                        "date_later": d_later,
                                        "utm_suffix": utm_suffix_k,
                                        "error": str(e),
                                        "reason": "no overlapping data values; both rasters contain only nodata in the overlap region.",
                                    }
                                )

        # Report skipped pairs due to CRS/UTM differences or errors
        report_path = (mode_dir / "data") / "log-difference_skipped_pairs.json"
        with open(report_path, "w") as f:
            json.dump(skipped, f, indent=2)
        print(
            f"[INFO] Log-difference skip report: {report_path} ({len(skipped)} skipped)"
        )
    return


def expand_region(region, width_deg=15, height_deg=10):
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
    Raises:
        ValueError: If the input region is not in the correct format.
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


def expand_region_to_aspect(region, target_aspect):
    """
    Expand the input region to match a target aspect ratio.
    Args:
        region (list): Input region in the form [xmin, xmax, ymin, ymax].
        target_aspect (float): Desired aspect ratio (width / height).
    Returns:
        list: New region with adjusted aspect ratio.
    Raises:
        ValueError: If the input region is not in the correct format.
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
    maps_dir,
    mosaic_path,
    short_name,
    layer,
    date,
    bbox,
    zoom_bbox=None,
    is_difference=False,
    utm_suffix="",
):
    """
    Create a map using PyGMT from the provided mosaic path.

    Args:
        maps_dir (Path): Directory where the map will be saved.
        mosaic_path (Path): Path to the mosaic file.
        short_name (str): Short name of the product.
        layer (str): Layer name to be used in the map.
        date (str): Date string in the format YYYY-MM-DD.
        bbox (list): Bounding box in the form [South, North, West, East].
        zoom_bbox (list, optional): Bounding box for the zoom-in inset map, in the form [South, North, West, East].
        is_difference (bool, optional): Flag to indicate if the mosaic is a difference product. Defaults to False.
        utm_suffix (str, optional): UTM zone suffix (e.g., '_18N') for unique naming. Defaults to "".

    Returns:
        map_name (Path): Path to the saved map image.
    Raises:
        ImportError: If required libraries are not installed.
    """
    import math
    import os
    import re

    import pygmt
    import rioxarray
    from pygmt.params import Box
    from pyproj import Geod

    # Determine date string for filename
    if is_difference:
        match = re.search(
            r"(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_(\_\d+N|\_\d+S|\_EPSG\d+|\_Hash\d+)?(?:log-)?diff",
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

    # Create a temporary path for the WGS84 reprojected file
    mosaic_wgs84 = Path(str(mosaic_path).replace(".tif", f"_WGS84_TMP{utm_suffix}.tif"))

    def cleanup_temp_file(filepath):
        """Helper to safely remove the temporary file."""
        if filepath.exists():
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"[WARN] Failed to clean up temporary WGS84 file {filepath}: {e}")

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

        # 'landslide' mode (RTC)
        if is_difference:
            is_log_diff = "_log-diff" in str(mosaic_path)
            
            # --- CASE A: CONTINUOUS SCALE (RTC / Landslide) ---
            if is_log_diff:
                data_values = grd.values[~np.isnan(grd.values)]
                if len(data_values) == 0:
                    print(f"[WARN] No valid data in {mosaic_path}")
                    cleanup_temp_file(mosaic_wgs84)
                    return None

                p2, p98 = np.percentile(data_values, [2, 98])
                symmetric_limit = max(abs(p2), abs(p98))
                if symmetric_limit == 0: symmetric_limit = 1 
                
                p_min = -symmetric_limit
                p_max = symmetric_limit
                inc = (p_max - p_min) / 1000.0

                cpt_name = "difference_cpt"
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
                    cpt_path = maps_dir / "binary_gain.cpt"
                    
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

                    # Simple Legend
                    legend_path = maps_dir / "binary_legend.txt"
                    with open(legend_path, "w") as f:
                        f.write("H 10p,Helvetica-Bold Water Change\n")
                        f.write("D 0.2c 1p\n") 
                        f.write("S 0.3c s 0.3c 0/0/200 0.25p 0.5c Water Gain\n")
                    
                    fig.legend(spec=str(legend_path), position="JBC+jTC+o0c/1.0c+w4c", box="+gwhite+p1p")
                    
                    # Cleanup
                    try:
                        os.remove(cpt_path)
                        os.remove(legend_path)
                    except: pass

                # --- Sub-Case B2: Full Categorical (Max value > 1) ---
                else:
                    cpt_path = maps_dir / "categorical_diff.cpt"
                    
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

                    # Full Legend
                    legend_path = maps_dir / "categorical_legend.txt"
                    with open(legend_path, "w") as f:
                        f.write("H 10p,Helvetica-Bold Water Change Classes\n")
                        f.write("D 0.2c 1p\n")
                        f.write("S 0.3c s 0.3c 0/0/200 0.25p 0.5c Water Gain (Inundation)\n")
                        f.write("S 0.3c s 0.3c 30/144/255 0.25p 0.5c Water Gain (Partial)\n")
                        f.write("S 0.3c s 0.3c 200/0/0 0.25p 0.5c Water Loss (Drying)\n")
                        f.write("S 0.3c s 0.3c 255/127/80 0.25p 0.5c Water Loss (Partial)\n")
                        f.write("S 0.3c s 0.3c 0/0/0 0.25p 0.5c Stable Water\n")
                        
                    fig.legend(spec=str(legend_path), position="JBC+jTC+o0c/1.0c+w5c", box="+gwhite+p1p")
                    
                    try: 
                        os.remove(cpt_path)
                        os.remove(legend_path)
                    except:
                        pass

        # Add grid image (based on product/layer)
        elif short_name == "OPERA_L3_DSWX-HLS_V1" and layer == "WTR":
            color_palette = "palettes/DSWx-HLS_WTR.cpt"
            fig.grdimage(
                grid=grd,
                region=region_padded,
                projection=projection,
                cmap=color_palette,
                frame=["WSne", "xaf", "yaf"],
                nan_transparent=True,
            )
            fig.colorbar(cmap=color_palette, equalsize=1.5)

        elif short_name == "OPERA_L3_DSWX-HLS_V1" and layer == "BWTR":
            color_palette = "palettes/DSWx-HLS_BWTR.cpt"
            fig.grdimage(
                grid=grd,
                region=region_padded,
                projection=projection,
                cmap=color_palette,
                frame=["WSne", "xaf", "yaf"],
                nan_transparent=True,
            )
            fig.colorbar(cmap=color_palette, equalsize=1.5)

        elif short_name == "OPERA_L3_DSWX-S1_V1" and layer == "WTR":
            color_palette = "palettes/DSWx-S1_WTR.cpt"
            fig.grdimage(
                grid=grd,
                region=region_padded,
                projection=projection,
                cmap=color_palette,
                frame=["WSne", "xaf", "yaf"],
                nan_transparent=True,
            )
            fig.colorbar(cmap=color_palette, equalsize=1.5)

        elif short_name == "OPERA_L3_DSWX-S1_V1" and layer == "BWTR":
            color_palette = "palettes/DSWx-S1_BWTR.cpt"
            fig.grdimage(
                grid=grd,
                region=region_padded,
                projection=projection,
                cmap=color_palette,
                frame=["WSne", "xaf", "yaf"],
                nan_transparent=True,
            )
            fig.colorbar(cmap=color_palette, equalsize=1.5)

        elif layer == "VEG-ANOM-MAX":
            color_palette = "palettes/VEG-ANOM-MAX.cpt"
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

        elif layer == "VEG-DIST-STATUS":
            color_palette = "palettes/VEG-DIST-STATUS.cpt"
            fig.grdimage(
                grid=grd,
                region=region_padded,
                projection=projection,
                cmap=color_palette,
                frame=["WSne", "xaf", "yaf"],
                nan_transparent=True,
            )
            fig.colorbar(cmap=color_palette, equalsize=1.5)

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

            cpt_name = "rtc_grayscale"

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
                if short_name == "OPERA_L3_DSWX-HLS_V1" and layer == "WTR":
                    fig.grdimage(
                        grid=grd,
                        region=zoom_region,
                        projection="M5c",
                        cmap=color_palette,
                        nan_transparent=True,
                    )
                elif short_name == "OPERA_L3_DSWX-HLS_V1" and layer == "BWTR":
                    fig.grdimage(
                        grid=grd,
                        region=zoom_region,
                        projection="M5c",
                        cmap=color_palette,
                        nan_transparent=True,
                    )
                elif short_name == "OPERA_L3_DSWX-S1_V1" and layer == "WTR":
                    fig.grdimage(
                        grid=grd,
                        region=zoom_region,
                        projection="M5c",
                        cmap=color_palette,
                        nan_transparent=True,
                    )
                elif short_name == "OPERA_L3_DSWX-S1_V1" and layer == "BWTR":
                    fig.grdimage(
                        grid=grd,
                        region=zoom_region,
                        projection="M5c",
                        cmap=color_palette,
                        nan_transparent=True,
                    )
                elif layer == "VEG-ANOM-MAX":
                    fig.grdimage(
                        grid=grd,
                        region=zoom_region,
                        projection="M5c",
                        cmap=color_palette,
                        nan_transparent=True,
                    )
                elif layer == "VEG-DIST-STATUS":
                    fig.grdimage(
                        grid=grd,
                        region=zoom_region,
                        projection="M5c",
                        cmap=color_palette,
                        nan_transparent=True,
                    )
                elif short_name.startswith("OPERA_L2_RTC"):
                    fig.grdimage(
                        grid=grd,
                        region=zoom_region,
                        projection="M5c",
                        cmap=cpt_name,
                        nan_transparent=True,
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
        map_name = maps_dir / f"{short_name}_{layer}_{date}{utm_suffix}_map.png"
        fig.savefig(map_name, dpi=900)
        cleanup_temp_file(mosaic_wgs84)

        return map_name

    except Exception as e:
        cleanup_temp_file(mosaic_wgs84)
        print(f"[ERROR] An error occurred during map generation for {mosaic_path}: {e}")
        raise


def make_layout(
    layout_dir,
    map_name,
    short_name,
    layer,
    date,
    layout_date,
    layout_title,
    reclassify_snow_ice=False,
    utm_suffix="",
):
    """
    Create a layout using matplotlib for the provided map.
    Args:
        layout_dir (Path): Directory where the layout will be saved.
        map_name (Path): Path to the map image.
        short_name (str): Short name of the product.
        layer (str): Layer name to be used in the layout.
        date (str): Date string in the format YYYY-MM-DD.
        layout_date (str): Date threshold in the format YYYY-MM-DD.
        layout_title (str): Title for the layout.
        reclassify_snow_ice (bool, optional): Flag indicating if snow/ice reclassification was applied. Defaults to False.
    """
    import textwrap

    import matplotlib.image as mpimg
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

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
    logo_opera = imread("logos/OPERA_logo.png")
    logo_new = imread("logos/ARIA_logo.png")

    # Create a new axes for logos in the bottom-right corner
    logo_ax = fig.add_axes([0.82, 0.02, 0.06, 0.08], anchor="SE", zorder=10)
    logo_ax.imshow(logo_opera)
    logo_ax.axis("off")

    logo_ax2 = fig.add_axes([0.89, 0.02, 0.06, 0.08], anchor="SE", zorder=10)
    logo_ax2.imshow(logo_new)
    logo_ax2.axis("off")

    # Layout control
    x_pos = 0.65  # Just right of the map
    line_spacing = 0.04  # vertical spacing between blocks

    # Define wrap width (in characters)
    wrap_width = 50

    # Map text elements
    if short_name == "OPERA_L3_DSWX-S1_V1":
        subtitle = "OPERA Dynamic Surface Water eXtent from Sentinel-1 (DSWx-S1)"
        map_information = (
            f"The ARIA/OPERA water extent map is derived from an OPERA DSWx-S1 mosaicked "
            f"product from Copernicus Sentinel-1 data."
            f"This map depicts regions of full surface water and inundated surface water. "
        )
        data_source = "Copernicus Sentinel-1"

    elif short_name == "OPERA_L3_DSWX-HLS_V1":
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

    acquisitions = f"{date}"

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

    plt.tight_layout()

    # layout_name = layout_dir / f"{short_name}_{layer}_{date}_layout.pdf"
    layout_name = layout_dir / f"{short_name}_{layer}_{date}{utm_suffix}_layout.pdf"
    plt.savefig(layout_name, format="pdf", bbox_inches="tight", dpi=400)
    return


def save_gtiff_as_cog(src_path: Path, dst_path: Path | None = None):
    if dst_path is None or src_path == dst_path:
        tmp_path = src_path.with_suffix(".cog.tmp.tif")
        dst_path = tmp_path
        in_place = True
    else:
        in_place = False

    ds = gdal.Open(str(src_path))
    if ds is None:
        raise RuntimeError(f"Could not open {src_path} for COG translation")

    creation_opts = [
        "COMPRESS=DEFLATE",
        "PREDICTOR=2",
        "BLOCKSIZE=512",
        "OVERVIEWS=IGNORE_EXISTING",
        "LEVEL=9",
        "BIGTIFF=IF_SAFER",
        "SPARSE_OK=YES",
        "RESAMPLING=AVERAGE",
    ]
    gdal.Translate(str(dst_path), ds, format="COG", creationOptions=creation_opts)

    if in_place:
        os.replace(dst_path, src_path)


def main():
    """
    Main entry point for the disaster analysis workflow.
    This function parses command line arguments, sets up the output directory,
    authenticates with Earthdata and ASF, and runs the next_pass module to generate
    disaster products based on the specified mode (flood, fire, earthquake).
    Raises:
        Exception: If there are issues with directory creation, CSV reading, or product generation.
    """
    args = parse_arguments()

    # Terminate if user selects 'earthquake' mode, for now
    if args.mode == "earthquake":
        print("[INFO] Earthquake mode coming soon. Exiting...")
        return

    output_dir = next_pass.run_next_pass(
        bbox=args.bbox,
        number_of_dates=args.number_of_dates,
        date=args.date,
        functionality=args.functionality,
    )

    make_output_dir(args.output_dir)
    dest = args.output_dir / output_dir.name
    output_dir.rename(dest)
    print(f"[INFO] Moved next_pass output directory to {dest}")

    # Read the metadata CSV file
    df_opera = read_opera_metadata_csv(dest)

    # Make a new directory with the mode name
    mode_dir = args.output_dir / args.mode
    make_output_dir(mode_dir)
    print(f"[INFO] Created mode directory: {mode_dir}")

    # Generate products based on the mode
    generate_products(
        df_opera,
        args.mode,
        mode_dir,
        args.layout_title,
        args.bbox,
        args.zoom_bbox,
        args.filter_date,
        args.reclassify_snow_ice,
    )

    return


if __name__ == "__main__":
    main()