import os
import json
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
from osgeo import gdal
import next_pass
import numpy as np
import rasterio
import rioxarray
import xarray as xr
from collections import defaultdict

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

    valid_layer_names = [
        "WTR", "BWTR", "VEG-ANOM-MAX", "VEG-DIST-STATUS"
    ]

    valid_modes = ["flood","fire","earthquake"]

    valid_functions = ["opera_search", "both"]

    parser.add_argument(
        "-b", "--bbox", nargs=4, type=float, metavar=("S", "N", "W", "E"),
        required=True, help="Bounding box in the form: South North West East"
    )

    parser.add_argument(
        "-o", "--output_dir", type=Path, required=True,
        help="Path to the directory where results and metadata will be saved."
    )

    parser.add_argument(
        "-sn", "--short_name", type=str, choices=valid_short_names,
        help="Short name to filter the DataFrame (must be one of the known OPERA products)"
    )

    parser.add_argument(
        "-l", "--layer_name", type=str, choices=valid_layer_names,
        help="Layer name to extract from metadata (e.g., 'WTR', 'BWTR', 'VEG-ANOM-MAX')"
    )

    parser.add_argument(
        "-d", "--date", type=str,
        help="Date string (YYYY-MM-DD) to filter rows by Start Date"
    )

    parser.add_argument(
        "-n", "--number_of_dates", type=int, default=5,
        help="Number of most recent dates to consider for OPERA products"
    )

    parser.add_argument(
        "-m", "--mode", type=str, default="flood", choices=valid_modes,
        help="Mode of operation: flood, fire, earthquake. Default is 'flood'."
    )

    parser.add_argument(
        "-f", "--functionality", type=str, default="opera_search", choices=valid_functions,
        help="Functionality to run: 'opera_search' or 'both'. Default is 'opera_search'."
    )

    parser.add_argument(
        "-lt", "--layout_title", type=str, required=True, default="Layout Title",
        help="Title for the PDF layout(s). Must be enclosed in double quotes and is required."
    )

    parser.add_argument(
        "-fd", "--filter_date", type=str, required=False, default=None,
        help="Date string (YYYY-MM-DD) to filter by date in the date filtering step in 'fire' mode."
    )

    return parser.parse_args()

def authenticate():
    """
    Authenticate with Earthdata and ASF for data access.
    Returns:
        tuple: (username, password) for Earthdata and ASF access.
    """
    import earthaccess
    import boto3
    import rasterio
    from rasterio.session import AWSSession
    import netrc

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
    Read the OPERA products metadata CSV file.
    Args:
        output_dir (Path): Path to the directory containing the CSV file.
    Returns:
        pd.DataFrame: DataFrame containing the metadata from the CSV file.
    Raises:
        FileNotFoundError: If the CSV file does not exist in the specified directory.
    """
    csv_path = output_dir / "opera_products_metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} rows from {csv_path}")
    return df

def compile_and_load_data(data_layer_links, mode, conf_layer_links=None, date_layer_links=None):
    """
    Compile and load data from the provided layer links for mosaicking. Also compile and load
    data for filtering (depending on whether it is provided).
    Args:
        data_layer_links (list): List of URLs corresponding to the OPERA data layers to mosaic.
        mode (str): Mode of operation, e.g., "flood", "fire", "earthquake".
        conf_layer_links (list, optional): List of URLs for additional layers to filter false positives.
        date_layer_links (list, optional): List of URLs for date layers to filter by date.
    Returns:
        DS: List of rioxarray datasets loaded from the provided links.
        conf_DS: List of rioxarray datasets for confidence layers (if applicable).
        date_DS: List of rioxarray datasets for date layers (if applicable).
    Raises:
        Exception: If there is an error loading any of the datasets.
    """
    from opera_utils.disp._remote import open_file
    from collections import Counter

    username, password = authenticate()

    DS = []
    for link in data_layer_links:
        try:
            DS.append(rioxarray.open_rasterio(link, masked=False))
        except Exception as e:
            f = open_file(
                link,
                earthdata_username=username,
                earthdata_password=password,
            )
            DS.append(rioxarray.open_rasterio(f, masked=False))

    # Sort DS by most common crs for merging
    crs_list = [str(ds.rio.crs) for ds in DS]
    crs_counter = Counter(crs_list)
    most_common_crs_str, _ = crs_counter.most_common(1)[0]
    DS.sort(key=lambda ds: 0 if str(ds.rio.crs) == most_common_crs_str else 1)

    # If conf_layer_links AND mode == 'flood' compile and load layers to use in filtering
    if conf_layer_links and mode == "flood":
        conf_DS = []
        for link in conf_layer_links:
            try:
                conf_DS.append(rioxarray.open_rasterio(link, masked=False))
            except Exception as e:
                f = open_file(
                    link,
                    earthdata_username=username,
                    earthdata_password=password,
                )
                conf_DS.append(rioxarray.open_rasterio(f, masked=False))

        # Sort conf_DS by most common crs for merging
        crs_list = [str(ds.rio.crs) for ds in conf_DS]
        crs_counter = Counter(crs_list)
        most_common_crs_str, _ = crs_counter.most_common(1)[0]
        conf_DS.sort(key=lambda ds: 0 if str(ds.rio.crs) == most_common_crs_str else 1)
        return DS, conf_DS

    # If conf_layer_links AND date_layer_links AND mode == 'fire' compile and load layers to use in filtering
    if conf_layer_links and date_layer_links and mode == "fire":
        conf_DS = []
        date_DS = []
        for link in conf_layer_links:
            try:
                conf_DS.append(rioxarray.open_rasterio(link, masked=False))
            except Exception as e:
                f = open_file(
                    link,
                    earthdata_username=username,
                    earthdata_password=password,
                )
                conf_DS.append(rioxarray.open_rasterio(f, masked=False))
        for link in date_layer_links:
            try:
                date_DS.append(rioxarray.open_rasterio(link, masked=False))
            except Exception as e:
                f = open_file(
                    link,
                    earthdata_username=username,
                    earthdata_password=password,
                )
                date_DS.append(rioxarray.open_rasterio(f, masked=False))

        # Sort conf_DS by most common crs for merging
        crs_list = [str(ds.rio.crs) for ds in conf_DS]
        crs_counter = Counter(crs_list)
        most_common_crs_str, _ = crs_counter.most_common(1)[0]
        conf_DS.sort(key=lambda ds: 0 if str(ds.rio.crs) == most_common_crs_str else 1)

        # Sort date_DS by most common crs for merging
        crs_list = [str(ds.rio.crs) for ds in date_DS]
        crs_counter = Counter(crs_list)
        most_common_crs_str, _ = crs_counter.most_common(1)[0]
        date_DS.sort(key=lambda ds: 0 if str(ds.rio.crs) == most_common_crs_str else 1)

        return DS, date_DS, conf_DS
    else:
        return DS

def reclassify_snow_ice_as_water(DS, conf_DS):
    """
    Reclassify false snow/ice positives (value 252) as water (value 1) based on the confidence layers.
    
    Args:
        DS (list): List of rioxarray datasets (BWTR layers).
        conf_DS (list): List of rioxarray datasets (CONF layers).
    
    Returns:
        list: List of updated rioxarray datasets with 252 reclassified as 1.
    """
    if conf_DS is None:
        raise ValueError("conf_DS must not be None when reclassifying snow/ice.")

    if len(DS) != len(conf_DS):
        raise ValueError("DS and conf_DS must be the same length.")
    
    values_to_reclassify = [1, 3, 4, 21, 23, 24]

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
            data_values,
            coords=da_data.coords,
            dims=da_data.dims,
            attrs=da_data.attrs
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

    return updated_list

def filter_by_date_and_confidence(DS, DS_dates, date_threshold,
                                  DS_conf=None, confidence_threshold=None,
                                  fill_value=None):
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
    list of xr.DataArray
        Filtered data granules.
    """
    assert len(DS) == len(DS_dates), "DS and DS_dates must be same length"
    if DS_conf is not None:
        assert len(DS_conf) == len(DS), "DS_conf must match DS in length"

    filtered_list = []

    for i, (da_data, da_date) in enumerate(zip(DS, DS_dates)):
        print(f"Filtering granule {i + 1}/{len(DS)}")

        # Date mask
        date_mask = da_date >= date_threshold

        # Optional confidence mask
        if DS_conf is not None and confidence_threshold is not None:
            conf_layer = DS_conf[i]
            print(f"  Confidence layer shape: {conf_layer.shape}")
            total_pixels = conf_layer.size

            # Keep only low-confidence values (<= 100)
            conf_mask = conf_layer >= confidence_threshold

            retained_pixels = conf_mask.sum().item()
            print(f"  Confidence retained: {retained_pixels} / {total_pixels} ({retained_pixels / total_pixels:.2%})")

            max_retained_conf = conf_layer.where(conf_mask).max().item()
            print(f"  Max confidence among retained pixels: {max_retained_conf}")

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

    return filtered_list

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
        return epsg_a == epsg_b and (str(epsg_a).startswith("326") or str(epsg_a).startswith("327"))
    except Exception:
        return False

def compute_and_write_difference(
    earlier_path: Path,
    later_path: Path,
    out_path: Path,
    nodata_value: float | int | None = None,
    ):
    """
    Create a difference raster: (later - earlier), aligned to the 'later' grid.
    If either pixel is nodata in the inputs, output nodata at that pixel.
    Writes a COG to out_path via the save_gtiff_as_cog helper.
    """
    import xarray as xr
    import rioxarray

    da_later = rioxarray.open_rasterio(later_path, masked=True)
    da_early = rioxarray.open_rasterio(earlier_path, masked=True)
    
    nd = nodata_value
    if nd is None:
        nd = da_later.rio.nodata
        if nd is None:
            nd = da_early.rio.nodat

    # Compute difference with nodata propagation: if either is nodata -> output nodata
    mask = xr.where(
        xr.ufuncs.isnan(da_later) | xr.ufuncs.isnan(da_early),
        True,
        False,
    )
    
    diff = da_later - da_early

    if nd is not None:
        diff = xr.where(mask, nd, diff)
        diff.rio.write_nodata(nd, encoded=True, inplace=True)
        diff.rio.write_crs(da_later.rio.crs, inplace=True)

    tmp_gtiff = out_path.with_suffix(".tmp.tif")
    diff.rio.to_raster(tmp_gtiff, compress="DEFLATE", tiled=True)

    save_gtiff_as_cog(tmp_gtiff, out_path)

    try:
        tmp_gtiff.unlink(missing_ok=True)
    except Exception:
        pass

def generate_products(df_opera, mode, mode_dir, layout_title, filter_date=None):
    """
    Generate mosaicked products, maps, and layouts based on the provided DataFrame and mode. 
    If the mode is "flood", the DSWx Confidence layer will be used to reclassify false snow/ice positives as water.
    This approach is specific to flood mode and will not be applied for other modes (i.e, fire or earthquake). 
    Args:
        df_opera (pd.DataFrame): DataFrame containing OPERA products metadata.
        mode (str): Mode of operation, e.g., "flood", "fire", "earthquake".
        mode_dir (Path): Path to the directory where products will be saved.
        layout_title (str): Title for the PDF layout(s).
        filter_date (str, optional): Date string (YYYY-MM-DD) to filter by date in the date filtering step in 'fire' mode.
    Raises:
        Exception: If the mode is not recognized or if there are issues with data processing.
    """
    import opera_mosaic
    from rasterio.shutil import copy

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
    elif mode == "earthquake":
        print("Earthquake mode coming soon. Exiting...")
        return

    df_opera['Start Time'] = pd.to_datetime(df_opera['Start Time'], errors='coerce')
    df_opera['Start Date'] = df_opera['Start Time'].dt.date.astype(str)
    unique_dates = df_opera['Start Date'].dropna().unique()

    # Create an index of mosaics created for use in pair-wise differencing
    mosaic_index = defaultdict(lambda: defaultdict(dict))

    for date in unique_dates:
        df_on_date = df_opera[df_opera['Start Date'] == date]

        for short_name in short_names:
            df_sn = df_on_date[df_on_date['Dataset'] == short_name]

            if df_sn.empty:
                continue  # No matching products for this short_name on this date

            for layer in layer_names:
                url_column = f"Download URL {layer}"
                if url_column not in df_sn.columns:
                    continue  # Skip if this layer column is not present

                urls = df_sn[url_column].dropna().tolist()
                if not urls:
                    continue  # Skip if no valid URLs for this layer on this date

                print(f"[INFO] Processing {short_name} - {layer} on {date}")
                print(f"Found {len(urls)} URLs")

                # Compile and load data
                if mode == "fire":
                    date_column = "Download URL VEG-DIST-DATE"
                    conf_column = "Download URL VEG-DIST-CONF"
                    date_layer_links = df_sn[date_column].dropna().tolist() if date_column in df_sn.columns else []
                    conf_layer_links = df_sn[conf_column].dropna().tolist() if conf_column in df_sn.columns else []
                    if not date_layer_links:
                        print(f"[WARN] No VEG-DIST-DATE URLs found for {short_name} on {date}")
                        date_DS = None
                    else:
                        print(f"[INFO] Found {len(date_layer_links)} VEG-DIST-DATE URLs")
                    if not conf_layer_links:
                        print(f"[WARN] No VEG-DIST-CONF URLs found for {short_name} on {date}")
                        conf_DS = None
                    else:
                        print(f"[INFO] Found {len(conf_layer_links)} CONF URLs")
                    DS, date_DS, conf_DS = compile_and_load_data(urls, mode,
                                                                 conf_layer_links=conf_layer_links,
                                                                 date_layer_links=date_layer_links)
                    try:
                        colormap = opera_mosaic.get_image_colormap(DS[0])
                    except Exception as e:
                        colormap = None

                    # Compute the date_threshold using either user-provided 'filter_date' or the date of the data acquistion
                    # and DIST reference date (12/31/2020)
                    if filter_date:
                        date_threshold = compute_date_threshold(filter_date)
                    else:
                        date_threshold = compute_date_threshold(str(date))

                    # Filter DIST layers by date and confidence
                    DS = filter_by_date_and_confidence(DS, date_DS, date_threshold, DS_conf=conf_DS, confidence_threshold=100, fill_value=None)
                
                if mode == "flood":
                    conf_column = "Download URL CONF"
                    conf_layer_links = df_sn[conf_column].dropna().tolist() if conf_column in df_sn.columns else []
                    if not conf_layer_links:
                        print(f"[WARN] No CONF URLs found for {short_name} on {date}")
                        conf_DS = None
                    else:
                        print(f"[INFO] Found {len(conf_layer_links)} CONF URLs")

                    DS, conf_DS = compile_and_load_data(urls, mode, conf_layer_links=conf_layer_links)
                    try:
                        colormap = opera_mosaic.get_image_colormap(DS[0])
                    except Exception as e:
                        colormap = None
                
                    # Reclassify false snow/ice positives
                    DS = reclassify_snow_ice_as_water(DS, conf_DS)

                # Mosaic the datasets using the appropriate method/rule
                mosaic, _, nodata = opera_mosaic.mosaic_opera(DS, product=short_name, merge_args={})
                image = opera_mosaic.array_to_image(mosaic, colormap=colormap, nodata=nodata)

                # Create filename and full paths
                mosaic_name = f"{short_name}_{layer}_{date}_mosaic.tif"
                mosaic_path = data_dir / mosaic_name
                tmp_path = data_dir / f"tmp_{mosaic_name}"

                # Save the mosaic to a temporary GeoTIFF
                copy(image, tmp_path, driver="GTiff")

                # Reproject/compress using GDAL directly into the final GeoTIFF
                gdal.Warp(
                    mosaic_path,
                    tmp_path,
                    xRes=30,
                    yRes=30,
                    creationOptions=["COMPRESS=DEFLATE"],
                )

                # Convert to COG (writes back into mosaic_path)
                save_gtiff_as_cog(mosaic_path, mosaic_path)

                print(f"[INFO] Mosaic written as COG: {mosaic_path}")

                # Clean up tmp file
                if tmp_path.exists():
                    tmp_path.unlink()

                # Add info to the mosiac index for pair-wise differencing
                with rasterio.open(mosaic_path) as ds:
                    mosaic_crs = ds.crs

                mosaic_index[short_name][layer][str(date)] = {
                    "path": mosaic_path,
                    "crs": mosaic_crs
                }

                # Make a map with PyGMT
                map_name = make_map(maps_dir, mosaic_path, short_name, layer, date)

                # Make a layout with matplotlib
                make_layout(layouts_dir, map_name, short_name, layer, date, layout_title)
        
    # Pair-wise differencing for 'fire' and 'flood' modes
    if mode in ("flood", "fire"):
        print("[INFO] Computing pairwise differences...")
        skipped = []  # collect CRS/UTM mismatches

        for short_name_k, layers_dict in mosaic_index.items():
            for layer_k, date_map in layers_dict.items():
                # all unique dates for which mosaics exist for this (short_name, layer)
                dates = sorted(date_map.keys())  

                # Generate difference for all possible pairs
                for i in range(len(dates)):
                    for j in range(i + 1, len(dates)):
                        d_early = dates[i]
                        d_later = dates[j]

                        early_info = date_map[d_early]
                        later_info = date_map[d_later]

                        # UTM/CRS check (skip the pair if different or unknown UTM zones)
                        crs_a = early_info["crs"]
                        crs_b = later_info["crs"]
                        if not same_utm_zone(crs_a, crs_b):
                            skipped.append({
                                "short_name": short_name_k,
                                "layer": layer_k,
                                "date_earlier": d_early,
                                "date_later": d_later,
                                "crs_earlier": crs_a.to_string() if crs_a else None,
                                "crs_later": crs_b.to_string() if crs_b else None,
                                "reason": "different or unknown UTM zone"
                            })
                            continue

                        # Name and path: {short}_{layer}_{LATER}_minus_{EARLIER}_diff.tif
                        diff_name = f"{short_name_k}_{layer_k}_{d_later}_minus_{d_early}_diff.tif"
                        diff_path = (mode_dir / "data") / diff_name

                        try:
                            compute_and_write_difference(
                                earlier_path=early_info["path"],
                                later_path=later_info["path"],
                                out_path=diff_path,
                                nodata_value=None
                            )
                            print(f"[INFO] Wrote diff COG: {diff_path}")
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

def make_map(maps_dir, mosaic_path, short_name, layer, date):
    """
    Create a map using PyGMT from the provided mosaic path.
    Args:
        maps_dir (Path): Directory where the map will be saved.
        mosaic_path (Path): Path to the mosaic file.
        short_name (str): Short name of the product.
        layer (str): Layer name to be used in the map.
        date (str): Date string in the format YYYY-MM-DD.
    Returns:
        map_name (Path): Path to the saved map image.
    Raises:
        ImportError: If required libraries are not installed.
    """
    import pygmt
    import rioxarray
    from pyproj import Geod
    import math

    mosaic_wgs84 = Path(str(mosaic_path).replace(".tif", "_WGS84.tif"))

    gdal.Warp(
        mosaic_wgs84,
        mosaic_path,
        dstSRS='EPSG:4326',
        resampleAlg='near',
        creationOptions=['COMPRESS=DEFLATE']
    )

    # === Load Mosaic ===
    grd = rioxarray.open_rasterio(mosaic_wgs84).squeeze()
    grd = grd.where(grd != 255)  # Remove nodata values

    # === Region ===
    bounds = grd.rio.bounds()  # xmin, ymin, xmax, ymax
    region = [bounds[0], bounds[2], bounds[1], bounds[3]]  # [xmin, xmax, ymin, ymax]

    # === Define Target Aspect Ratio ===
    # To match matplotlib layout: extent=[0, 60, 0, 100] → aspect ratio = width / height
    target_aspect = 60 / 100

    # Pad region to match target aspect ratio
    region_padded = expand_region_to_aspect(region, target_aspect)

    # === Define Projection ===
    center_lon = (region_padded[0] + region_padded[1]) / 2
    projection_width_cm = 15  # can be any fixed value; output height adjusts
    projection = f"M{center_lon}/{projection_width_cm}c"

    # === Create PyGMT Figure ===
    fig = pygmt.Figure()

    # Base coast layer (optional)
    fig.coast(
        region=region_padded,
        projection=projection,
        borders="2/thin",
        shorelines="thin",
        land="grey",
        water="lightblue",
        frame="a",
    )

    # === Add Grid Image ===
    if short_name == 'OPERA_L3_DSWX-HLS_V1' and layer == 'WTR':
        color_palette = 'palettes/DSWx-HLS_WTR.cpt'
        fig.grdimage(
            grid=grd,
            region=region_padded,
            projection=projection,
            cmap=color_palette,
            frame=["WSne", 'xaf', 'yaf'],
            nan_transparent=True
        )
        fig.colorbar(
            cmap=color_palette,
            equalsize=1.5,
        )

    elif short_name == 'OPERA_L3_DSWX-HLS_V1' and layer == 'BWTR':
        color_palette = 'palettes/DSWx-HLS_BWTR.cpt'
        fig.grdimage(
            grid=grd,
            region=region_padded,
            projection=projection,
            cmap=color_palette,
            frame=["WSne", 'xaf', 'yaf'],
            nan_transparent=True
        )
        fig.colorbar(
            cmap=color_palette,
            equalsize=1.5,
        )

    elif short_name == 'OPERA_L3_DSWX-S1_V1' and layer == 'WTR':
        color_palette = 'palettes/DSWx-S1_WTR.cpt'
        fig.grdimage(
            grid=grd,
            region=region_padded,
            projection=projection,
            cmap=color_palette,
            frame=["WSne", 'xaf', 'yaf'],
            nan_transparent=True
        )
        fig.colorbar(
            cmap=color_palette,
            equalsize=1.5,
        )

    elif short_name == 'OPERA_L3_DSWX-S1_V1' and layer == 'BWTR':
        color_palette = 'palettes/DSWx-S1_BWTR.cpt'
        fig.grdimage(
            grid=grd,
            region=region_padded,
            projection=projection,
            cmap=color_palette,
            frame=["WSne", 'xaf', 'yaf'],
            nan_transparent=True
        )
        fig.colorbar(
            cmap=color_palette,
            equalsize=1.5,
        )

    elif layer == "VEG-ANOM-MAX":
        color_palette = 'palettes/VEG-ANOM-MAX.cpt'
        fig.grdimage(
            grid=grd,
            region=region_padded,
            projection=projection,
            cmap=color_palette,
            frame=["WSne", 'xaf', 'yaf'],
            nan_transparent=True
        )
        fig.colorbar(
            cmap=color_palette,
            frame="xaf+lVEG-ANOM-MAX(%)",
        )

    elif layer == "VEG-DIST-STATUS":
        color_palette = 'palettes/VEG-DIST-STATUS.cpt'
        fig.grdimage(
            grid=grd,
            region=region_padded,
            projection=projection,
            cmap=color_palette,
            frame=["WSne", 'xaf', 'yaf'],
            nan_transparent=True
        )
        fig.colorbar(
            cmap=color_palette,
            equalsize=1.5,
        )

    # === Add Scale Bar and Compass ===
    xmin, xmax, ymin, ymax = region_padded
    center_lat = (ymin + ymax) / 2
    geod = Geod(ellps="WGS84")
    _, _, distance_m = geod.inv(xmin, center_lat, xmax, center_lat)

    # Set scalebar to ~25% of region width
    raw_length_km = distance_m * 0.25 / 1000  # 25% of map width in km
    exponent = math.floor(math.log10(raw_length_km))
    base = 10 ** exponent

    for factor in [1, 2, 5, 10]:
        scalebar_length_km = base * factor
        if scalebar_length_km >= raw_length_km:
            break
        
    fig.basemap(map_scale=f"jBR+w{scalebar_length_km:.0f}k+o0.5c/0.5c")
    fig.basemap(rose="jTR+o0.5c/0.5c+w1.5c")

    bounds = grd.rio.bounds()
    region = [bounds[0], bounds[2], bounds[1], bounds[3]]
    region_expanded = expand_region(region, width_deg=25, height_deg=15)

    # === Add Inset Map ===
    with fig.inset(
        position="jBL+o0.2c/0.2c",
        box="+pblack",
        region=region_expanded,
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
            [xmin, ymin],  # Close the loop
        ]

        # Plot the rectangle on the inset
        fig.plot(
            x=[pt[0] for pt in rectangle],
            y=[pt[1] for pt in rectangle],
            pen="2p,red"
        )

    # # === Export Map with Consistent Aspect Ratio ===
    map_name = maps_dir / f"{short_name}_{layer}_{date}_map.png"
    fig.savefig(map_name, dpi=900)

    return map_name

def make_layout(layout_dir, map_name, short_name, layer, date, layout_title):
    """
    Create a layout using matplotlib for the provided map.
    Args:
        layout_dir (Path): Directory where the layout will be saved.
        map_name (Path): Path to the map image.
        short_name (str): Short name of the product.
        layer (str): Layer name to be used in the layout.
        date (str): Date string in the format YYYY-MM-DD.
        layout_title (str): Title for the layout.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.image as mpimg
    import textwrap
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from matplotlib.image import imread


    # Create blank figure
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_axis_off()

    # Set background
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # === Add Main Map ===
    map_img = mpimg.imread(map_name)
    ax.imshow(map_img, extent=[0, 60, 0, 100])  # Main map on left 60% of layout

    # === Add Logos ===
    logo_opera = imread("logos/OPERA_logo.png")
    logo_new = imread("logos/ARIA_logo.png") 

    # Create a new axes for logos in the bottom-right corner
    logo_ax = fig.add_axes([0.82, 0.02, 0.06, 0.08], anchor='SE', zorder=10)
    logo_ax.imshow(logo_opera)
    logo_ax.axis('off')

    logo_ax2 = fig.add_axes([0.89, 0.02, 0.06, 0.08], anchor='SE', zorder=10)
    logo_ax2.imshow(logo_new)
    logo_ax2.axis('off')

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
        map_information = (
            f"The ARIA/OPERA water extent map is derived from an OPERA DSWx-HLS mosaicked " 
            f"product from Harmonized Landsat and Sentinel-2 data."
            f"This map depicts regions of vegetation disturbance."
        )
        data_source = "Copernicus Harmonized Landsat and Sentinel-2"
        
    elif short_name == "OPERA_L3_DIST-ALERT-S1_V1":
        subtitle = "OPERA Surface Disturbance Alert from Sentinel-1 (DIST-ALERT-S1)"
        map_information = (
            f"The ARIA/OPERA surface disturbance alert map is derived from an OPERA DIST-ALERT-S1 mosaicked "
            f"product from Copernicus Sentinel-1 data."
            f"This map depicts regions of surface disturbance."
        )
        data_source = "Copernicus Sentinel-1"

    elif short_name == "OPERA_L3_DIST-ALERT-HLS_V1":
            subtitle = "OPERA Surface Disturbance Alert from Harmonized Landsat and Sentinel-2 (DIST-ALERT-HLS)"
            map_information = (
            f"The ARIA/OPERA surface disturbance alert map is derived from an OPERA DIST-ALERT-HLS mosaicked "
            f"product from Harmonized Landsat and Sentinel-2 data."
            f"This map depicts regions of vegetation disturbance."
            )
            data_source = "Copernicus Harmonized Landsat and Sentinel-2"

    acquisitions = f"{date}"

    data_sources = textwrap.dedent(f"""\
        Product: {short_name}

        Layer: {layer}

        Data Source: {data_source}

        Resolution: 30 meters
    """)

    data_availability = textwrap.dedent(f"""\
        This product is available at: https://aria-share.jpl.nasa.gov/

        Visit the OPERA website: https://www.jpl.nasa.gov/go/opera/
    """)


    disclaimer = ("The results posted here are preliminary and unvalidated, "
                "intended to aid field response and provide a first look at the disaster-affected region.")


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

    # === Title ===
    ax.text(x_pos, y_start, title_wrp,
            fontsize=14, weight='bold',
            ha='left', va='top', transform=ax.transAxes)

    # === Subtitle ===
    ax.text(x_pos, y_start - line_spacing * 1, subtitle_wrp,
            fontsize=8, fontweight='bold', ha='left', va='top', transform=ax.transAxes)

    # === Acquisition Heading ===
    ax.text(x_pos, y_start - line_spacing * 3.5, "Data Acquisitions:",
            fontsize=8, fontweight='bold', ha='left', va='top', transform=ax.transAxes)

    # === Acquisition Dates ===
    ax.text(x_pos, y_start - line_spacing * 4, acquisitions,
            fontsize=8, ha='left', va='top', transform=ax.transAxes)

    # === Map Information Heading ===
    ax.text(x_pos, y_start - line_spacing * 6, "Map Information:",
            fontsize=8, fontweight='bold', ha='left', va='top', transform=ax.transAxes)

    # === Map Information Text ===
    ax.text(x_pos, y_start - line_spacing * 6.5, map_information_wrp,
            fontsize=8, ha='left', va='top', transform=ax.transAxes)

    # === Data Sources Heading ===
    ax.text(x_pos, y_start - line_spacing * 10, "Data Sources:",
            fontsize=8, fontweight='bold', ha='left', va='top', transform=ax.transAxes)

    # === Data Sources Text ===
    ax.text(x_pos, y_start - line_spacing * 10.5, data_sources,
            fontsize=8, ha='left', va='top', transform=ax.transAxes,linespacing=1, wrap=True
    )
    # === Data Availability Heading ===
    ax.text(x_pos, y_start - line_spacing * 15, "Product Availability:",
            fontsize=8, fontweight='bold', ha='left', va='top', transform=ax.transAxes)

    # === Data Availability Text ===
    ax.text(x_pos, y_start - line_spacing * 15.5, data_availability,
            fontsize=8, ha='left', va='top', linespacing=1, transform=ax.transAxes, wrap=True)

    # === Disclaimer Heading ===
    ax.text(x_pos, y_start - line_spacing * 18, "Disclaimer:",
            fontsize=8, fontweight='bold', ha='left', va='top', transform=ax.transAxes)

    # === Disclaimer (bottom-left) ===
    ax.text(x_pos, y_start - line_spacing * 18.5, disclaimer_wrp,
            fontsize=8, ha='left', va='top', transform=ax.transAxes)

    plt.tight_layout()

    layout_name = layout_dir / f"{short_name}_{layer}_{date}_layout.pdf"
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
        "RESAMPLING=AVERAGE"
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
        print("Earthquake mode coming soon. Exiting...")
        return

    output_dir = next_pass.run_next_pass(
        bbox=args.bbox,
        number_of_dates=args.number_of_dates,
        date=args.date,
        functionality=args.functionality
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
    generate_products(df_opera, args.mode, mode_dir, args.layout_title, args.filter_date)

    return

if __name__ == "__main__":
    main()