import concurrent.futures
import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pyproj
import rasterio
import rioxarray
import xarray as xr
from rasterio.transform import Affine
from rioxarray.merge import merge_arrays
from opera_utils.disp._remote import open_file

from .auth import authenticate

logger = logging.getLogger(__name__)


def get_master_crs(df_opera: pd.DataFrame, mode: str) -> Optional[str]:
    """
    Calculates the most common UTM CRS from the metadata WKT geometries.
    """
    logger.info("Determining Global Master CRS from geometric metadata...")
    from shapely import wkt

    valid_geoms = pd.Series(dtype=object)
    if "Geometry (WKT)" in df_opera.columns:
        # Grab all valid WKT geometries from the pre-filtered dataframe
        valid_geoms = df_opera["Geometry (WKT)"].dropna()
        valid_geoms = valid_geoms[valid_geoms != "N/A"]

    epsg_counter = Counter()

    for geom_str in valid_geoms:
        try:
            geom = wkt.loads(geom_str)
            center_lon = geom.centroid.x
            center_lat = geom.centroid.y

            # Geometrically calculate the UTM zone number
            zone_number = int((center_lon + 180) / 6) + 1
            is_northern = center_lat >= 0

            # Map to standard EPSG codes for UTM
            epsg = 32600 + zone_number if is_northern else 32700 + zone_number
            epsg_counter[epsg] += 1
        except Exception:
            continue

    if not epsg_counter:
        logger.info("Falling back to raster header CRS detection.")
        url_cols = [c for c in df_opera.columns if c.startswith("Download URL")]
        all_files = []
        for col in url_cols:
            all_files.extend(df_opera[col].dropna().tolist())

        crs_counter = Counter()
        for filepath in set(all_files):
            try:
                with rasterio.open(filepath) as src:
                    if src.crs is not None:
                        crs_counter[src.crs.to_epsg() or src.crs.to_string()] += 1
            except Exception:
                continue

        if not crs_counter:
            raise RuntimeError(
                "Could not determine CRS: no valid geometries or readable raster CRS metadata found."
            )

        most_common_crs, count = crs_counter.most_common(1)[0]
        crs_obj = pyproj.CRS.from_user_input(most_common_crs)
        logger.info(
            f"Global Master CRS determined from raster headers: {crs_obj.name} ({crs_obj.to_string()})"
        )
        logger.info(f"Found in {count}/{len(set(all_files))} granules.")
        return crs_obj.to_string()

    most_common_epsg, count = epsg_counter.most_common(1)[0]

    crs_obj = pyproj.CRS.from_epsg(most_common_epsg)
    utm_name = crs_obj.name

    logger.info(f"Global Master CRS determined: {utm_name} (EPSG:{most_common_epsg})")
    logger.info(f"Found in {count}/{len(valid_geoms)} granules.")

    return f"EPSG:{most_common_epsg}"


def get_master_grid_props(bbox_latlon: list, target_crs_proj4: str, target_res: int = 30) -> dict:
    """
    Defines a master pixel-aligned grid based on a lat/lon BBOX and target CRS.

    Args:
        bbox_latlon (list): Bounding box [S, N, W, E] in EPSG:4326.
        target_crs_proj4 (str): The PROJ4 string for the target master CRS.
        target_res (int): The target resolution in meters.

    Returns:
        dict: A dictionary with 'dst_crs', 'shape', 'transform' for rioxarray.reproject.
    """
    # Define transformers
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", target_crs_proj4, always_xy=True
    )
    
    # Get corners in target CRS
    corners_lon = [bbox_latlon[2], bbox_latlon[3], bbox_latlon[3], bbox_latlon[2]]
    corners_lat = [bbox_latlon[0], bbox_latlon[0], bbox_latlon[1], bbox_latlon[1]]
    
    xs, ys = transformer.transform(corners_lon, corners_lat)

    # Find min/max of transformed coordinates
    xmin = min(xs)
    ymin = min(ys)
    xmax = max(xs)
    ymax = max(ys)

    # Snap extent to be pixel-aligned to the resolution, ensuring any grid defined this way will be aligned.
    xmin = np.floor(xmin / target_res) * target_res
    ymin = np.floor(ymin / target_res) * target_res
    xmax = np.ceil(xmax / target_res) * target_res
    ymax = np.ceil(ymax / target_res) * target_res

    # Calculate final width and height in pixels
    width = int((xmax - xmin) / target_res)
    height = int((ymax - ymin) / target_res)

    # Create the GDAL/Rasterio Affine transform
    transform = Affine.translation(xmin, ymax) * Affine.scale(target_res, -target_res)

    return {
        "dst_crs": target_crs_proj4,
        "shape": (height, width),
        "transform": transform,
    }


def compile_and_load_data(data_layer_links, mode, conf_layer_links=None, date_layer_links=None, benchmark_stats=None, username=None, password=None):
    """
    Compile and load data from the provided layer links for mosaicking using multithreading.
    
    Args:
        data_layer_links (list): List of URLs corresponding to the OPERA data layers to mosaic.
        mode (str): Mode of operation, e.g., "flood", "fire", "landslide", "earthquake".
        conf_layer_links (list, optional): List of URLs for additional layers to filter false positives.
        date_layer_links (list, optional): List of URLs for date layers to filter by date.
        benchmark_stats (dict, optional): Mutable dictionary to track benchmarking stats. 
                                          If provided, enables benchmark mode.
        username (str, optional): Earthdata username.
        password (str, optional): Earthdata password.

    Returns:
        list or tuple: List of rioxarray datasets loaded from the provided links (in granule order).
                       May also return conf_DS and date_DS if applicable.
    """
    # If the first link exists as a local path, assume all are local and skip auth.
    is_local = False
    if data_layer_links and Path(data_layer_links[0]).exists():
        is_local = True
        logger.info("Local files detected. Skipping Earthdata authentication.")
    else:
        # If credentials weren't passed, authenticate (fallback)
        if not username or not password:
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
        logger.info(
            f"Most common satellite type: {most_common_satellite}, keeping only those links."
        )

        # Create a boolean mask to filter all lists consistently
        is_most_common = [most_common_satellite in link for link in data_layer_links]

        data_layer_links = [
            link for i, link in enumerate(data_layer_links) if is_most_common[i]
        ]

        # Filter auxiliary links consistently if they exist
        if conf_layer_links:
            conf_layer_links = [link for i, link in enumerate(conf_layer_links) if is_most_common[i]]
        if date_layer_links:
            date_layer_links = [link for i, link in enumerate(date_layer_links) if is_most_common[i]]

    # Define helpers for loading data
    def _load_single(link):
        """Helper to load a single dataset, handling auth fallback."""
        try:
            return rioxarray.open_rasterio(link, masked=False)
        except Exception:
            f = open_file(link, earthdata_username=username, earthdata_password=password)
            return rioxarray.open_rasterio(f, masked=False)

    def _run_sequential(links):
        """Sequential loading for benchmarking."""
        return [_load_single(link) for link in links]

    def _run_concurrent(links):
        """Concurrent loading using ThreadPoolExecutor."""
        if not links:
            return []
        max_workers = min(20, len(links))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(_load_single, links))

    def load_datasets(links, label="Dataset"):
        """Orchestrates loading. If benchmark_stats is set, runs both and tracks cumulative stats."""
        if not links:
            return []
            
        if benchmark_stats is not None:
            print(f"\n[BENCHMARK] Testing load speeds for {len(links)} items ({label})...")
            
            # Run Sequential
            t0 = time.time()
            _ = _run_sequential(links)
            t_seq = time.time() - t0
            
            # Run Concurrent
            t0 = time.time()
            results = _run_concurrent(links)
            t_conc = time.time() - t0
            
            # Update Globals - Using the 'loading' key
            if 'loading' in benchmark_stats:
                benchmark_stats['loading']['seq'] += t_seq
                benchmark_stats['loading']['conc'] += t_conc
                
                # Calculate metrics for printout
                cum_seq = benchmark_stats['loading']['seq']
                cum_conc = benchmark_stats['loading']['conc']
                cum_saved = cum_seq - cum_conc
                cum_speedup = cum_seq / cum_conc if cum_conc > 0 else 0
            else:
                # Fallback if structure is simple
                cum_saved = 0
                cum_speedup = 0
            
            # Calculate local metrics
            speedup = t_seq / t_conc if t_conc > 0 else 0
            saved = t_seq - t_conc
            
            print(f"   - Sequential: {t_seq:.2f}s")
            print(f"   - Concurrent: {t_conc:.2f}s")
            print(f"   >>> SPEEDUP: {speedup:.2f}x (Saved {saved:.2f}s)")
            if 'loading' in benchmark_stats:
                print(f"   [CUMULATIVE] Total Saved: {cum_saved:.2f}s | Global Speedup: {cum_speedup:.2f}x")
            print("-" * 50)
            
            return results
        else:
            # Standard fast path (concurrent only)
            logger.info(f"Loading {len(links)} '{label}' granules concurrently...")
            return _run_concurrent(links)
    
    # Load the primary data layer (DS)
    DS = load_datasets(data_layer_links, label="Primary Data")

    # If conf_layer_links AND mode == 'flood'
    if conf_layer_links and mode == "flood":
        conf_DS = load_datasets(conf_layer_links, label="Confidence")
        return DS, conf_DS

    # If conf_layer_links AND date_layer_links AND mode == 'fire' or 'landslide'
    if (conf_layer_links and date_layer_links and (mode == "fire" or mode == "landslide")):
        date_DS = load_datasets(date_layer_links, label="Date")
        conf_DS = load_datasets(conf_layer_links, label="Confidence")
        return DS, date_DS, conf_DS
    else:
        return DS


def mosaic_opera(DS: list, product: str = "OPERA_L3_DSWX-S1_V1", merge_args: dict = {}) -> Tuple[xr.DataArray, Optional[dict], float]:
    """
    Mosaics a list of OPERA product granules into a single image (in memory).

    Args:
        DS (list): A list of OPERA product granules opened as xarray.DataArray objects.
        product (str): OPERA product short name. Used to define pixel prioritization scheme in regions of OPERA granule overlap.
            Options include: "OPERA_L3_DSWX-HLS_V1","OPERA_L3_DSWX-S1_V1", "OPERA_L3_DIST-ALERT-HLS_V1", "OPERA_L3_DIST-ANN-HLS_V1", "OPERA_L2_RTC-S1_V1"
            Default: "OPERA_L3_DSWX-S1_V1"
        merge_args (dict, optional): A dictionary of arguments to pass to the rioxarray.merge_arrays function. Defaults to {}.

    Returns:
        da_mosaic: An xarray.DataArray containing the mosaic of the individual OPERA product granule DataArrays.
        colormap: A colormap for the mosaic, if in the original OPERA metadata, otherwise None.
        nodata: The nodata value for the mosaic corresponding to the original OPERA product granule metadata.
    """
    DA = []
    for ds in DS:
        nodata = ds.rio.nodata
        da = ds.fillna(nodata)
        DA.append(da)

    # Define 'valid' values for each product type
    if product.startswith("OPERA_L3_DSWX"):
        priority = {
            1: 100,
            2: 95,
            3: 90,
            0: 50,
            250: 20,
            251: 15,
            252: 10,
            253: 5,
            254: 1,
            255: 0,
        }
    elif product.startswith("OPERA_L3_DIST"):
        priority = {
            1: 100,
            2: 100,
            3: 100,
            4: 100,
            5: 100,
            6: 100,
            7: 100,
            8: 100,
            9: 100,
            10: 100,
            0: 10,
            255: 0,
        }
    elif product.startswith("OPERA_L2_RTC"):
        priority = {}
    else:
        priority = {}

    valid_values = set(priority.keys())

    # Check if any DataArray contains non-valid values, if so fall back to defaul rasterio.merge method
    if contains_unexpected_values(DA, valid_values):
        method = "first"
    elif product.startswith("OPERA_L3_DIST") or product.startswith("OPERA_L2_RTC"):
        method = "first"
    else:
        method = opera_rules(product=product, nodata=nodata)

    merged_arr = merge_arrays(DA, method=method)

    try:
        colormap = get_image_colormap(DS[0])
    except Exception as e:
        colormap = None
    return merged_arr, colormap, nodata


def opera_rules(product: str = "OPERA_L3_DSWX-S1_V1", nodata: int = 255):
    """
    Returns a custom callabale rasterio.merge method for OPERA products using pixel priority rules.
    
    Args:
        product (str): OPERA product short name, used to determine pixel prioritization in regions of OPERA granule overlap.
            Options include: "OPERA_L3_DSWX-HLS_V1","OPERA_L3_DSWX-S1_V1", "OPERA_L3_DIST-ALERT-HLS_V1", "OPERA_L3_DIST-ANN-HLS_V1", "OPERA_L2_RTC-S1_V1"
            Default: "OPERA_L3_DSWX-S1_V1"
        nodata (int): The nodata value for the OPERA product. Default is 255.
        
    Returns:
        method (function): A function that implements the custom merge method for the specified OPERA product.
    """

    if product in ("OPERA_L3_DSWX-HLS_V1", "OPERA_L3_DSWX-S1_V1"):
        priority = {
            1: 100,  # Open water (DSWx-HLS, DSWx-S1)
            2: 95,  # Partial surface water (DSWx-HLS)
            3: 95,  # Inundated vegetation (DSWx-S1)
            0: 50,  # Not water (DSWx-HLS, DSWx-S1)
            250: 20,  # Height Above Nearest Drainage (HAND) masked (DSWx-S1)
            251: 15,  # Layover/shadow masked (DSWx-S1)
            252: 10,  # Snow/Ice (DSWx-HLS)
            253: 5,  # Cloud/Cloud shadow (DSWx-HLS)
            254: 1,  # Ocean masked (DSWx-HLS)
            255: 0,  # Fill value (no data) (DSWx-HLS, DSWx-S1)
        }
    elif product in ("OPERA_L3_DIST-ALERT-HLS_V1", "OPERA_L3_DIST-ANN-HLS_V1"):
        priority = {
            1: 100,  # first <50%
            2: 100,  # provisional <50%
            3: 100,  # confirmed <50%
            4: 100,  # first ≥50%
            5: 100,  # provisional ≥50%
            6: 100,  # confirmed ≥50%
            7: 100,  # confirmed <50%, finished
            8: 100,  # confirmed ≥50%, finished
            9: 100,  # confirmed previous year <50%
            10: 100,  # confirmed previous year ≥50%
            0: 10,  # No disturbance
            255: 0,  # No data
        }
    elif product == "OPERA_L2_RTC-S1_V1":
        priority = {
            1: 100,  # Open water (DSWx-HLS, DSWx-S1)
            2: 95,  # Partial surface water (DSWx-HLS)
            3: 90,  # Inundated vegetation (DSWx-S1)
            0: 50,  # Not water (DSWx-HLS, DSWx-S1)
            250: 20,  # Height Above Nearest Drainage (HAND) masked (DSWx-S1)
            251: 15,  # Layover/shadow masked (DSWx-S1)
            252: 10,  # Snow/Ice (DSWx-HLS)
            253: 5,  # Cloud/Cloud shadow (DSWx-HLS)
            254: 1,  # Ocean masked (DSWx-HLS)
            255: 0,  # Fill value (no data) (DSWx-HLS, DSWx-S1)
        }

    else:
        raise ValueError(
            f"Unknown product type: {product}. Supported products are DSWx, DIST, RTC."
        )

    def method(
        old_data,
        new_data,
        old_nodata=None,
        new_nodata=None,
        index=None,
        roff=None,
        coff=None,
    ):
        """
        Custom merge method for OPERA products using pixel priority rules.

        Args:
            old_data (numpy.ndarray): The existing data array.
            new_data (numpy.ndarray): The new data array to merge.
            old_nodata (int, optional): The nodata value for the existing data. Defaults to None. Required by rasterio.merge.
            new_nodata (int, optional): The nodata value for the new data. Defaults to None. Required by rasterio.merge.
            index (tuple, optional): The index of the pixel being merged. Defaults to None. Required by rasterio.merge.
            roff (int, optional): Row offset. Defaults to None. Required by rasterio.merge.
            coff (int, optional): Column offset. Defaults to None. Required by rasterio.merge.

        Returns:
            numpy.ndarray: The merged data array.
        """
        max_val = max(priority.keys()) + 1
        priority_array = np.full(max_val, -1, dtype=np.int16)
        for val, pri in priority.items():
            priority_array[val] = pri

        valid_mask = new_data[0] != nodata
        
        new_vals_safe = np.clip(new_data[0], 0, max_val - 1)
        old_vals_safe = np.clip(old_data[0], 0, max_val - 1)

        new_priorities = priority_array[new_vals_safe]
        old_priorities = priority_array[old_vals_safe]

        update_mask = (valid_mask) & (new_priorities > old_priorities)

        # Apply the update
        for i in range(old_data.shape[0]):
            old_data[i][update_mask] = new_data[i][update_mask]

        return old_data

    return method


def contains_unexpected_values(DA: list, valid_values: set) -> bool:
    """
    Check if any DataArray contains non-valid values.

    Args:
        DA (list): List of DataArrays.
        valid_values (set): Set of expected valid pixel values.

    Returns:
        bool: True if unexpected values are found, False otherwise.
    """
    for da in DA:
        unique_vals = np.unique(da.values)
        if not set(unique_vals).issubset(valid_values):
            return True
    return False


def get_image_colormap(image, index: int = 1) -> Optional[dict]:
    """
    Retrieve the colormap from an image.

    Args:
        image (str, rasterio.io.DatasetReader, rioxarray.DataArray):
            The input image. It can be:
            - A file path to a raster image (string).
            - A rasterio dataset.
            - A rioxarray DataArray.
        index (int): The band index to retrieve the colormap from (default is 1).

    Returns:
        dict: A dictionary representing the colormap (value: (R, G, B, A)), or None if no colormap is found.

    Raises:
        ValueError: If the input image type is unsupported.
    """
    dataset = None

    if isinstance(image, str):  # File path
        with rasterio.open(image) as ds:
            return ds.colormap(index) if ds.count > 0 else None
    elif isinstance(image, rasterio.io.DatasetReader):  # rasterio dataset
        dataset = image
    elif isinstance(image, xr.DataArray) or isinstance(image, xr.Dataset):
        source = image.encoding.get("source")
        if source:
            with rasterio.open(source) as ds:
                return ds.colormap(index) if ds.count > 0 else None
        else:
            raise ValueError(
                "Cannot extract colormap: DataArray does not have a source."
            )
    else:
        raise ValueError(
            "Unsupported input type. Provide a file path, rasterio dataset, or rioxarray DataArray."
        )

    if dataset:
        return dataset.colormap(index) if dataset.count > 0 else None


def array_to_memory_file(
    array,
    source: str = None,
    dtype: str = None,
    compress: str = "deflate",
    transpose: bool = True,
    cellsize: float = None,
    crs: str = None,
    transform: tuple = None,
    driver="COG",
    colormap: dict = None,
    **kwargs,
):
    """
    Convert a NumPy array to a memory file.

    Args:
        array (numpy.ndarray): The input NumPy array.
        source (str, optional): Path to the source file to extract metadata from. Defaults to None.
        dtype (str, optional): The desired data type of the array. Defaults to None.
        compress (str, optional): The compression method for the output file. Defaults to "deflate".
        transpose (bool, optional): Whether to transpose the array from (bands, rows, columns) to (rows, columns, bands). Defaults to True.
        cellsize (float, optional): The cell size of the array if source is not provided. Defaults to None.
        crs (str, optional): The coordinate reference system of the array if source is not provided. Defaults to None.
        transform (tuple, optional): The affine transformation matrix if source is not provided.
            Can be rio.transform() or a tuple like (0.5, 0.0, -180.25, 0.0, -0.5, 83.780361). Defaults to None.
        driver (str, optional): The driver to use for creating the output file, such as 'GTiff'. Defaults to "COG".
        colormap (dict, optional): A dictionary defining the colormap (value: (R, G, B, A)).
        **kwargs: Additional keyword arguments to be passed to the rasterio.open() function.

    Returns:
        rasterio.DatasetReader: The rasterio dataset reader object for the converted array.
    """
    if isinstance(array, xr.DataArray):
        coords = [coord for coord in array.coords]
        if coords[0] == "time":
            x_dim = coords[1]
            y_dim = coords[2]
            array = (
                array.isel(time=0).rename({y_dim: "y", x_dim: "x"}).transpose("y", "x")
            )
        if hasattr(array, "rio"):
            if hasattr(array.rio, "crs"):
                if array.rio.crs is not None:
                    crs = array.rio.crs
            if transform is None and hasattr(array.rio, "transform"):
                transform = array.rio.transform()
        elif source is None:
            if hasattr(array, "encoding"):
                if "source" in array.encoding:
                    source = array.encoding["source"]
        array = array.values

    if array.ndim == 3 and transpose:
        array = np.transpose(array, (1, 2, 0))
    if source is not None:
        with rasterio.open(source) as src:
            crs = src.crs
            transform = src.transform
            if compress is None:
                compress = src.compression
    else:
        if crs is None:
            raise ValueError(
                "crs must be provided if source is not provided, such as EPSG:3857"
            )

        if transform is None:
            if cellsize is None:
                raise ValueError("cellsize must be provided if source is not provided")
            # Define the geotransformation parameters
            xmin, ymin, xmax, ymax = (
                0,
                0,
                cellsize * array.shape[1],
                cellsize * array.shape[0],
            )
            # (west, south, east, north, width, height)
            transform = rasterio.transform.from_bounds(
                xmin, ymin, xmax, ymax, array.shape[1], array.shape[0]
            )
        elif isinstance(transform, Affine):
            pass
        elif isinstance(transform, (tuple, list)):
            transform = Affine(*transform)

        kwargs["transform"] = transform

    if dtype is None:
        # Determine the minimum and maximum values in the array
        min_value = np.min(array)
        max_value = np.max(array)
        # Determine the best dtype for the array
        if min_value >= 0 and max_value <= 1:
            dtype = np.float32
        elif min_value >= 0 and max_value <= 255:
            dtype = np.uint8
        elif min_value >= -128 and max_value <= 127:
            dtype = np.int8
        elif min_value >= 0 and max_value <= 65535:
            dtype = np.uint16
        elif min_value >= -32768 and max_value <= 32767:
            dtype = np.int16
        else:
            dtype = np.float64

    # Convert the array to the best dtype
    array = array.astype(dtype)
    # Define the GeoTIFF metadata
    metadata = {
        "driver": driver,
        "height": array.shape[0],
        "width": array.shape[1],
        "dtype": array.dtype,
        "crs": crs,
        "transform": transform,
    }

    if array.ndim == 2:
        metadata["count"] = 1
    elif array.ndim == 3:
        metadata["count"] = array.shape[2]
    if compress is not None:
        metadata["compress"] = compress

    metadata.update(**kwargs)

    # Create a new memory file and write the array to it
    memory_file = rasterio.MemoryFile()
    dst = memory_file.open(**metadata)

    if array.ndim == 2:
        dst.write(array, 1)
        if colormap:
            dst.write_colormap(1, colormap)
    elif array.ndim == 3:
        for i in range(array.shape[2]):
            dst.write(array[:, :, i], i + 1)
            if colormap:
                dst.write_colormap(i + 1, colormap)

    dst.close()
    # Read the dataset from memory
    dataset_reader = rasterio.open(dst.name, mode="r")

    return dataset_reader


def array_to_image(
    array,
    output: str = None,
    source: str = None,
    dtype: str = None,
    compress: str = "deflate",
    transpose: bool = True,
    cellsize: float = None,
    crs: str = None,
    transform: tuple = None,
    driver: str = "COG",
    colormap: dict = None,
    **kwargs,
) -> str:
    """Save a NumPy array as a GeoTIFF using the projection information from an existing GeoTIFF file.

    Args:
        array (np.ndarray): The NumPy array to be saved as a GeoTIFF.
        output (str): The path to the output image. If None, a temporary file will be created. Defaults to None.
        source (str, optional): The path to an existing GeoTIFF file with map projection information. Defaults to None.
        dtype (np.dtype, optional): The data type of the output array. Defaults to None.
        compress (str, optional): The compression method. Can be one of the following: "deflate", "lzw", "packbits", "jpeg". Defaults to "deflate".
        transpose (bool, optional): Whether to transpose the array from (bands, rows, columns) to (rows, columns, bands). Defaults to True.
        cellsize (float, optional): The resolution of the output image in meters. Defaults to None.
        crs (str, optional): The CRS of the output image. Defaults to None.
        transform (tuple, optional): The affine transformation matrix, can be rio.transform() or a tuple like (0.5, 0.0, -180.25, 0.0, -0.5, 83.780361).
            Defaults to None.
        driver (str, optional): The driver to use for creating the output file, such as 'GTiff'. Defaults to "COG".
        colormap (dict, optional): A dictionary defining the colormap (value: (R, G, B, A)).
        **kwargs: Additional keyword arguments to be passed to the rasterio.open() function.
    """
    if output is None:
        return array_to_memory_file(
            array,
            source,
            dtype,
            compress,
            transpose,
            cellsize,
            crs=crs,
            transform=transform,
            driver=driver,
            colormap=colormap,
            **kwargs,
        )

    if isinstance(array, xr.DataArray):
        if (
            hasattr(array, "rio")
            and (array.rio.crs is not None)
            and (array.rio.transform() is not None)
        ):

            if "latitude" in array.dims and "longitude" in array.dims:
                array = array.rename({"latitude": "y", "longitude": "x"})
            elif "lat" in array.dims and "lon" in array.dims:
                array = array.rename({"lat": "y", "lon": "x"})

            if array.ndim == 2 and ("x" in array.dims) and ("y" in array.dims):
                array = array.transpose("y", "x")
            elif array.ndim == 3 and ("x" in array.dims) and ("y" in array.dims):
                dims = list(array.dims)
                dims.remove("x")
                dims.remove("y")
                array = array.transpose(dims[0], "y", "x")
            if "long_name" in array.attrs:
                array.attrs.pop("long_name")

            array.rio.to_raster(
                output, driver=driver, compress=compress, dtype=dtype, **kwargs
            )
            return output

    if array.ndim == 3 and transpose:
        array = np.transpose(array, (1, 2, 0))

    out_dir = os.path.dirname(os.path.abspath(output))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ext = os.path.splitext(output)[-1].lower()
    if ext == "":
        output += ".tif"
        driver = "COG"
    elif ext == ".png":
        driver = "PNG"
    elif ext == ".jpg" or ext == ".jpeg":
        driver = "JPEG"
    elif ext == ".jp2":
        driver = "JP2OpenJPEG"
    elif ext == ".tiff":
        driver = "GTiff"
    else:
        driver = "COG"

    if source is not None:
        with rasterio.open(source) as src:
            crs = src.crs
            transform = src.transform
            if compress is None:
                compress = src.compression
    else:
        if cellsize is None:
            raise ValueError("resolution must be provided if source is not provided")
        if crs is None:
            raise ValueError(
                "crs must be provided if source is not provided, such as EPSG:3857"
            )

        if transform is None:
            # Define the geotransformation parameters
            xmin, ymin, xmax, ymax = (
                0,
                0,
                cellsize * array.shape[1],
                cellsize * array.shape[0],
            )
            transform = rasterio.transform.from_bounds(
                xmin, ymin, xmax, ymax, array.shape[1], array.shape[0]
            )
        elif isinstance(transform, Affine):
            pass
        elif isinstance(transform, (tuple, list)):
            transform = Affine(*transform)

        kwargs["transform"] = transform

    if dtype is None:
        # Determine the minimum and maximum values in the array
        min_value = np.min(array)
        max_value = np.max(array)
        # Determine the best dtype for the array
        if min_value >= 0 and max_value <= 1:
            dtype = np.float32
        elif min_value >= 0 and max_value <= 255:
            dtype = np.uint8
        elif min_value >= -128 and max_value <= 127:
            dtype = np.int8
        elif min_value >= 0 and max_value <= 65535:
            dtype = np.uint16
        elif min_value >= -32768 and max_value <= 32767:
            dtype = np.int16
        else:
            dtype = np.float64

    # Convert the array to the best dtype
    array = array.astype(dtype)

    # Define the GeoTIFF metadata
    metadata = {
        "driver": driver,
        "height": array.shape[0],
        "width": array.shape[1],
        "dtype": array.dtype,
        "crs": crs,
        "transform": transform,
    }

    if array.ndim == 2:
        metadata["count"] = 1
    elif array.ndim == 3:
        metadata["count"] = array.shape[2]
    if compress is not None and (driver in ["GTiff", "COG"]):
        metadata["compress"] = compress

    metadata.update(**kwargs)
    # Create a new GeoTIFF file and write the array to it
    with rasterio.open(output, "w", **metadata) as dst:
        if array.ndim == 2:
            dst.write(array, 1)
            if colormap:
                dst.write_colormap(1, colormap)
        elif array.ndim == 3:
            for i in range(array.shape[2]):
                dst.write(array[:, :, i], i + 1)
                if colormap:
                    dst.write_colormap(i + 1, colormap)
    return output
