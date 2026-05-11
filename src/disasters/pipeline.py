from __future__ import annotations

# Standard Library Imports
import concurrent.futures
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# Third-Party Imports
import next_pass
import numpy as np
import pandas as pd
import pyproj
import rasterio
import xarray as xr
from osgeo import gdal
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

# Local/Relative Imports
from .auth import authenticate
from .catalog import cluster_by_time, read_opera_metadata, get_S1_orbit_direction
from .diff import (
    compute_and_write_difference,
    compute_and_write_difference_positive_change_only,
    create_rtc_rgb_visualization,
    save_gtiff_as_cog,
)
from .filters import (
    compute_date_threshold,
    filter_by_date_and_confidence,
    generate_coastal_mask,
    process_dem_and_slope,
    reclassify_snow_ice_as_water,
)
from .io import cleanup_temp_file, ensure_directory, scan_local_directory
from .layouts import make_layout, make_map
from .mosaic import (
    array_to_image,
    compile_and_load_data,
    get_image_colormap,
    get_master_crs,
    get_master_grid_props,
    mosaic_opera,
    warp_dataarray_to_grid,
)
logger = logging.getLogger(__name__)

gdal.DontUseExceptions()


class DeferredExecutor:
    """Collect submitted work and run it sequentially at shutdown."""

    def __init__(self):
        self._jobs = []

    def submit(self, fn, *args, **kwargs):
        future = concurrent.futures.Future()
        self._jobs.append((future, fn, args, kwargs))
        return future

    def shutdown(self, wait: bool = True):
        for future, fn, args, kwargs in self._jobs:
            if future.done():
                continue
            try:
                future.set_result(fn(*args, **kwargs))
            except Exception as exc:
                future.set_exception(exc)
        self._jobs.clear()
        return None

@dataclass
class PipelineConfig:
    """
    Configuration for running the OPERA disaster pipeline.
    """
    bbox: Sequence[float] | str
    output_dir: Path
    layout_title: str
    zoom_bbox: Sequence[float] | None = None
    local_dir: Path | None = None
    product: str | None = None
    layer_name: str | None = None
    date: str | None = None
    number_of_dates: int = 5
    mode: str | None = None
    functionality: str = "opera_search"
    filter_date: str | None = None
    reclassify_snow_ice: bool = False
    slope_threshold: int | None = None
    benchmark: bool = False
    no_mask: bool = False
    compute_cloudiness: bool = False
    skip_existing: bool = False


def run_pipeline(config: PipelineConfig) -> Path | None:
    """
    Run the end-to-end disaster pipeline (CLI-independent).

    Args:
        config (PipelineConfig): Configuration parameters for the pipeline execution.

    Returns:
        Path | None: The mode directory path (e.g., `<output_dir>/flood`) if the pipeline ran, 
                     or None if exited early (e.g., earthquake mode).
    """
    from datetime import datetime, timezone

    if config.mode == "earthquake":
        logger.info("Earthquake mode coming soon. Exiting...")
        return None

    if not config.local_dir:
        try:
            username, password = authenticate()
            logger.info("Authentication successful.")
        except Exception as e:
            logger.warning(f"Authentication failed: {e}")
            username, password = None, None
    else:
        username, password = None, None

    ensure_directory(config.output_dir)
    
    folder_name = config.mode if config.mode else config.product
    mode_dir = config.output_dir / folder_name

    if config.local_dir:
        logger.info(f"Running in LOCAL mode using data from: {config.local_dir}")
        if not config.local_dir.exists():
            logger.error(f"Local directory {config.local_dir} does not exist.")
            return None
        df_opera = scan_local_directory(config.local_dir)
        if df_opera.empty: return None
        ensure_directory(mode_dir)
    else:
        # Cloud Logic
        logger.info("Running in CLOUD SEARCH mode.")

        next_pass_bbox = [config.bbox] if isinstance(config.bbox, str) else config.bbox
        
        output_dir = next_pass.run_next_pass(
            bbox=next_pass_bbox,
            number_of_dates=config.number_of_dates,
            date=config.date,
            functionality=config.functionality,
            compute_cloudiness=config.compute_cloudiness
        )
        
        output_dir = Path(output_dir)
        dest = config.output_dir / output_dir.name
        
        if output_dir.resolve() != dest.resolve():
            if not dest.exists():
                output_dir.rename(dest)
                processing_dir = dest
            else:
                logger.warning(f"Destination {dest} already exists. Using existing folder.")
                processing_dir = dest
        else:
            processing_dir = output_dir
            
        # Read OPERA metadata returned by next_pass
        df_opera = read_opera_metadata(processing_dir)
        
        # Ensure mode directory exists
        ensure_directory(mode_dir)

    logger.info(f"Outputting results to: {mode_dir}")

    # Convert WKT/File to an SNWE list for internal mosaicking logic
    if isinstance(config.bbox, str):
        # Use next_pass parsers for both local and cloud modes
        try:
            from utils.utils import bbox_type, bbox_to_geometry
            bbox_parsed = bbox_type([config.bbox])
            geom, bounds, centroid = bbox_to_geometry(bbox_parsed, mode_dir)
            minx, miny, maxx, maxy = bounds
            internal_bbox = [miny, maxy, minx, maxx]
            logger.info(f"Extracted SNWE bounding envelope from geometry: {internal_bbox}")
        except Exception as e:
            logger.error(f"Failed to parse geometry from string/file (KML, GeoJSON, WKT): {e}")
            return None
    else:
        internal_bbox = list(config.bbox)

    # Set up benchmarking stats if requested
    benchmark_stats = None
    if config.benchmark:
        benchmark_stats = {
            'loading': {'seq': 0.0, 'conc': 0.0},
            'plotting': {'seq': 0.0, 'conc': 0.0},
            'differencing': {'seq': 0.0, 'conc': 0.0}
        }

    # Generate products
    generate_products(
        df_opera=df_opera,
        mode=config.mode,
        mode_dir=mode_dir,
        layout_title=config.layout_title,
        bbox=internal_bbox,
        zoom_bbox=list(config.zoom_bbox) if config.zoom_bbox is not None else None,
        filter_date=config.filter_date,
        reclassify_snow_ice=config.reclassify_snow_ice,
        slope_threshold=config.slope_threshold,
        benchmark_stats=benchmark_stats,
        username=username,
        password=password,
        no_mask=config.no_mask,
        skip_existing=config.skip_existing,
        product=config.product
    )

    if config.benchmark and benchmark_stats:
        print("\n" + "="*50)
        print("FINAL BENCHMARK REPORT")
        print("="*50)
        
        # Report benchmarking results for the 'loading' stage
        l_seq = benchmark_stats['loading']['seq']
        l_conc = benchmark_stats['loading']['conc']
        l_saved = l_seq - l_conc
        print(f"DATA LOADING:\n  Sequential: {l_seq:.2f}s | Concurrent: {l_conc:.2f}s\n  Saved:      {l_saved:.2f}s")
        
        # Report benchmarking results for the 'differencing' stage
        d_seq = benchmark_stats['differencing']['seq']
        d_conc = benchmark_stats['differencing']['conc']
        d_saved = d_seq - d_conc
        if d_seq > 0:
            print(f"DIFFERENCING (Backgrounded):\n  Sequential: {d_seq:.2f}s | Concurrent: ~0s (Overlapped)\n  Saved:      {d_saved:.2f}s")
        
        # Report benchmarking results for the 'plotting' stage
        p_seq = benchmark_stats['plotting']['seq']
        p_conc = benchmark_stats['plotting']['conc']
        p_saved = p_seq - p_conc
        print(f"PLOTTING (Backgrounded):\n  Sequential: {p_seq:.2f}s | Concurrent: ~0s (Overlapped)\n  Saved:      {p_saved:.2f}s")
        
        print("-" * 50)
        print(f"TOTAL TIME SAVED: {l_saved + d_saved + p_saved:.2f}s")
        print("="*50 + "\n")

    return mode_dir


def run_search_only(
    bbox: Sequence[float] | str,
    output_dir: Path,
    date: str | None = None,
    number_of_dates: int = 5,
    mode: str | None = None,
    product: str | None = None,
    functionality: str = "opera_search",
    compute_cloudiness: bool = False
) -> Path | None:
    """
    Runs next_pass to discover products and generate metadata without downloading imagery.

    Args:
        bbox (Sequence[float] | str): Bounding box in [S, N, W, E] format, WKT, or geojson path.
        output_dir (Path): Local directory to save the metadata file.
        date (str | None): Optional date string for filtering products.
        number_of_dates (int): Number of dates to retrieve if 'date' is specified.
        mode (str | None): If specified, filters the metadata summary to only include relevant datasets/layers for this mode.
        product (str | None): If specified, filters the metadata summary to only include this specific product.
        compute_cloudiness (bool): Whether to compute cloudiness metrics during next_pass search.
    """
    import shutil

    ensure_directory(output_dir)

    logger.info("Running Cloud Search to discover available granules...")
    next_pass_bbox = [bbox] if isinstance(bbox, str) else bbox

    # Run the next_pass engine
    output_dir_np = next_pass.run_next_pass(
        bbox=next_pass_bbox,
        number_of_dates=number_of_dates,
        date=date,
        functionality=functionality,
        compute_cloudiness=compute_cloudiness
    )

    output_dir_np = Path(output_dir_np)

    # Move the nextpass output folder into user-specified directory
    dest = output_dir / output_dir_np.name
    if output_dir_np.resolve() != dest.resolve():
        if dest.exists():
            logger.warning(f"Destination {dest} already exists. Replacing it with fresh search results.")
            shutil.rmtree(dest)
        shutil.move(str(output_dir_np), str(dest))
        output_dir_np = dest

    # Read the metadata just to provide a helpful summary to the user
    df_opera = read_opera_metadata(output_dir_np)
    if df_opera.empty:
        logger.warning("No products found for the specified criteria.")
        return output_dir_np

    # If mode or product is provided, calculate how many of those found granules actually apply
    if mode:
        if mode == "flood":
            short_names = ["OPERA_L3_DSWX-HLS_V1", "OPERA_L3_DSWX-S1_V1"]
        elif mode == "fire":
            short_names = ["OPERA_L3_DIST-ALERT-HLS_V1", "OPERA_L3_DIST-ALERT-S1_V1"]
        elif mode == "landslide":
            short_names = ["OPERA_L3_DIST-ALERT-HLS_V1", "OPERA_L2_RTC-S1_V1"]
        elif mode == "rtc-rgb":
            short_names = ["OPERA_L2_RTC-S1_V1"]
        elif mode == "earthquake":
            logger.info("Earthquake mode coming soon. Exiting...")
            return None
        else:
            short_names = []
            
        df_filtered = df_opera[df_opera["Dataset"].isin(short_names)]
        logger.info(f"Search complete. Found {len(df_filtered)} granules relevant to '{mode}' mode (out of {len(df_opera)} total).")
    
    elif product:
        short_names = [product]
        df_filtered = df_opera[df_opera["Dataset"].isin(short_names)]
        logger.info(f"Search complete. Found {len(df_filtered)} granules matching product '{product}' (out of {len(df_opera)} total).")
        
    else:
        logger.info(f"Search complete. Found {len(df_opera)} total OPERA granules.")

    return output_dir_np


def run_download_only(
    bbox: Sequence[float] | str, 
    output_dir: Path, 
    date: str | None = None, 
    number_of_dates: int = 5, 
    mode: str | None = None,
    product: str | None = None,
    functionality: str = "opera_search",
    compute_cloudiness: bool = False
) -> Path | None:
    """
    Runs next_pass to discover products and downloads the raw GeoTIFFs to a local directory.
    If 'mode' is specified, aggressively filters downloads to only include necessary datasets and auxiliary layers.

    Args:
        bbox (Sequence[float] | str): Bounding box in [S, N, W, E] format, WKT, or geojson path.
        output_dir (Path): Local directory to save downloaded files.
        date (str | None): Optional date string for filtering products.
        number_of_dates (int): Number of dates to retrieve if 'date' is specified.
        mode (str | None): If specified, filters downloads to only include relevant datasets/layers for this mode.
        product (str | None): If specified, filters downloads to only include this specific product.
        compute_cloudiness (bool): Whether to compute cloudiness metrics during next_pass search.
    """
    import shutil
    from opera_utils.disp._remote import open_file
    import concurrent.futures

    # Authenticate with Earthdata
    try:
        username, password = authenticate()
        logger.info("Authentication successful.")
    except Exception as e:
        logger.warning(f"Authentication failed: {e}")
        return None

    # Set up directories
    ensure_directory(output_dir)
    data_dir = ensure_directory(output_dir / "data")

    logger.info("Running Cloud Search to discover available granules...")
    next_pass_bbox = [bbox] if isinstance(bbox, str) else bbox
    
    # Run the next_pass engine
    output_dir_np = next_pass.run_next_pass(
        bbox=next_pass_bbox,
        number_of_dates=number_of_dates,
        date=date,
        functionality=functionality,
        compute_cloudiness=compute_cloudiness
    )
    
    output_dir_np = Path(output_dir_np)
    
    # Read the metadata
    df_opera = read_opera_metadata(output_dir_np)
    if df_opera.empty:
        logger.warning("No products found for the specified criteria.")
        return None

    # Apply Filtering if requested
    if mode:
        logger.info(f"Filtering downloads for '{mode}' mode...")
        
        # Define target datasets and primary + auxiliary layers
        if mode == "flood":
            short_names = ["OPERA_L3_DSWX-HLS_V1", "OPERA_L3_DSWX-S1_V1"]
            target_layers = ["WTR", "BWTR", "CONF"] 
        elif mode == "fire":
            short_names = ["OPERA_L3_DIST-ALERT-HLS_V1", "OPERA_L3_DIST-ALERT-S1_V1"]
            target_layers = ["VEG-ANOM-MAX", "VEG-DIST-STATUS", "VEG-DIST-DATE", "VEG-DIST-CONF"]
        elif mode == "landslide":
            short_names = ["OPERA_L3_DIST-ALERT-HLS_V1", "OPERA_L2_RTC-S1_V1"]
            target_layers = ["VEG-ANOM-MAX", "VEG-DIST-STATUS", "VEG-DIST-DATE", "VEG-DIST-CONF", "RTC-VV", "RTC-VH"]
        elif mode == "rtc-rgb":
            short_names = ["OPERA_L2_RTC-S1_V1"]
            target_layers = ["RTC-VV", "RTC-VH"]
        elif mode == "earthquake":
            logger.info("Earthquake mode coming soon. Exiting...")
            return None
            
        # Filter rows by Dataset
        df_opera = df_opera[df_opera["Dataset"].isin(short_names)]
        
        # Filter URL columns by Layer
        url_cols = [f"Download URL {layer}" for layer in target_layers if f"Download URL {layer}" in df_opera.columns]
        
    elif product:
        logger.info(f"Filtering downloads for '{product}' product...")
        short_names = [product]
        if "DSWX" in product:
            target_layers = ["WTR", "BWTR", "CONF"]
        elif "DIST" in product:
            target_layers = ["VEG-ANOM-MAX", "VEG-DIST-STATUS", "VEG-DIST-DATE", "VEG-DIST-CONF"]
        elif "RTC" in product:
            target_layers = ["RTC-VV", "RTC-VH"]
            
        df_opera = df_opera[df_opera["Dataset"].isin(short_names)]
        url_cols = [f"Download URL {layer}" for layer in target_layers if f"Download URL {layer}" in df_opera.columns]

    else:
        logger.info("No mode or product specified. Downloading ALL available OPERA products and layers.")
        url_cols = [c for c in df_opera.columns if c.startswith("Download URL")]

    if df_opera.empty or not url_cols:
        logger.warning(f"No corresponding products found in the catalog for the specified criteria.")
        return None

    # Copy the metadata excel file to the user's output directory
    metadata_file = output_dir_np / "opera_products_metadata.xlsx"
    if metadata_file.exists():
        shutil.copy(metadata_file, output_dir / "opera_products_metadata.xlsx")

    # Extract all valid URLs
    urls_to_download = []
    for c in url_cols:
        urls_to_download.extend(df_opera[c].dropna().tolist())
    urls_to_download = list(set(urls_to_download))

    if not urls_to_download:
        logger.warning("No valid download URLs found after filtering.")
        return None

    logger.info(f"Found {len(urls_to_download)} files to download.")

    # Multithreaded Downloader Function
    def download_file(url):
        filename = url.split('/')[-1]
        local_path = data_dir / filename
        
        # Skip if already downloaded
        if local_path.exists():
            logger.info(f"File already exists, skipping: {filename}")
            return
            
        logger.info(f"Downloading {filename}...")
        try:
            # Use Earthdata authenticated file opener and stream to disk chunk-by-chunk
            with open_file(url, earthdata_username=username, earthdata_password=password) as f_in:
                with open(local_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")

    # Download concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(download_file, urls_to_download))

    return data_dir


def run_mosaic_only(input_dir: Path, output_dir: Path, bbox: Sequence[float] | str | None, benchmark: bool) -> Path | None:
    """
    Run a standalone mosaicking pipeline on local data.
    """
    logger.info(f"Running standalone MOSAIC pipeline using data from: {input_dir}")
    
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist.")
        return None
        
    df_opera = scan_local_directory(input_dir)
    if df_opera.empty:
        return None
        
    ensure_directory(output_dir)
    
    # Calculate spatial properties directly from files
    auto_bbox, target_crs_proj4 = get_local_spatial_properties(df_opera)
    
    # Handle user bbox override
    if bbox is not None:
        if isinstance(bbox, str):
            try:
                from shapely import wkt
                geom = wkt.loads(bbox)
                b_minx, b_miny, b_maxx, b_maxy = geom.bounds
                internal_bbox = [b_miny, b_maxy, b_minx, b_maxx]
                logger.info(f"Using user-provided WKT bounds: {internal_bbox}")
            except Exception as e:
                logger.error(f"Failed to parse user WKT: {e}")
                return None
        else:
            internal_bbox = list(bbox)
            logger.info(f"Using user-provided S N W E bounds: {internal_bbox}")
    else:
        internal_bbox = auto_bbox
        logger.info(f"Auto-calculated bounding box from input files: {internal_bbox}")
        
    # Calculate Master Grid
    crs_obj = pyproj.CRS.from_string(target_crs_proj4)
    target_res = 0.0002695 if crs_obj.is_geographic else 30
    master_grid = get_master_grid_props(internal_bbox, target_crs_proj4, target_res=target_res)
    
    # Extract Datasets and Dates
    df_opera["Start Date"] = df_opera["Start Time"].dt.date.astype(str)
    unique_datasets = df_opera["Dataset"].dropna().unique()
    
    for short_name in unique_datasets:
        df_sn = df_opera[df_opera["Dataset"] == short_name]
        unique_dates = df_sn["Start Date"].dropna().unique()
        
        # Dynamically find all layer columns for this specific dataset
        layer_cols = [c.replace("Download URL ", "") for c in df_sn.columns if c.startswith("Download URL ")]
        
        resampling_method = Resampling.bilinear if "RTC" in short_name else Resampling.nearest
        
        for date in unique_dates:
            df_on_date = df_sn[df_sn["Start Date"] == date]
            
            # Cluster by time to separate distinct satellite passes
            from .catalog import cluster_by_time
            time_clusters = cluster_by_time(df_on_date, time_col="Start Time", threshold_minutes=120)
            
            for cluster_df in time_clusters:
                pass_id = cluster_df["Start Time"].min().strftime("%Y%m%dT%H%M")
                
                for layer in layer_cols:
                    # Skip auxiliary layers in the main loop (they get processed with their parent layer)
                    if layer in ["CONF", "VEG-DIST-DATE", "VEG-DIST-CONF"]: 
                        continue
                        
                    url_column = f"Download URL {layer}"
                    if url_column not in cluster_df.columns: 
                        continue
                        
                    urls = cluster_df[url_column].dropna().tolist()
                    if not urls: 
                        continue
                        
                    logger.info(f"Mosaicking {short_name} - {layer} for pass {pass_id}")
                    
                    # Use GDAL direct-to-disk for memory-heavy RTC products
                    if "RTC" in short_name:
                        gdal.PushErrorHandler('CPLQuietErrorHandler')
                        
                        height, width = master_grid['shape']
                        transform = master_grid['transform']
                        min_x = transform.c
                        max_y = transform.f
                        max_x = min_x + (transform.a * width)
                        min_y = max_y + (transform.e * height)
                        output_bounds = [min_x, min_y, max_x, max_y]

                        mosaic_name = f"{short_name}_{layer}_{pass_id}_mosaic.tif"
                        mosaic_path = output_dir / mosaic_name
                        tmp_path = output_dir / f"tmp_{mosaic_name}"

                        opened_datasets = []
                        for u in urls:
                            ds = gdal.Open(u)
                            if ds is not None: opened_datasets.append(ds)

                        gdal.PopErrorHandler()
                        if not opened_datasets: continue

                        warp_options = gdal.WarpOptions(
                            format='GTiff', outputBounds=output_bounds, width=width, height=height,
                            dstSRS=master_grid['dst_crs'], resampleAlg='bilinear', dstNodata=np.nan,
                            creationOptions=["COMPRESS=DEFLATE", "NUM_THREADS=ALL_CPUS"],
                            warpOptions=["NUM_THREADS=ALL_CPUS"], warpMemoryLimit=4096,
                            outputType=gdal.GDT_Float32
                        )
                        
                        gdal.Warp(str(tmp_path), opened_datasets, options=warp_options)

                        # Explicitly close datasets to free C++ memory
                        for ds in opened_datasets: ds = None
                        opened_datasets = []

                        save_gtiff_as_cog(tmp_path, mosaic_path)
                        cleanup_temp_file(tmp_path)
                        continue

                    # xarray/rioxarray for rule-based DSWx/DIST products
                    # Look for a CONF layer to stack for synchronized pixel selection
                    conf_column = "Download URL CONF"
                    conf_urls = cluster_df[conf_column].dropna().tolist() if conf_column in cluster_df.columns else []
                    
                    # Trick `compile_and_load_data` into returning the CONF datasets
                    if conf_urls:
                        DS, conf_DS = compile_and_load_data(urls, mode="flood", conf_layer_links=conf_urls, benchmark_stats=None)
                    else:
                        DS = compile_and_load_data(urls, mode="other", benchmark_stats=None)
                        conf_DS = None

                    all_warped_ds = []
                    warp_temp_dir = output_dir / ".warp_cache"
                    
                    for i, da in enumerate(DS):
                        try:
                            da_warped = warp_dataarray_to_grid(
                                da,
                                master_grid,
                                resampling_method,
                                warp_temp_dir,
                                temp_prefix=f"{short_name}_{layer}_{pass_id}_data_{i}",
                                username=username,
                                password=password,
                            )
                            
                            # If a CONF layer exists, warp it and concatenate it underneath the main layer
                            # This ensures mosaic_opera brings the exact matching CONF pixel along with the WTR pixel.
                            if conf_DS is not None and i < len(conf_DS):
                                conf_warped = warp_dataarray_to_grid(
                                    conf_DS[i],
                                    master_grid,
                                    resampling_method,
                                    warp_temp_dir,
                                    temp_prefix=f"{short_name}_{layer}_{pass_id}_conf_{i}",
                                    username=username,
                                    password=password,
                                )
                                combined = xr.concat([da_warped, conf_warped], dim="band")
                                combined = combined.assign_coords(band=[1, 2])
                                all_warped_ds.append(combined)
                            else:
                                all_warped_ds.append(da_warped)
                        except Exception as e:
                            logger.warning(f"Failed to reproject a granule: {e}")

                    if not all_warped_ds:
                        continue
                        
                    # Apply the OPERA pixel-priority rules (Water beats Cloud, etc.)
                    mosaic, colormap, nodata = mosaic_opera(all_warped_ds, product=short_name, merge_args={})

                    # Split the synchronized CONF layer back out if we stacked it
                    conf_mosaic = None
                    if mosaic.shape[0] == 2:
                        conf_mosaic = mosaic.isel(band=[1]).copy() # Band 2 is CONF
                        mosaic = mosaic.isel(band=[0]).copy()      # Band 1 is Main Data

                    # Save Main Layer
                    mosaic_name = f"{short_name}_{layer}_{pass_id}_mosaic.tif"
                    mosaic_path = output_dir / mosaic_name
                    
                    array_to_image(
                        mosaic,
                        output=mosaic_path,
                        driver="GTiff",
                        colormap=colormap,
                        nodata=nodata,
                    )

                    # Translate the standard GTiff into a COG in-place
                    save_gtiff_as_cog(mosaic_path, mosaic_path)
                    
                    # Save Synchronized CONF Layer (if it was requested via stacking)
                    if conf_mosaic is not None:
                        conf_name = f"{short_name}_CONF_{pass_id}_mosaic.tif"
                        conf_path = output_dir / conf_name
                        
                        array_to_image(
                            conf_mosaic,
                            output=conf_path,
                            driver="GTiff",
                            colormap=None,
                            nodata=255,
                        )

                        # Translate the standard GTiff into a COG in-place
                        save_gtiff_as_cog(mosaic_path, mosaic_path)

                    # Close all xarray handles to prevent sys.excepthook teardown crashes
                    if DS is not None:
                        for da in DS: da.close()
                    if conf_DS is not None:
                        for da in conf_DS: da.close()
                        
    return output_dir


def run_plotting_task(
    maps_dir, layouts_dir, mosaic_path, short_name, layer, 
    date_id, layout_date, layout_title, bbox, zoom_bbox, 
    reclassify_snow_ice, is_difference, skip_existing, benchmark_mode=False
) -> float:
    """
    Wrapper function to run map and layout generation in a separate process.

    Args:
        maps_dir (Path): Directory to save output map images.
        layouts_dir (Path): Directory to save output layouts.
        mosaic_path (Path): Path to the generated GeoTIFF mosaic.
        short_name (str): Product short name (e.g., OPERA_L3_DSWX-HLS_V1).
        layer (str): Specific layer being mapped.
        date_id (str): Formatted pass string.
        layout_date (str): Title string for the layout date.
        layout_title (str): Title string for the final layout.
        bbox (list[float]): Boundary box coordinates for mapping.
        zoom_bbox (list[float] | None): Inset boundary box coordinates for zooming.
        reclassify_snow_ice (bool): Reclassification flag.
        is_difference (bool): Flag indicating if this is a diff map.
        skip_existing (bool): Whether to skip plotting if output already exists.
        benchmark_mode (bool): Toggle for benchmark timings.

    Returns:
        float: Elapsed time if successful, 0.0 otherwise.
    """

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    t0 = time.time()
    try:
        map_name = make_map(
            maps_dir, mosaic_path, short_name, layer, date_id, 
            bbox, zoom_bbox, is_difference, skip_existing=skip_existing
        )
        if map_name:
            make_layout(
                layouts_dir, map_name, short_name, layer, 
                date_id, layout_date, layout_title, reclassify_snow_ice, skip_existing=skip_existing
            )
        return time.time() - t0
    except Exception as e:
        logger.error(f"Background plotting failed for {short_name} {layer} {date_id}: {e}")
        return 0.0


def run_difference_pipeline(
    earlier_path, later_path, diff_path, mode,
    maps_dir, layouts_dir, short_name, layer,
    diff_id, diff_date_str, layout_title, bbox, zoom_bbox,
    reclassify_snow_ice, skip_existing
) -> tuple:
    """
    Combined task pipeline computing difference maps and plotting the layouts.

    Args:
        earlier_path (Path): Filepath to the chronologically earlier mosaic.
        later_path (Path): Filepath to the chronologically later mosaic.
        diff_path (Path): Destination filepath for the calculated difference map.
        mode (str): Mode of the pipeline execution (e.g., 'flood', 'landslide').
        maps_dir (Path): Output directory for the raw maps.
        layouts_dir (Path): Output directory for formatted layouts.
        short_name (str): The product short name.
        layer (str): The specific layer being compared.
        diff_id (str): Identifier joining the compared dates.
        diff_date_str (str): Date string to be displayed in layout.
        layout_title (str): Primary layout map title.
        bbox (list[float]): Coordinate bounds for visualization.
        zoom_bbox (list[float] | None): Optional zoomed inset map bounds.
        reclassify_snow_ice (bool): Rule flag indicating snow/ice processing.
        skip_existing (bool): Whether to skip processing if outputs already exist.

    Returns:
        tuple: (diff_time, plot_time) floating point times in seconds.
    """
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Differencing
    t0_diff = time.time()
    t_diff = 0.0

    # Skip if diff already exists
    if skip_existing and diff_path.exists():
        logger.info(f"Difference map already exists, skipping: {diff_path.name}")
        is_diff = True
    else:
        try:
            if mode == "flood":
                compute_and_write_difference_positive_change_only(earlier_path, later_path, diff_path)
                is_diff = True
            elif mode == "landslide":
                compute_and_write_difference(earlier_path, later_path, diff_path, nodata_value=None, log=True)
                is_diff = True
            else:
                return 0.0, 0.0
        except Exception as e:
            logger.error(f"Diff computation failed: {e}")
            return 0.0, 0.0
        t_diff = time.time() - t0_diff

    # Plotting (Sequential within this worker, but parallel to main)
    t_plot = run_plotting_task(
        maps_dir, layouts_dir, diff_path, short_name, layer,
        diff_id, diff_date_str, layout_title, bbox, zoom_bbox,
        reclassify_snow_ice, is_diff, skip_existing=skip_existing
        )
    return t_diff, t_plot


def run_max_extent_pipeline(
    input_paths, out_path, maps_dir, layouts_dir, short_name, layer,
    diff_id_str, diff_date_str_layout, layout_title, bbox, zoom_bbox,
    reclassify_snow_ice, skip_existing
) -> tuple:
    """Computes max flood extent and waits for it to finish before plotting.
    
    Args:
        input_paths (list[Path]): List of filepaths to all mosaics to be included in the max extent calculation.
        out_path (Path): Destination filepath for the calculated max extent map.
        maps_dir (Path): Output directory for the raw maps.
        layouts_dir (Path): Output directory for formatted layouts.
        short_name (str): The product short name.
        layer (str): The specific layer being processed.
        diff_id_str (str): Identifier joining the compared dates.
        diff_date_str_layout (str): Date string to be displayed in layout.
        layout_title (str): Primary layout map title.
        bbox (list[float]): Coordinate bounds for visualization.
        zoom_bbox (list[float] | None): Optional zoomed inset map bounds.
        reclassify_snow_ice (bool): Rule flag indicating snow/ice reclassification.
        skip_existing (bool): Whether to skip processing if outputs already exist.

    Returns:
        tuple: (diff_time, plot_time) floating point times in seconds. Max extent calculation time is included in diff_time.
    """
    from .diff import compute_and_write_max_flood_extent
    
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Skip if it exists
    if skip_existing and out_path.exists():
        logger.info(f"Maximum flood extent already exists, skipping: {out_path.name}")
    else:
        # Generate the tiff
        try:
            compute_and_write_max_flood_extent(input_paths, out_path)
        except Exception as e:
            logger.error(f"Max Extent computation failed: {e}")
            return 0.0, 0.0
        
    # Plot it after generation is complete
    run_plotting_task(
        maps_dir, layouts_dir, out_path, short_name, layer,
        diff_id_str, diff_date_str_layout, layout_title, bbox, zoom_bbox,
        reclassify_snow_ice, is_difference = False, skip_existing=skip_existing
    )
    return 0.0, 0.0


def run_rgb_task(vv_path, vh_path, rgb_path, skip_existing) -> float:
    """
    Wrapper function to execute RTC RGB composite visualizations and catch exceptions.

    Args:
        vv_path (Path): Source path to the VV Float32 mosaic.
        vh_path (Path): Source path to the VH Float32 mosaic.
        rgb_path (Path): Output destination for the calculated RGB GeoTIFF.
        skip_existing (bool): Whether to skip processing if the RGB composite already exists.

    Returns:
        float: Elapsed processing time in seconds.
    """

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    t0 = time.time()

    # Skip if already exists
    if skip_existing and rgb_path.exists():
        logger.info(f"RGB composite already exists, skipping: {rgb_path.name}")
        return 0.0

    try:
        create_rtc_rgb_visualization(vv_path, vh_path, rgb_path)
        logger.info(f"Successfully generated RGB composite: {rgb_path.name}")
    except Exception as e:
        logger.error(f"RGB Generation failed for {rgb_path.name}: {e}")
    return time.time() - t0


def generate_products(
    df_opera, mode, mode_dir: Path, layout_title: str, bbox: list[float], zoom_bbox: list[float] | None,
    filter_date: str | None = None, reclassify_snow_ice: bool = False, slope_threshold: int | None = None,
    benchmark_stats: dict | None = None, username: str | None = None, password: str | None = None,
    no_mask: bool = False, skip_existing: bool = False, product: str | None = None
) -> None:
    """
    Generate mosaicked products, maps, and layouts based on the provided DataFrame and mode. 
    Granules are reprojected to the most common UTM zone present in the data for a given date.

    Args:
        df_opera (pd.DataFrame): Dataframe of aggregated metadata generated by next_pass.
        mode (str): Contextual mode (e.g., "flood", "fire", "landslide", "rtc-rgb").
        mode_dir (Path): Active output directory path.
        layout_title (str): Output string mapped into the PDF layout.
        bbox (list[float]): Working bounds.
        zoom_bbox (list[float] | None): Sub-region bounds for inset.
        filter_date (str | None): Target comparison threshold date string.
        reclassify_snow_ice (bool): Triggers specific filters for DSWx rules.
        slope_threshold (int | None): Degree limit for masking pixels via topography.
        benchmark_stats (dict | None): Optional collector dict for execution timings.
        username (str | None): Earthdata auth credentials.
        password (str | None): Earthdata auth credentials.
        no_mask (bool): If True, skips coastal masking step.
        skip_existing (bool): If True, skips processing steps for outputs that already exist.
        product (str | None): Specific OPERA product to generate (overrides mode).

    Returns:
        None
    """
    # Define short names and layer names based on mode or product FIRST
    if mode:
        if mode == "flood":
            short_names = ["OPERA_L3_DSWX-HLS_V1", "OPERA_L3_DSWX-S1_V1"]
            layer_names = ["WTR", "BWTR"]
        elif mode == "fire":
            short_names = ["OPERA_L3_DIST-ALERT-HLS_V1", "OPERA_L3_DIST-ALERT-S1_V1"]
            layer_names = ["VEG-ANOM-MAX", "VEG-DIST-STATUS"]
        elif mode == "landslide":
            short_names = ["OPERA_L3_DIST-ALERT-HLS_V1", "OPERA_L2_RTC-S1_V1"]
            layer_names = ["VEG-ANOM-MAX", "VEG-DIST-STATUS", "RTC-VV", "RTC-VH"]
        elif mode == "rtc-rgb":
            short_names = ["OPERA_L2_RTC-S1_V1"]
            layer_names = ["RTC-VV", "RTC-VH"]
        elif mode == "earthquake":
            logger.info("Earthquake mode coming soon. Exiting...")
            return
            
    elif product:
        short_names = [product]
        if "DSWX" in product:
            layer_names = ["WTR", "BWTR", "CONF"]
            mode = "flood" # Inherit mode logic for downstream processing
        elif "DIST" in product:
            layer_names = ["VEG-ANOM-MAX", "VEG-DIST-STATUS", "VEG-DIST-DATE", "VEG-DIST-CONF"]
            mode = "fire"
        elif "RTC" in product:
            layer_names = ["RTC-VV", "RTC-VH"]
            mode = "rtc-rgb"
            
    # Filter to see if we have ANY data for the products required by this mode, if not, exit
    df_mode_data = df_opera[df_opera["Dataset"].isin(short_names)]
    if df_mode_data.empty:
        logger.warning(f"No corresponding products ({', '.join(short_names)}) found for this date range. Exiting gracefully.")
        return

    # Create directories
    data_dir = ensure_directory(mode_dir / "data")
    maps_dir = ensure_directory(mode_dir / "maps")
    layouts_dir = ensure_directory(mode_dir / "layouts")

    # Determine most common UTM CRS to warp all granules to across all dates
    target_crs_proj4 = get_master_crs(df_mode_data, mode)
    
    # Detect if the CRS is geographic to set the correct resolution
    crs_obj = pyproj.CRS.from_string(target_crs_proj4)
    if crs_obj.is_geographic:
        target_res = 0.0002695 # ~30m in degrees
    else:
        target_res = 30 # 30m in projected units

    # Define the master grid properties
    master_grid = get_master_grid_props(bbox, target_crs_proj4, target_res=target_res)
    
    # Generate Slope Mask if requested
    global_slope_mask = None
    if mode in ["landslide", "rtc-rgb"] and slope_threshold is not None:
        global_slope_mask = process_dem_and_slope(df_opera, master_grid, slope_threshold, data_dir, skip_existing=skip_existing)

    # Generate Global Coastal Mask only if not explicitly disabled
    global_coastal_mask = None
    if not no_mask:
        logger.info("Generating global coastal mask...")
        global_coastal_mask = generate_coastal_mask(bbox, master_grid)
    else:
        logger.info("Coastal masking disabled by user.")
    
    # Define the resampling method.
    resampling_method = Resampling.bilinear if mode in ["landslide", "rtc-rgb"] else Resampling.nearest

    # Extract and find unique dates, sort them
    df_opera["Start Date"] = df_opera["Start Time"].dt.date.astype(str)
    unique_dates = sorted(df_opera["Start Date"].dropna().unique())

    # Create an index of mosaics created for use in pair-wise differencing
    mosaic_index = defaultdict(lambda: defaultdict(dict))
    
    # Avoid background process pools here. Spawning workers while GDAL/rasterio
    # work is still ongoing has been corrupting native library state on Linux.
    executor = DeferredExecutor()
    plotting_futures = []
    differencing_futures = []

    def ensure_executor():
        return executor

    try:
        for date in unique_dates:
            df_on_date = df_opera[df_opera["Start Date"] == date]

            for short_name in short_names:
                df_sn = df_on_date[df_on_date["Dataset"] == short_name]
                if df_sn.empty: continue

                # Cluster granules by time to separate Ascending/Descending passes
                # Threshold: 120 minutes (2 hours). Passes are typically >10 hours apart.
                time_clusters = cluster_by_time(df_sn, time_col="Start Time", threshold_minutes=120)

                for layer in layer_names:
                    url_column = f"Download URL {layer}"
                    if url_column not in df_sn.columns: continue
                    
                    # Iterate through each clustered "pass" for this date
                    for cluster_df in time_clusters:
                        # Determine unique PassID from the earliest time in the cluster
                        # Format: YYYYMMDDtHHMM (e.g., 20241010t0019)
                        start_time_min = cluster_df["Start Time"].min()
                        raw_pass_id = start_time_min.strftime("%Y%m%dT%H%M")
                        
                        urls = cluster_df[url_column].dropna().tolist()
                        if not urls: continue

                        # Extract S1 orbit direction for S1 ---
                        flight_dir = ""
                        if "S1" in short_name:
                            flight_dir = get_S1_orbit_direction(urls, username, password)

                        pass_id = f"{raw_pass_id}{flight_dir}"

                        logger.info(f"Processing {short_name} - {layer} for pass {pass_id} (Date: {date})")
                        logger.info(f"Found {len(urls)} URLs for this pass")
                        
                        layout_date = ""

                        # Use GDAL direct-to-disk mosaicking for RTC products to conserve RAM
                        if short_name == "OPERA_L2_RTC-S1_V1":
                            mosaic_name = f"{short_name}_{layer}_{pass_id}_mosaic.tif"
                            mosaic_path = data_dir / mosaic_name
                            tmp_path = data_dir / f"tmp_{mosaic_name}"

                            # Skip if RTC mosaic already exists
                            if skip_existing and mosaic_path.exists():
                                logger.info(f"Mosaic already exists, skipping: {mosaic_name}")
                                with rasterio.open(mosaic_path) as ds:
                                    mosaic_crs = ds.crs
                                # Register it for differencing later!
                                mosaic_index[short_name][layer][pass_id] = {
                                    "path": mosaic_path,
                                    "crs": mosaic_crs,
                                    "flight_dir": flight_dir
                                    }
                                
                                if mode != "rtc-rgb":
                                    future = ensure_executor().submit(
                                        run_plotting_task, maps_dir, layouts_dir, mosaic_path, short_name, layer,
                                        pass_id, layout_date, layout_title, bbox, zoom_bbox,
                                        reclassify_snow_ice, False, benchmark_mode=(benchmark_stats is not None),
                                        skip_existing=skip_existing
                                    )
                                    plotting_futures.append(future)
                                continue

                            logger.info("Using GDAL direct-to-disk mosaicking for RTC to conserve RAM.")
                            
                            gdal.PushErrorHandler('CPLQuietErrorHandler')

                            # Extract bounds from master_grid for exact pixel alignment
                            height, width = master_grid['shape']
                            transform = master_grid['transform']
                            min_x = transform.c
                            max_y = transform.f
                            max_x = min_x + (transform.a * width)
                            min_y = max_y + (transform.e * height)
                            output_bounds = [min_x, min_y, max_x, max_y]

                            # Open datasets to avoid GDALDatasetShadow errors
                            opened_datasets = []
                            for u in urls:
                                ds = gdal.Open(u)
                                if ds is None:
                                    logger.warning(f"GDAL failed to open URL (likely auth or missing file), skipping: {u}")
                                    continue
                                opened_datasets.append(ds)

                            gdal.PopErrorHandler()

                            if not opened_datasets:
                                logger.error(f"No valid datasets could be opened for pass {pass_id}. Skipping.")
                                continue

                            # Define Memory-Capped GDAL Warp Options
                            warp_options = gdal.WarpOptions(
                                format='GTiff',
                                outputBounds=output_bounds,
                                width=width,
                                height=height,
                                dstSRS=master_grid['dst_crs'],
                                resampleAlg='bilinear',
                                dstNodata=np.nan,
                                creationOptions=["COMPRESS=DEFLATE", "NUM_THREADS=ALL_CPUS"],
                                warpOptions=["NUM_THREADS=ALL_CPUS"],
                                warpMemoryLimit=4096, # Cap RAM usage at 4GB
                                outputType=gdal.GDT_Float32
                            )
                            
                            # Execute Warp straight to disk
                            gdal.Warp(str(tmp_path), opened_datasets, options=warp_options)

                            # Explicitly close datasets to free C++ memory!
                            for ds in opened_datasets:
                                ds = None
                            opened_datasets = []

                            # Apply optional masks quickly to the single, cropped temp file
                            if global_slope_mask is not None or global_coastal_mask is not None:
                                with rasterio.open(tmp_path, "r+") as ds:
                                    arr = ds.read(1)
                                    if global_slope_mask is not None and arr.shape == global_slope_mask.shape:
                                        arr[global_slope_mask] = np.nan
                                    if global_coastal_mask is not None and arr.shape == global_coastal_mask.shape:
                                        arr[~global_coastal_mask.values] = np.nan
                                    ds.write(arr, 1)

                            # Convert to COG
                            save_gtiff_as_cog(tmp_path, mosaic_path)
                            cleanup_temp_file(tmp_path)

                            # Register to index for downstream differencing
                            mosaic_index[short_name][layer][pass_id] = {
                                "path": mosaic_path,
                                "crs": master_grid["dst_crs"],
                                "flight_dir": flight_dir
                            }

                            # Submit Plotting Task (Skip individual layouts for rtc-rgb mode)
                            if mode != "rtc-rgb":
                                future = ensure_executor().submit(
                                    run_plotting_task,
                                    maps_dir, layouts_dir, mosaic_path, short_name, layer,
                                    pass_id, layout_date, layout_title, bbox, zoom_bbox,
                                    reclassify_snow_ice, False, benchmark_mode=(benchmark_stats is not None),
                                    skip_existing=skip_existing
                                )
                                plotting_futures.append(future)

                            continue # Skip the xarray processing loops entirely for RTC

                        # For non-RTC products, we load the granules into xarray DataArrays for filtering and mosaicking
                        mosaic_name = f"{short_name}_{layer}_{pass_id}_mosaic.tif"
                        mosaic_path = data_dir / mosaic_name
                        tmp_path = data_dir / f"tmp_{mosaic_name}"
                        conf_name = f"{short_name}_CONF_{pass_id}_mosaic.tif"
                        conf_path = data_dir / conf_name
                        conf_tmp = data_dir / f"tmp_{conf_name}"
                        
                        conf_column = "Download URL CONF" if mode == "flood" else "Download URL VEG-DIST-CONF"
                        has_conf = conf_column in cluster_df.columns and not cluster_df[conf_column].dropna().empty
                        needs_conf = (mode == "flood" and layer == "WTR" and has_conf)

                        if skip_existing and mosaic_path.exists() and (not needs_conf or conf_path.exists()):
                            logger.info(f"Mosaic already exists, skipping: {mosaic_name}")
                            with rasterio.open(mosaic_path) as ds:
                                mosaic_crs = ds.crs
                            
                            mosaic_index[short_name][layer][pass_id] = {
                                "path": mosaic_path,
                                "crs": mosaic_crs,
                                "flight_dir": flight_dir}
                            
                            future = ensure_executor().submit(
                                run_plotting_task, maps_dir, layouts_dir, mosaic_path, short_name, layer,
                                pass_id, layout_date, layout_title, bbox, zoom_bbox,
                                reclassify_snow_ice, False, benchmark_mode=(benchmark_stats is not None),
                                skip_existing=skip_existing
                            )
                            plotting_futures.append(future)
                            
                            if needs_conf:
                                future = ensure_executor().submit(
                                    run_plotting_task, maps_dir, layouts_dir, conf_path, short_name, "CONF",
                                    pass_id, layout_date, layout_title, bbox, zoom_bbox,
                                    False, False, benchmark_mode=(benchmark_stats is not None),
                                    skip_existing=skip_existing
                                )
                                plotting_futures.append(future)
                            continue

                        DS, conf_DS, date_DS = None, None, None
                        conf_colormap = None

                        if mode == "fire":
                            date_column = "Download URL VEG-DIST-DATE"
                            conf_column = "Download URL VEG-DIST-CONF"
                            date_layer_links = cluster_df[date_column].dropna().tolist() if date_column in cluster_df.columns else []
                            conf_layer_links = cluster_df[conf_column].dropna().tolist() if conf_column in cluster_df.columns else []
                            
                            DS, date_DS, conf_DS = compile_and_load_data(
                                urls, mode, conf_layer_links=conf_layer_links, date_layer_links=date_layer_links,
                                benchmark_stats=benchmark_stats, username=username, password=password
                            )
                            if filter_date:
                                date_threshold = compute_date_threshold(filter_date)
                                layout_date = str(filter_date)
                            else:
                                date_threshold = 0
                                layout_date = "All Dates"

                        elif mode == "landslide":
                            if short_name == "OPERA_L3_DIST-ALERT-HLS_V1":
                                date_column = "Download URL VEG-DIST-DATE"
                                conf_column = "Download URL VEG-DIST-CONF"
                                date_layer_links = cluster_df[date_column].dropna().tolist() if date_column in cluster_df.columns else []
                                conf_layer_links = cluster_df[conf_column].dropna().tolist() if conf_column in cluster_df.columns else []
                                
                                DS, date_DS, conf_DS = compile_and_load_data(
                                    urls, mode, conf_layer_links=conf_layer_links, date_layer_links=date_layer_links,
                                    benchmark_stats=benchmark_stats, username=username, password=password
                                )
                                if filter_date:
                                    date_threshold = compute_date_threshold(filter_date)
                                    layout_date = str(filter_date)
                                else:
                                    date_threshold = 0
                                    layout_date = "All Dates"

                        elif mode == "flood":
                            conf_column = "Download URL CONF"
                            conf_layer_links = cluster_df[conf_column].dropna().tolist() if conf_column in cluster_df.columns else []
                            if not conf_layer_links:
                                logger.warning(f"No CONF URLs found for {short_name} on {pass_id}")
                                conf_DS = None
                                DS = compile_and_load_data(urls, mode, benchmark_stats=benchmark_stats, username=username, password=password)
                            else:
                                DS, conf_DS = compile_and_load_data(urls, mode, conf_layer_links=conf_layer_links, benchmark_stats=benchmark_stats, username=username, password=password)
                                if conf_DS and len(conf_DS) > 0:
                                    try:
                                        conf_colormap = get_image_colormap(conf_DS[0])
                                    except Exception:
                                        pass

                        # Group loaded DataArrays by CRS (UTM Zone)
                        crs_groups = defaultdict(list)
                        conf_groups = defaultdict(list)
                        date_groups = defaultdict(list)

                        # Ensure all lists are non-empty before zipping
                        if not DS: continue

                        # Determine auxiliary list lengths for zipping
                        aux_lists = []
                        if conf_DS is not None and mode == "flood": aux_lists.append(conf_DS)
                        elif conf_DS is not None and mode in ["fire", "landslide"]: aux_lists.extend([date_DS, conf_DS])

                        if aux_lists:
                            # Zip DS with auxiliary layers (conf_DS, date_DS)
                            for i, (da_data, *aux_data) in enumerate(zip(DS, *aux_lists)):
                                try: crs_str = str(da_data.rio.crs)
                                except AttributeError: continue
                                crs_groups[crs_str].append(da_data)
                                if mode == "flood": conf_groups[crs_str].append(aux_data[0])
                                elif mode in ["fire", "landslide"] and short_name.startswith("OPERA_L3_DIST"):
                                    date_groups[crs_str].append(aux_data[0])
                                    conf_groups[crs_str].append(aux_data[1])
                        else:
                            for i, da_data in enumerate(DS):
                                try: crs_str = str(da_data.rio.crs)
                                except AttributeError: continue
                                crs_groups[crs_str].append(da_data)

                        all_warped_ds = []
                        colormap = None
                        warp_temp_dir = data_dir / ".warp_cache"

                        # Iterate through each CRS group to process and mosaic
                        for crs_str, ds_group in crs_groups.items():
                            current_conf_DS = conf_groups.get(crs_str)
                            current_date_DS = date_groups.get(crs_str)

                            # Filtering/Reclassification (Per CRS Group)
                            if mode == "fire" or (mode == "landslide" and short_name.startswith("OPERA_L3_DIST")):
                                ds_group, cmap_temp = filter_by_date_and_confidence(
                                    ds_group, current_date_DS, date_threshold, DS_conf=current_conf_DS, confidence_threshold=0, fill_value=None
                                )
                                if cmap_temp is not None: colormap = cmap_temp
                            elif mode == "flood":
                                if reclassify_snow_ice and short_name == "OPERA_L3_DSWX-HLS_V1" and layer in ["BWTR", "WTR"]:
                                    if current_conf_DS is not None:
                                        ds_group, cmap_temp = reclassify_snow_ice_as_water(ds_group, current_conf_DS)
                                        if cmap_temp is not None: colormap = cmap_temp

                            for i, da in enumerate(ds_group):
                                da_warped = warp_dataarray_to_grid(
                                    da,
                                    master_grid,
                                    resampling_method,
                                    warp_temp_dir,
                                    temp_prefix=f"{short_name}_{layer}_{pass_id}_data_{i}",
                                    username=username,
                                    password=password,
                                )
                                
                                # If processing WTR/BWTR, stack CONF as a second band to sync pixel selection
                                if mode == "flood" and layer == "WTR" and current_conf_DS is not None and i < len(current_conf_DS):
                                    conf_da = current_conf_DS[i]
                                    conf_warped = warp_dataarray_to_grid(
                                        conf_da,
                                        master_grid,
                                        resampling_method,
                                        warp_temp_dir,
                                        temp_prefix=f"{short_name}_{layer}_{pass_id}_conf_{i}",
                                        username=username,
                                        password=password,
                                    )
                                    # Concatenate into a single 2-band dataset
                                    combined = xr.concat([da_warped, conf_warped], dim="band")
                                    combined = combined.assign_coords(band=[1, 2])
                                    all_warped_ds.append(combined)
                                else:
                                    all_warped_ds.append(da_warped)
                        
                        if not all_warped_ds: continue
                        
                        if colormap is None:
                            try:
                                colormap = get_image_colormap(DS[0])
                            except Exception:
                                colormap = None
                        # Mosaic the datasets using the single global master grid setup
                        mosaic, _, nodata = mosaic_opera(all_warped_ds, product=short_name, merge_args={})

                        # Check if we have a synchronized CONF layer to split out
                        conf_mosaic = None
                        if mosaic.shape[0] == 2:
                            conf_mosaic = mosaic.isel(band=[1]).copy() # Band 2 is CONF
                            mosaic = mosaic.isel(band=[0]).copy()      # Band 1 is WTR

                        # Apply slope mask if it has been generated previously
                        if global_slope_mask is not None:
                            # Ensure shape compatibility
                            if mosaic.shape[-2:] == global_slope_mask.shape:
                                # Set pixels with slope < threshold to nodata
                                mosaic.values[..., global_slope_mask] = nodata
                                if conf_mosaic is not None: conf_mosaic.values[..., global_slope_mask] = 255
                            else:
                                logger.warning(f"Mask shape {global_slope_mask.shape} mismatches mosaic {mosaic.shape}. Skipping slope filter.")

                        # Apply coastal mask to ocean pixels
                        if global_coastal_mask is not None:
                            if mosaic.shape[-2:] == global_coastal_mask.shape:
                                # Mask out ocean (where global_coastal_mask is False)
                                mosaic.values[..., ~global_coastal_mask.values] = nodata
                                if conf_mosaic is not None: conf_mosaic.values[..., ~global_coastal_mask.values] = 255
                            else:
                                logger.warning("Coastal mask shape mismatches mosaic. Skipping coastal filter.")

                        # Mask DIST-HLS by corresponding DSWx-HLS to remove water-related disturbance (e.g., water banklines)
                        if mode == "landslide" and short_name == "OPERA_L3_DIST-ALERT-HLS_V1":
                            if slope_threshold is not None:
                                # Find DSWx-HLS WTR granules for the same datetime
                                dswx_rows = df_opera[(df_opera["Dataset"] == "OPERA_L3_DSWX-HLS_V1") & (df_opera["Start Date"] == date)]
                                
                                if "Download URL WTR" in dswx_rows.columns:
                                    dswx_urls = dswx_rows["Download URL WTR"].dropna().tolist()
                                else:
                                    dswx_urls = []
                                
                                if dswx_urls:
                                    logger.info(f"[Water Mask] Generating concurrent DSWx-HLS water mask for {date} to filter water-related disturbance...")
                                    
                                    # Load DSWx granules into xarray
                                    dswx_ds_list = compile_and_load_data(
                                        dswx_urls, mode="other", benchmark_stats=benchmark_stats, username=username, password=password
                                    )
                                    
                                    if dswx_ds_list:
                                        dswx_warped = []
                                        warp_temp_dir = data_dir / ".warp_cache"
                                        # Reproject to master grid
                                        for j, da_dswx in enumerate(dswx_ds_list):
                                            try:
                                                da_w = warp_dataarray_to_grid(
                                                    da_dswx,
                                                    master_grid,
                                                    Resampling.nearest,
                                                    warp_temp_dir,
                                                    temp_prefix=f"water_mask_{date}_{j}",
                                                    username=username,
                                                    password=password,
                                                )
                                                dswx_warped.append(da_w)
                                            except Exception as e:
                                                logger.warning(f"[Water Mask] Failed to reproject DSWx granule: {e}")
                                        
                                        if dswx_warped:
                                            # Apply OPERA pixel-priority rules
                                            dswx_mosaic, _, _ = mosaic_opera(dswx_warped, product="OPERA_L3_DSWX-HLS_V1", merge_args={})
                                            
                                            # Extract the numpy array
                                            water_arr = dswx_mosaic.squeeze().values
                                            
                                            # Identify Water (1) and Partial Water (2)
                                            is_water = (water_arr == 1) | (water_arr == 2) | (water_arr == 252)
                                            
                                            # Apply the mask to the DIST mosaic
                                            if mosaic.shape[-2:] == is_water.shape:
                                                mosaic.values[..., is_water] = nodata
                                                if conf_mosaic is not None:
                                                    conf_mosaic.values[..., is_water] = 255
                                                logger.info(f"[Water Mask] Success: Removed {np.sum(is_water)} water-related disturbance pixels.")
                                            else:
                                                logger.warning(f"[Water Mask] Shape {is_water.shape} mismatches mosaic {mosaic.shape[-2:]}. Skipping.")
                                            
                                            # Clean up the mosaic array
                                            dswx_mosaic.close()
                                            del dswx_mosaic
                                        
                                        # Close all DSWx arrays
                                        for da_dswx in dswx_warped: da_dswx.close()
                                        for da_dswx in dswx_ds_list: da_dswx.close()
                                        del dswx_warped
                                        del dswx_ds_list
                                        
                                        import gc
                                        gc.collect()
                                else:
                                    logger.warning(f"[Water Mask] No DSWx-HLS WTR granules found for {date}. Skipping water-related disturbance filtering.")
                            else:
                                logger.info("[Water Mask] No slope threshold (-st) specified. Skipping water-related disturbance filtering.")

                        array_to_image(
                            mosaic,
                            output=mosaic_path,
                            driver="GTiff",
                            colormap=colormap,
                            nodata=nodata,
                        )

                        # Translate the standard GTiff into a COG in-place
                        save_gtiff_as_cog(mosaic_path, mosaic_path)

                        with rasterio.open(mosaic_path) as ds:
                            mosaic_crs = ds.crs

                        # Add info to the mosiac index for pair-wise differencing
                        mosaic_index[short_name][layer][pass_id] = {
                            "path": mosaic_path,
                            "crs": mosaic_crs,
                            "flight_dir": flight_dir
                        }

                        # --- Background Plotting ---
                        logger.info(f"Submitting background plotting task for {pass_id}...")
                        if mode != "rtc-rgb":
                            future = ensure_executor().submit(
                                run_plotting_task,
                                maps_dir, layouts_dir, mosaic_path, short_name, layer,
                                pass_id, layout_date, layout_title, bbox, zoom_bbox,
                                reclassify_snow_ice, False, benchmark_mode=(benchmark_stats is not None),
                                skip_existing=skip_existing
                            )
                            plotting_futures.append(future)

                        # Save and plot the perfectly synced CONF layer if we generated it
                        if conf_mosaic is not None:
                            array_to_image(
                                conf_mosaic,
                                output=conf_path,
                                driver="GTiff",
                                colormap=conf_colormap,
                                nodata=255,
                            )

                            # Translate the standard GTiff into a COG in-place
                            save_gtiff_as_cog(conf_path, conf_path)

                            logger.info(f"Saved spatially synchronized CONF layer: {conf_name}")

                            # Submit Plotting Task for CONF
                            if mode != "rtc-rgb":
                                future = ensure_executor().submit(
                                    run_plotting_task,
                                    maps_dir, layouts_dir, conf_path, short_name, "CONF",
                                    pass_id, layout_date, layout_title, bbox, zoom_bbox,
                                    False, False, benchmark_mode=(benchmark_stats is not None),
                                    skip_existing=skip_existing
                                )
                                plotting_futures.append(future)

                        # Explicitly close xarray file
                        if DS is not None:
                            for da in DS: da.close()
                        if conf_DS is not None:
                            for da in conf_DS: da.close()
                        if date_DS is not None:
                            for da in date_DS: da.close()

        # RTC RGB Visualization Generation
        if mode in ["landslide", "rtc-rgb"] and "OPERA_L2_RTC-S1_V1" in mosaic_index:
            logger.info("Submitting concurrent RTC RGB visualization tasks...")
            rtc_dict = mosaic_index["OPERA_L2_RTC-S1_V1"]
            
            # Check if both VV and VH layers were successfully generated
            if "RTC-VV" in rtc_dict and "RTC-VH" in rtc_dict:
                # Find passes where we have both VV and VH
                vv_passes = set(rtc_dict["RTC-VV"].keys())
                vh_passes = set(rtc_dict["RTC-VH"].keys())
                common_passes = vv_passes.intersection(vh_passes)
                
                for pass_id in common_passes:
                    vv_path = rtc_dict["RTC-VV"][pass_id]["path"]
                    vh_path = rtc_dict["RTC-VH"][pass_id]["path"]
                    
                    rgb_name = f"OPERA_L2_RTC-S1_V1_RGB_{pass_id}.tif"
                    rgb_path = data_dir / rgb_name
                    
                    # Correctly submit the wrapper task instead of the core generation function
                    future = ensure_executor().submit(
                        run_rgb_task, 
                        vv_path, vh_path, rgb_path, skip_existing=skip_existing
                    )
                    plotting_futures.append(future)

        # Concurrent differencing
        if mode in ["flood", "landslide"]:
            logger.info(f"Submitting concurrent pair-wise differencing tasks ({mode})...")
            for short_name_k, layers_dict in mosaic_index.items():
                # Filter for relevant products only (RTC for Landslide, all for Flood)
                if mode == "landslide" and short_name_k != "OPERA_L2_RTC-S1_V1": continue
                
                for layer_k, date_map in layers_dict.items():
                    # Only generate water gain for WTR
                    if mode == "flood" and layer_k != "WTR": 
                        continue
                    dates = sorted(date_map.keys())
                    
                    for i in range(len(dates)):
                        for j in range(i + 1, len(dates)):
                            d_early = dates[i]
                            d_later = dates[j]

                            early_info = date_map[d_early]
                            later_info = date_map[d_later]
                            
                            if early_info["crs"] != later_info["crs"]:
                                continue

                            # Setup filenames and paths (d_early and d_later already contain A/D!)
                            suffix = "water_gain.tif" if mode == "flood" else "log-diff.tif"
                            diff_name = f"{short_name_k}_{layer_k}_{d_later}_{d_early}_{suffix}"
                            diff_path = data_dir / diff_name
                            
                            diff_id_str = f"{d_later}_{d_early}"
                            diff_date_str_layout = f"{d_early}, {d_later}"

                            # Submit Pipeline Task (Compute Diff -> Map -> Layout)
                            future = ensure_executor().submit(
                                run_difference_pipeline,
                                early_info["path"], later_info["path"], diff_path, mode,
                                maps_dir, layouts_dir, short_name_k, layer_k,
                                diff_id_str, diff_date_str_layout, layout_title, bbox, zoom_bbox,
                                reclassify_snow_ice, skip_existing=skip_existing
                            )
                            differencing_futures.append(future)
        
        # Compute Maximum Flood Extent
        if mode == "flood":
            logger.info("Submitting maximum flood extent tasks...")
            
            for short_name_k, layers_dict in mosaic_index.items():
                if "WTR" not in layers_dict: 
                    continue
                    
                date_map = layers_dict["WTR"]
                pass_ids = sorted(date_map.keys())
                
                # Requires at least 2 dates for a cumulative map
                if len(pass_ids) < 2:
                    continue 
                
                input_paths = [date_map[pid]["path"] for pid in pass_ids]
                
                earliest_date = pass_ids[0]
                latest_date = pass_ids[-1]

                earliest_clean = earliest_date[:-1] if earliest_date[-1].isalpha() else earliest_date
                latest_clean = latest_date[:-1] if latest_date[-1].isalpha() else latest_date

                out_name = f"{short_name_k}_WTR_{earliest_clean}_{latest_clean}_max_extent.tif"
                out_path = data_dir / out_name

                diff_id_str = f"{earliest_clean}_{latest_clean}"
                diff_date_str_layout = f"{earliest_clean}, {latest_clean}"
                
                future = ensure_executor().submit(
                    run_max_extent_pipeline,
                    input_paths, out_path,
                    maps_dir, layouts_dir, short_name_k, "MAX-EXTENT",
                    diff_id_str, diff_date_str_layout, layout_title, bbox, zoom_bbox,
                    reclassify_snow_ice, skip_existing=skip_existing
                )
                differencing_futures.append(future)
    finally:
        logger.info("Waiting for all background tasks to finish...")
        executor.shutdown(wait=True)
        if benchmark_stats is not None:
            # Process Plotting Futures (Standard Mosaics)
            total_plotting_time = sum(f.result() for f in plotting_futures if f.exception() is None)
            total_diff_time = 0.0
            
            # Process Differencing Pipeline Futures (Returns (diff_time, plot_time))
            for f in differencing_futures:
                if f.exception() is None:
                    d_t, p_t = f.result()
                    total_diff_time += d_t
                    total_plotting_time += p_t
            
            # Update Stats
            if 'plotting' in benchmark_stats: benchmark_stats['plotting']['seq'] = total_plotting_time
            if 'differencing' in benchmark_stats: benchmark_stats['differencing']['seq'] = total_diff_time
        logger.info("All tasks complete.")
