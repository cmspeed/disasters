from __future__ import annotations

import concurrent.futures
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import next_pass
import numpy as np
import pandas as pd
import pyproj
import rasterio
from osgeo import gdal
from rasterio.enums import Resampling
from rasterio.shutil import copy

from .auth import authenticate
from .catalog import cluster_by_time, read_opera_metadata
from .diff import (compute_and_write_difference,
                   compute_and_write_difference_positive_change_only,
                   save_gtiff_as_cog, create_rtc_rgb_visualization
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
from .mosaic import array_to_image, compile_and_load_data, get_master_crs, get_master_grid_props, get_image_colormap, mosaic_opera

logger = logging.getLogger(__name__)

gdal.DontUseExceptions()

@dataclass
class PipelineConfig:
    """
    Configuration for running the OPERA disaster pipeline.
    """
    bbox: Sequence[float]
    output_dir: Path
    layout_title: str

    zoom_bbox: Sequence[float] | None = None
    local_dir: Path | None = None
    short_name: str | None = None
    layer_name: str | None = None
    date: str | None = None
    number_of_dates: int = 5
    mode: str = "flood"
    functionality: str = "opera_search"
    filter_date: str | None = None
    reclassify_snow_ice: bool = False
    slope_threshold: int | None = None
    benchmark: bool = False


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
    mode_dir = config.output_dir / config.mode

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
        
        output_dir = next_pass.run_next_pass(
            bbox=config.bbox,
            number_of_dates=config.number_of_dates,
            date=config.date, # This can now safely be "2026-02-02/2026-02-27"
            functionality=config.functionality
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
        bbox=list(config.bbox),
        zoom_bbox=list(config.zoom_bbox) if config.zoom_bbox is not None else None,
        filter_date=config.filter_date,
        reclassify_snow_ice=config.reclassify_snow_ice,
        slope_threshold=config.slope_threshold,
        benchmark_stats=benchmark_stats,
        username=username,
        password=password
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


def run_plotting_task(
    maps_dir, layouts_dir, mosaic_path, short_name, layer, 
    date_id, layout_date, layout_title, bbox, zoom_bbox, 
    reclassify_snow_ice, is_difference, benchmark_mode=False
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
        benchmark_mode (bool): Toggle for benchmark timings.

    Returns:
        float: Elapsed time if successful, 0.0 otherwise.
    """
    t0 = time.time()
    try:
        map_name = make_map(
            maps_dir, mosaic_path, short_name, layer, date_id, 
            bbox, zoom_bbox, is_difference
        )
        if map_name:
            make_layout(
                layouts_dir, map_name, short_name, layer, 
                date_id, layout_date, layout_title, reclassify_snow_ice
            )
        return time.time() - t0
    except Exception as e:
        logger.error(f"Background plotting failed for {short_name} {layer} {date_id}: {e}")
        return 0.0


def run_difference_pipeline(
    earlier_path, later_path, diff_path, mode,
    maps_dir, layouts_dir, short_name, layer,
    diff_id, diff_date_str, layout_title, bbox, zoom_bbox,
    reclassify_snow_ice
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

    Returns:
        tuple: (diff_time, plot_time) floating point times in seconds.
    """
    # Differencing
    t0_diff = time.time()
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
        reclassify_snow_ice, is_diff
    )
    return t_diff, t_plot


def run_rgb_task(vv_path, vh_path, rgb_path) -> float:
    """
    Wrapper function to execute RTC RGB composite visualizations and catch exceptions.

    Args:
        vv_path (Path): Source path to the VV Float32 mosaic.
        vh_path (Path): Source path to the VH Float32 mosaic.
        rgb_path (Path): Output destination for the calculated RGB GeoTIFF.

    Returns:
        float: Elapsed processing time in seconds.
    """
    t0 = time.time()
    try:
        create_rtc_rgb_visualization(vv_path, vh_path, rgb_path)
        logger.info(f"Successfully generated RGB composite: {rgb_path.name}")
    except Exception as e:
        logger.error(f"RGB Generation failed for {rgb_path.name}: {e}")
    return time.time() - t0


def generate_products(
    df_opera, mode, mode_dir: Path, layout_title: str, bbox: list[float], zoom_bbox: list[float] | None,
    filter_date: str | None = None, reclassify_snow_ice: bool = False, slope_threshold: int | None = None,
    benchmark_stats: dict | None = None, username: str | None = None, password: str | None = None
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

    Returns:
        None
    """
    import multiprocessing

    # Define short names and layer names based on mode FIRST
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
        logger.info("Earthquake mode coming soon. Exiting...")
        return
    
    # Filter to see if we have ANY data for the products required by this mode, if not, exit
    df_mode_data = df_opera[df_opera["Dataset"].isin(short_names)]
    if df_mode_data.empty:
        logger.warning(f"No {mode.upper()} products ({', '.join(short_names)}) found for this date range. Exiting gracefully.")
        return

    # Create directories
    data_dir = ensure_directory(mode_dir / "data")
    maps_dir = ensure_directory(mode_dir / "maps")
    layouts_dir = ensure_directory(mode_dir / "layouts")

    # Determine most common UTM CRS to warp all granules to across all dates
    target_crs_proj4 = get_master_crs(df_mode_data, mode)
    
    # Detect if the CRS is geographic to set the correct resolution
    crs_obj = pyproj.CRS.from_proj4(target_crs_proj4)
    if crs_obj.is_geographic:
        target_res = 0.0002695 # ~30m in degrees
    else:
        target_res = 30 # 30m in projected units

    # Define the master grid properties
    master_grid = get_master_grid_props(bbox, target_crs_proj4, target_res=target_res)
    
    # Generate Slope Mask if requested
    global_slope_mask = None
    if mode in ["landslide", "rtc-rgb"] and slope_threshold is not None:
        global_slope_mask = process_dem_and_slope(df_opera, master_grid, slope_threshold, data_dir)

    # Generate Global Coastal Mask
    global_coastal_mask = generate_coastal_mask(bbox, master_grid)
    
    # Define the resampling method.
    resampling_method = Resampling.bilinear if mode in ["landslide", "rtc-rgb"] else Resampling.nearest

    # Extract and find unique dates, sort them
    df_opera["Start Date"] = df_opera["Start Time"].dt.date.astype(str)
    unique_dates = df_opera["Start Date"].dropna().unique()
    unique_dates.sort()

    # Create an index of mosaics created for use in pair-wise differencing
    mosaic_index = defaultdict(lambda: defaultdict(dict))
    
    # Initialize Executor for Plotting and Differencing
    ctx = multiprocessing.get_context('spawn')
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=ctx)
    plotting_futures = []
    differencing_futures = []

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
                        pass_id = start_time_min.strftime("%Y%m%dT%H%M")
                        
                        urls = cluster_df[url_column].dropna().tolist()
                        if not urls: continue

                        logger.info(f"Processing {short_name} - {layer} for pass {pass_id} (Date: {date})")
                        logger.info(f"Found {len(urls)} URLs for this pass")
                        
                        layout_date = ""

                        # Use GDAL direct-to-disk mosaicking for RTC products to conserve RAM
                        if short_name == "OPERA_L2_RTC-S1_V1":
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

                            mosaic_name = f"{short_name}_{layer}_{pass_id}_mosaic.tif"
                            mosaic_path = data_dir / mosaic_name
                            tmp_path = data_dir / f"tmp_{mosaic_name}"

                            # Open datasets to avoid GDALDatasetShadow errors
                            opened_datasets = []
                            for u in urls:
                                vsi_url = f"/vsicurl/{u}" if u.startswith("http") and not u.startswith("/vsi") else u
                                ds = gdal.Open(vsi_url)
                                if ds is None:
                                    logger.warning(f"GDAL failed to open URL (likely auth or missing file), skipping: {vsi_url}")
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
                                "crs": master_grid["dst_crs"]
                            }

                            # Submit Plotting Task (Skip individual layouts for rtc-rgb mode)
                            if mode != "rtc-rgb":
                                future = executor.submit(
                                    run_plotting_task,
                                    maps_dir, layouts_dir, mosaic_path, short_name, layer,
                                    pass_id, layout_date, layout_title, bbox, zoom_bbox,
                                    reclassify_snow_ice, False, benchmark_mode=(benchmark_stats is not None)
                                )
                                plotting_futures.append(future)

                            continue # Skip the xarray processing loops entirely for RTC

                        # For non-RTC products, we load the granules into xarray DataArrays for filtering and mosaicking
                        DS, conf_DS, date_DS = None, None, None

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

                            # Reproject to master grid
                            for da in ds_group:
                                grid_props = master_grid.copy()
                                dst_crs_val = grid_props.pop("dst_crs")
                                da_warped = da.rio.reproject(dst_crs_val, **grid_props, resampling=resampling_method)
                                all_warped_ds.append(da_warped)
                        
                        if not all_warped_ds: continue
                        
                        if colormap is None:
                            try:
                                colormap = get_image_colormap(DS[0])
                            except Exception:
                                colormap = None
                            
                        # Mosaic the datasets using the single global master grid setup
                        mosaic, _, nodata = mosaic_opera(all_warped_ds, product=short_name, merge_args={})

                        # Apply slope mask if it has been generated previously
                        if global_slope_mask is not None:
                            # Ensure shape compatibility
                            if mosaic.shape[-2:] == global_slope_mask.shape:
                                # Set pixels with slope < threshold to nodata
                                mosaic.values[..., global_slope_mask] = nodata
                            else:
                                logger.warning(f"Mask shape {global_slope_mask.shape} mismatches mosaic {mosaic.shape}. Skipping slope filter.")

                        # Apply coastal mask to ocean pixels
                        if global_coastal_mask is not None:
                            if mosaic.shape[-2:] == global_coastal_mask.shape:
                                # Mask out ocean (where global_coastal_mask is False)
                                mosaic.values[..., ~global_coastal_mask.values] = nodata
                            else:
                                logger.warning("Coastal mask shape mismatches mosaic. Skipping coastal filter.")

                        image = array_to_image(mosaic, colormap=colormap, nodata=nodata)
                        
                        # Create filename and full paths using pass_id (YYYYMMDDtHHMM)
                        mosaic_name = f"{short_name}_{layer}_{pass_id}_mosaic.tif"
                        mosaic_path = data_dir / mosaic_name
                        tmp_path = data_dir / f"tmp_{mosaic_name}"

                        # Save the mosaic to a temporary GeoTIFF
                        copy(image, tmp_path, driver="GTiff")
                        warp_args = {"xRes": 30, "yRes": 30, "creationOptions": ["COMPRESS=DEFLATE"]}
                        
                        # Reproject/compress using GDAL directly into the final GeoTIFF
                        gdal.Warp(str(mosaic_path), str(tmp_path), **warp_args)
                        
                        # Convert to COG (writes back into mosaic_path)
                        save_gtiff_as_cog(mosaic_path, mosaic_path)
                        
                        # Clean up tmp file
                        cleanup_temp_file(tmp_path)

                        with rasterio.open(mosaic_path) as ds:
                            mosaic_crs = ds.crs

                        # Add info to the mosiac index for pair-wise differencing
                        mosaic_index[short_name][layer][pass_id] = {
                            "path": mosaic_path, "crs": mosaic_crs
                        }

                        # --- Background Plotting ---
                        logger.info(f"Submitting background plotting task for {pass_id}...")
                        future = executor.submit(
                            run_plotting_task,
                            maps_dir, layouts_dir, mosaic_path, short_name, layer,
                            pass_id, layout_date, layout_title, bbox, zoom_bbox,
                            reclassify_snow_ice, False, benchmark_mode=(benchmark_stats is not None)
                        )
                        plotting_futures.append(future)

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
                    future = executor.submit(
                        run_rgb_task, 
                        vv_path, vh_path, rgb_path
                    )
                    plotting_futures.append(future)

        # Concurrent differencing
        if mode in ["flood", "landslide"]:
            logger.info(f"Submitting concurrent pair-wise differencing tasks ({mode})...")
            for short_name_k, layers_dict in mosaic_index.items():
                # Filter for relevant products only (RTC for Landslide, all for Flood)
                if mode == "landslide" and short_name_k != "OPERA_L2_RTC-S1_V1": continue
                
                for layer_k, date_map in layers_dict.items():
                    # dates is now a list of sorted pass_ids (YYYYMMDDtHHMM)
                    dates = sorted(date_map.keys())
                    
                    for i in range(len(dates)):
                        for j in range(i + 1, len(dates)):
                            d_early = dates[i]
                            d_later = dates[j]

                            early_info = date_map[d_early]
                            later_info = date_map[d_later]
                            
                            if early_info["crs"] != later_info["crs"]:
                                continue

                            # Setup filenames and paths
                            suffix = "water_gain.tif" if mode == "flood" else "log-diff.tif"
                            diff_name = f"{short_name_k}_{layer_k}_{d_later}_{d_early}_{suffix}"
                            diff_path = data_dir / diff_name
                            
                            diff_id_str = f"{d_later}_{d_early}"
                            diff_date_str_layout = f"{d_early}, {d_later}"

                            # Submit Pipeline Task (Compute Diff -> Map -> Layout)
                            future = executor.submit(
                                run_difference_pipeline,
                                early_info["path"], later_info["path"], diff_path, mode,
                                maps_dir, layouts_dir, short_name_k, layer_k,
                                diff_id_str, diff_date_str_layout, layout_title, bbox, zoom_bbox,
                                reclassify_snow_ice
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