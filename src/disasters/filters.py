import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pygmt
import xarray as xr
from osgeo import gdal
from rasterio.enums import Resampling

from .mosaic import get_image_colormap

logger = logging.getLogger(__name__)


def reclassify_snow_ice_as_water(DS: list, conf_DS: list) -> Tuple[list, Optional[dict]]:
    """
    Reclassify false snow/ice positives (value 252) as water (value 1) based on the confidence layers. Only applicable for DSWx-HLS.

    Args:
        DS (list): List of rioxarray datasets (BWTR layers).
        conf_DS (list): List of rioxarray datasets (CONF layers).

    Returns:
        tuple: List of updated rioxarray datasets with 252 reclassified as 1, and the colormap.
    
    Raises:
        ValueError: If conf_DS is missing or if lists do not match in length.
    """
    if conf_DS is None:
        raise ValueError("conf_DS must not be None when reclassifying snow/ice.")

    if len(DS) != len(conf_DS):
        raise ValueError("DS and conf_DS must be the same length.")

    values_to_reclassify = [1, 3, 4, 21, 23, 24]

    try:
        colormap = get_image_colormap(DS[0])
        logger.info("Colormap successfully retrieved and will be used in reclassified output")
    except Exception:
        logger.info("Unable to get colormap")
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
    DS: list,
    DS_dates: list,
    date_threshold: int,
    DS_conf: Optional[list] = None,
    confidence_threshold: Optional[float] = None,
    fill_value: Optional[float] = None,
) -> Tuple[list, Optional[dict]]:
    """
    Filters each data xarray in `DS` based on date threshold and optional confidence threshold.
    Pixels not meeting the criteria are set to `fill_value`. If `fill_value` is None, defaults to da_data.rio.nodata or NaN.

    Args:
        DS (list): List of data granules (e.g., VEG-DIST-STATUS tiles).
        DS_dates (list): List of corresponding date granules.
        date_threshold (int): Pixels with dates >= this value are retained.
        DS_conf (list, optional): List of confidence rasters corresponding to `DS`. Default is None.
        confidence_threshold (float, optional): Pixels with confidence >= this value are retained.
        fill_value (number, optional): Value to fill where condition is not met.

    Returns:
        tuple: list of xr.DataArray filtered data granules, and the original colormap.
    """
    assert len(DS) == len(DS_dates), "DS and DS_dates must be same length"
    if DS_conf is not None:
        assert len(DS_conf) == len(DS), "DS_conf must match DS in length"

    try:
        colormap = get_image_colormap(DS[0])
        logger.info("Colormap successfully retrieved and will be used in reclassified output")
    except Exception:
        logger.info("Unable to get colormap")
        colormap = None

    filtered_list = []

    for i, (da_data, da_date) in enumerate(zip(DS, DS_dates)):
        logger.info(f"Filtering granule {i + 1}/{len(DS)}")

        # Create a mask that excludes "No Data" and "No Disturbance" values
        valid_data_mask = da_data != 0

        # Create date threshold mask
        date_threshold_mask = da_date >= date_threshold

        # Combine masks
        date_mask = valid_data_mask & date_threshold_mask
        
        # Optional confidence mask
        if DS_conf is not None and confidence_threshold is not None:
            conf_layer = DS_conf[i]
            logger.info(f"Confidence layer shape: {conf_layer.shape}")
            total_pixels = conf_layer.size

            # Construct confidence mask based on confidence_threshold
            conf_mask = conf_layer >= confidence_threshold

            retained_pixels = conf_mask.sum().item()
            logger.info(f"Confidence retained: {retained_pixels} / {total_pixels} ({retained_pixels / total_pixels:.2%})")

            max_retained_conf = conf_layer.where(conf_mask).max().item()
            logger.info(f"Max confidence among retained pixels: {max_retained_conf}")

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


def compute_date_threshold(date_str: str) -> int:
    """
    Compute the date threshold in days from a reference date (2020-12-31).

    Args:
        date_str (str): Date string in the format YYYY-MM-DD.

    Returns:
        int: Number of days from the reference date to the target date.
    """
    # Define the reference date and the target date
    reference_date = datetime(2020, 12, 31)
    target_date = datetime.strptime(date_str, "%Y-%m-%d")

    # Calculate the difference between the two dates
    delta = target_date - reference_date

    # Get the number of days from the timedelta object
    return delta.days


def generate_coastal_mask(bbox: list, master_grid: dict) -> Optional[xr.DataArray]:
    """
    Generates a coastal mask using PyGMT's global coastline database.

    Args:
        bbox (list): Bounding box in the form [South, North, West, East].
        master_grid (dict): Dictionary with 'crs', 'shape', 'transform'.

    Returns:
        xr.DataArray: Boolean xarray DataArray where True = Land/Inland Water, False = Ocean.
    """
    logger.info("Generating global coastal water mask using PyGMT...")
    
    # bbox is [South, North, West, East]
    # pygmt region is [xmin, xmax, ymin, ymax]
    region = [bbox[2], bbox[3], bbox[0], bbox[1]]

    try:
        # mask_values=[ocean, land, lake, island, pond]
        # 0 for ocean, 1 for everything else to preserve inland water
        mask_geo = pygmt.grdlandmask(
            region=region,
            spacing='30e',  
            maskvalues=[0, 1, 1, 1, 1],
            resolution='f'
        )
        
        # Assign CRS so rioxarray can reproject it
        mask_geo = mask_geo.rio.write_crs("EPSG:4326")

        # Reproject to align exactly with the master UTM grid
        mask_utm = mask_geo.rio.reproject(
            master_grid['dst_crs'],
            shape=master_grid['shape'],
            transform=master_grid['transform'],
            resampling=Resampling.nearest
        )
        
        # Squeeze to 2D boolean mask (True for valid land/inland water)
        return mask_utm.squeeze() == 1

    except Exception as e:
        logger.warning(f"Failed to generate PyGMT coastal mask: {e}")
        return None


def process_dem_and_slope(df: pd.DataFrame, master_grid: dict, threshold: float, output_dir: Path, skip_existing: bool = True):
    """
    Fetches all DSWx-HLS Band 10 URLs (or downloads them if missing) and mosaics them 
    into 'dem.tif' saved at output_dir. Calculates slope and returns a boolean mask 
    (True where slope < threshold).

    Args:
        df (pd.DataFrame): DataFrame containing OPERA products metadata.
        master_grid (dict): Dictionary with 'shape' and 'transform'.
        threshold (float): Slope threshold in degrees.
        output_dir (Path): Output directory for the resulting masks.
        skip_existing (bool): Whether to skip GDAL processing if slope.tif already exists.

    Returns:
        np.ndarray: Mask indicating areas where slope is below threshold.
    """
    from .catalog import fetch_missing_dems
    from .io import scan_local_directory
    from .pipeline import get_local_spatial_properties

    logger.info(f"[Filters] Processing DEM and generating slope mask (> {threshold} degrees)...")
    
    dem_output_path = output_dir / "dem.tif"
    slope_output_path = output_dir / "slope.tif"

    # Skip Processing if slope.tif already exists
    if skip_existing and slope_output_path.exists():
        logger.info(f"Slope mask already exists on disk, skipping DEM download/processing.")
        try:
            import rasterio
            with rasterio.open(slope_output_path) as src:
                slope_arr = src.read(1)
            
            mask = (slope_arr < threshold) & (slope_arr != -9999)
            logger.info(f"Loaded existing slope mask. Masking {np.sum(mask)} pixels < {threshold}°.")
            return mask
        except Exception as e:
            logger.warning(f"Failed to read existing slope mask: {e}. Proceeding to recompute...")

    # Check if we have DEM files locally or DSWx-HLS cloud links
    all_paths = []
    if 'Filepath' in df.columns:
        all_paths.extend(df['Filepath'].dropna().astype(str).tolist())
    for col in df.columns:
        if str(col).startswith('Download URL'):
                all_paths.extend(df[col].dropna().astype(str).tolist())

    has_dem_files = any('_B10_DEM' in p for p in all_paths)
    has_dswx_cloud = 'Dataset' in df.columns and df['Dataset'].str.contains('DSWx-HLS', na=False).any()

    # If there are no local DEMs and no cloud DSWx links, download them
    if not has_dem_files and not has_dswx_cloud:
        logger.info("[Filters] No DEM data found. Initiating dynamic DEM fetcher...")
        
        from .pipeline import get_local_spatial_properties
        from .catalog import fetch_missing_dems
        
        auto_bbox, _ = get_local_spatial_properties(df)
        local_dir = Path(all_paths[0]).parent
        
        fetch_missing_dems(auto_bbox, local_dir)
        
        # Inject newly downloaded DEMs into our path list
        new_dems = [str(p) for p in local_dir.glob("*_B10_DEM*.tif")]
        if new_dems:
            all_paths.extend(new_dems)
        else:
            logger.warning("[Filters] Earthdata fetch completed, but no DEMs were found on disk.")
            return None
            
    # Gather the final list of DEM paths for GDAL processing
    explicit_dems = [p for p in all_paths if '_B10_DEM' in p]
    
    if explicit_dems:
        dem_urls = explicit_dems
    else:
        # Fallback for cloud mode: deduce DEM URLs from DSWx WTR URLs
        dswx_rows = df[df['Dataset'] == 'OPERA_L3_DSWX-HLS_V1']
        for url in dswx_rows['Download URL WTR'].dropna().unique():
            if '_B01_WTR' in url:
                if url.startswith('http') and not url.startswith('/vsi'):
                    dem_url = url.replace('_B01_WTR', '_B10_DEM')
                    dem_urls.append(f'/vsicurl/{dem_url}')
                else:
                    local_dem_path = url.replace('_B01_WTR', '_B10_DEM')
                    dem_urls.append(local_dem_path)
                    
    dem_urls = list(set(dem_urls))
    
    if not dem_urls:
        logger.warning("[Filters] Failed to identify or fetch any DEM URLs. Skipping slope masking.")
        return None

    # Extract Master Grid Properties
    height, width = master_grid['shape'] 
    transform = master_grid['transform']
    
    # Calculate bounds (minX, minY, maxX, maxY)
    min_x = transform.c
    max_y = transform.f
    max_x = min_x + (transform.a * width)
    min_y = max_y + (transform.e * height)
    
    output_bounds = [min_x, min_y, max_x, max_y]
    dst_crs = master_grid.get('dst_crs')

    try:
        # Warp DEMs to Disk (dem.tif), matching the master grid resolution and bounds
        warp_options = gdal.WarpOptions(
            format='GTiff',
            outputBounds=output_bounds,
            width=width,
            height=height,
            dstSRS=dst_crs,
            resampleAlg='bilinear',
            dstNodata=-9999
        )
        
        logger.info(f"Writing DEM mosaic to: {dem_output_path}")
        dem_ds = gdal.Warp(str(dem_output_path), dem_urls, options=warp_options)
        
        if dem_ds is None:
            logger.warning("DEM Warp failed.")
            return None

        # Calculate Slope (In-Memory from the DEM dataset we just created)
        slope_options = gdal.DEMProcessingOptions(
            format='GTiff', 
            computeEdges=True,
            slopeFormat='degree'
        )
        
        # Compute and write slope to slope.tif
        slope_ds = gdal.DEMProcessing(str(slope_output_path), dem_ds, 'slope', options=slope_options)
        slope_arr = slope_ds.ReadAsArray()
        
        # Create Mask: True where slope < threshold (and valid data)
        mask = (slope_arr < threshold) & (slope_arr != -9999)
        
        logger.info(f"Slope mask generated. Masking {np.sum(mask)} pixels < {threshold}°.")
        
        # Clean up GDAL handles
        dem_ds = None 
        slope_ds = None
        
        return mask

    except Exception as e:
        logger.warning(f"Slope processing failed: {e}")
        return None
    

def apply_slope_mask_to_raster(target_tif: Path, slope_tif: Path, threshold: float, output_tif: Path):
    """
    Dynamically reprojects the slope mask to match the target raster's grid, 
    applies the mask, and saves the filtered output.
    """
    import rasterio
    import numpy as np
    from rasterio.warp import reproject, Resampling
    import logging
    
    logger = logging.getLogger(__name__)

    try:
        with rasterio.open(target_tif) as src:
            target_meta = src.meta.copy()
            target_arr = src.read(1)
            target_crs = src.crs
            target_transform = src.transform
            nodata_val = src.nodata if src.nodata is not None else -9999

        with rasterio.open(slope_tif) as slope_src:
            # Create an empty array matching the target raster's exact shape
            aligned_slope_arr = np.empty_like(target_arr, dtype=np.float32)
            
            # Reproject the slope data dynamically into the target's grid space
            reproject(
                source=rasterio.band(slope_src, 1),
                destination=aligned_slope_arr,
                src_transform=slope_src.transform,
                src_crs=slope_src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear
            )

        # Apply the mask: If slope >= threshold OR slope is nodata, set target pixel to nodata
        filtered_arr = np.where(
            (aligned_slope_arr >= threshold) | (aligned_slope_arr == -9999), 
            nodata_val, 
            target_arr
        )

        # Ensure the output meta has a defined nodata value and uses compression
        target_meta.update({
            "driver": "COG",
            "compress": "deflate",
            "nodata": nodata_val
        })

        with rasterio.open(output_tif, 'w', **target_meta) as dst:
            dst.write(filtered_arr, 1)
            
    except Exception as e:
        logger.error(f"Failed to apply slope mask to {target_tif.name}: {e}")