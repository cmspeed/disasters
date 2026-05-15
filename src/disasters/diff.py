import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import rioxarray
import xarray as xr
from osgeo import gdal

logger = logging.getLogger(__name__)


def save_gtiff_as_cog(src_path: Path, dst_path: Optional[Path] = None) -> None:
    """
    Save a GeoTIFF as a Cloud Optimized GeoTIFF (COG).

    Args:
        src_path (Path): Path to the source file.
        dst_path (Path, optional): Path for destination. Edits in-place if None.
    
    Raises:
        RuntimeError: If the source file cannot be opened.
    """
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


def compute_and_write_difference(
    earlier_path: Path,
    later_path: Path,
    out_path: Path,
    nodata_value: Optional[float] = None,
    log: bool = False,
) -> None:
    """
    Create a difference raster for 'flood' or 'landslide' mode.
    Writes a uint8 categorical raster with an embedded color palette for DSWx products.

    Args:
        earlier_path (Path): Path to the earlier raster.
        later_path (Path): Path to the later raster.
        out_path (Path): Output difference raster path.
        nodata_value (float, optional): Custom no-data value.
        log (bool): Flag to compute log difference for RTC-S1.
    """
    # Open rasters and apply mask for nodata handling
    da_later = rioxarray.open_rasterio(later_path, masked=True)
    da_early = rioxarray.open_rasterio(earlier_path, masked=True)

    try:
        # Determine the nodata value to use (nd) from inputs
        nd = nodata_value
        if nd is None:
            nd = da_later.rio.nodata
            if nd is None:
                nd = da_early.rio.nodata

        # Landslide mode
        if log:
            # Compute log difference for RTC-S1
            L = da_later.where(da_later > 0)
            E = da_early.where(da_early > 0)

            diff = 10 * np.log10(L / E)
            diff = diff.astype("float32")
            logger.info("Computed log difference for RTC-S1.")
            
            diff.attrs.clear()
            diff.attrs["DESCRIPTION"] = "Log Ratio Difference (Later / Earlier) for RTC-S1"

            # Handle Nodata
            input_nodata_mask = xr.where(da_later.isnull() | da_early.isnull(), True, False)
            result_nan_mask = np.isnan(diff)
            result_inf_mask = np.isinf(diff)
            final_nodata_mask = input_nodata_mask | result_nan_mask | result_inf_mask

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
            logger.info("Computing categorical transition codes for DSWx...")
            
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

            logger.info(f"Categorical difference written to {out_path}")
    finally:
        da_later.close()
        da_early.close()


def compute_and_write_difference_positive_change_only(
    earlier_path: Path,
    later_path: Path,
    out_path: Path,
) -> None:
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

    Args:
        earlier_path (Path): Path to the earlier raster.
        later_path (Path): Path to the later raster.
        out_path (Path): Output difference raster path.
    """
    logger.info(f"Computing generalized positive change (gain) layer for {out_path.name}...")

    ds_later = rioxarray.open_rasterio(later_path, masked=False)
    da_later = ds_later.squeeze()
    
    ds_early = rioxarray.open_rasterio(earlier_path, masked=False)
    da_early = ds_early.squeeze()

    try:
        # Define Groups
        # Valid Water Classes across S1 and HLS WTR/BWTR:
        # 1: Open Water (S1/HLS)
        # 2: Partial Surface Water (HLS)
        # 3: Inundated Vegetation (S1)
        any_water = [1, 2, 3]
        partial_or_veg = [2, 3]
        open_water = 1
        not_water = 0

        # Create masks 
        # Nodata/Masks: Values >= 250 are considered invalid/masked in all DSWx products for differencing purposes
        #(250=HAND, 251=Layover, 252=Snow/Ice, 253=Cloud, 254=Ocean, 255=Fill)
        mask_invalid = (da_early >= 250) | (da_later >= 250)

        # Condition 1: New Water (0 -> 1, 2, 3)
        cond_new_water = (da_early == not_water) & (np.isin(da_later, any_water))

        # Condition 2: Intensification (2, 3 -> 1), Partial/Veg becoming Open Water
        cond_intensification = (np.isin(da_early, partial_or_veg)) & (da_later == open_water)

        # Combine Gain Conditions
        mask_gain = cond_new_water | cond_intensification

        # Create Output Array (uint8) initally all zeros
        out_data = np.zeros_like(da_early.values, dtype="uint8")
        
        # Apply Water Gain (Set to 1)
        out_data[mask_gain.values] = 1
        
        # Apply Nodata (Set to 255) - Overwrites any previous assignment
        out_data[mask_invalid.values] = 255

        # Wrap in xarray for CRS/Transform handling
        da_out = xr.DataArray(
            out_data,
            coords=da_later.coords,
            dims=da_later.dims,
            attrs=da_later.attrs
        )
        da_out.rio.write_crs(da_later.rio.crs, inplace=True)
        da_out.rio.write_nodata(255, inplace=True)

        # Write to Temporary GeoTIFF
        tmp_gtiff = out_path.with_suffix(".tmp.tif")
        da_out.rio.to_raster(tmp_gtiff, compress="DEFLATE", tiled=True, dtype="uint8")

        # Add a Colormap and Metadata
        # Color 0: White (255, 255, 255, 255)
        # Color 1: Blue  (0, 0, 200, 255)
        custom_colormap = {
            0: (255, 255, 255, 255),
            1: (0, 0, 200, 255)
        }

        # Define class names for metadata
        class_names = {
            0: "No Change or Water Loss",
            1: "Water Gain"
        }

        with rasterio.open(tmp_gtiff, 'r+') as dst:
            dst.write_colormap(1, custom_colormap)
            tags = {f"CLASS_{k}": v for k, v in class_names.items()}
            dst.update_tags(**tags)

        # Convert to COG
        save_gtiff_as_cog(tmp_gtiff, out_path)

        # Cleanup
        try:
            tmp_gtiff.unlink(missing_ok=True)
        except:
            pass

        logger.info(f"Positive change difference written to {out_path}")
        return
    finally:
        ds_later.close()
        ds_early.close()


def compute_and_write_max_flood_extent(
    input_paths: list[Path],
    out_path: Path,
) -> None:
    """
    Computes a cumulative 'Maximum Flood Extent' raster from a time-series of DSWx mosaics.
    
    Logic (Assign 1 - Blue):
      - If the pixel was EVER classified as water (1=Open, 2=Partial, 3=Inundated Veg)
        across any of the time slices.
    
    Logic (Assign 0 - White):
      - If the pixel was NEVER water, but had valid observations (0) in at least one slice.
      
    Logic (Assign 255 - NoData):
      - If the pixel was masked/invalid (>= 250) across ALL time slices.

    Args:
        input_paths (list[Path]): Chronological list of raster paths to process.
        out_path (Path): Output destination for the cumulative COG.
    """
    logger.info(f"Computing maximum flood extent for {out_path.name}...")

    any_water_mask = None
    ever_valid_mask = None
    
    base_x = None
    base_y = None
    base_crs = None
    base_transform = None

    water_classes = [1, 2, 3]

    for i, path in enumerate(input_paths):
        with rioxarray.open_rasterio(path, masked=False) as da:
            # Extract raw numpy arrays
            val = da.squeeze().values.copy()
            
            if i == 0:
                base_x = da.x.values.copy()
                base_y = da.y.values.copy()
                base_crs = da.rio.crs
                base_transform = da.rio.transform()
                any_water_mask = np.zeros_like(val, dtype=bool)
                ever_valid_mask = np.zeros_like(val, dtype=bool)

            valid_mask = val < 250
            water_mask = np.isin(val, water_classes)

            ever_valid_mask |= valid_mask
            any_water_mask |= (valid_mask & water_mask)

    # Construct the final array
    out_data = np.zeros_like(any_water_mask, dtype="uint8")
    out_data[ever_valid_mask] = 0 
    out_data[any_water_mask] = 1  
    out_data[~ever_valid_mask] = 255 

    # Build DataArray entirely from scratch in memory
    da_out = xr.DataArray(
        out_data,
        coords={"y": base_y, "x": base_x},
        dims=["y", "x"]
    )
    da_out.rio.write_crs(base_crs, inplace=True)
    da_out.rio.write_nodata(255, inplace=True)
    da_out.rio.write_transform(base_transform, inplace=True)

    # Write to Temporary GeoTIFF
    tmp_gtiff = out_path.with_suffix(".tmp.tif")
    da_out.rio.to_raster(tmp_gtiff, compress="DEFLATE", tiled=True, dtype="uint8")

    # Add a Colormap and Metadata
    custom_colormap = {
        0: (255, 255, 255, 255), # White (Not Water)
        1: (0, 0, 200, 255)      # Blue (Maximum Flood Extent)
    }
    class_names = {
        0: "Never Flooded",
        1: "Maximum Flood Extent"
    }

    with rasterio.open(tmp_gtiff, 'r+') as dst:
        dst.write_colormap(1, custom_colormap)
        tags = {f"CLASS_{k}": v for k, v in class_names.items()}
        dst.update_tags(**tags)

    # Convert to COG
    save_gtiff_as_cog(tmp_gtiff, out_path)

    try:
        tmp_gtiff.unlink(missing_ok=True)
    except:
        pass

    logger.info(f"Maximum flood extent written to {out_path}")


def create_rtc_rgb_visualization(vv_path: Path | str, vh_path: Path | str, out_path: Path | str) -> None:
    """
    Creates an 8-bit 3-band RGB composite from Float32 VV and VH mosaics using windowed processing.
    Red: sqrt(VV) scaled [0.14, 0.52]
    Green: sqrt(VH) scaled [0.05, 0.259]
    Blue: sqrt(VV) scaled [0.14, 0.52]

    Args:
        vv_path (Path | str): Path to the VV band raster.
        vh_path (Path | str): Path to the VH band raster.
        out_path (Path | str): Output path for the RGB composite GeoTIFF.
    """
    import numpy as np
    import rasterio
    from rasterio.windows import Window
    
    # Helper function to stretch values to 8-bit
    def stretch_to_8bit(arr, vmin, vmax):
        # Prevent NaNs by forcing a tiny floor.
        with np.errstate(invalid='ignore'):
            arr_safe = np.maximum(arr, 1e-6)
            arr_sqrt = np.sqrt(arr_safe)
        
        # Apply the linear stretch formula.
        stretched = (arr_sqrt - vmin) / (vmax - vmin) * 255
        
        # Handle valid but very low values.
        stretched_clamped = np.clip(stretched, 1, 255)
        
        # Handle Nodata
        np.nan_to_num(stretched_clamped, copy=False, nan=0.0)
        
        return stretched_clamped.astype(np.uint8)

    with rasterio.open(vv_path) as src_vv, rasterio.open(vh_path) as src_vh:
        profile = src_vv.profile.copy()
        
        # Update profile for an 8-bit, 3-band RGB image
        profile.update(
            dtype=rasterio.uint8,
            count=3,
            nodata=0,
            compress='deflate',
            photometric='RGB'
        )

        with rasterio.open(out_path, 'w', **profile) as dst:
            # Process block by block to prevent memory crashes
            for block_index, window in src_vv.block_windows(1):
                vv_data = src_vv.read(1, window=window)
                vh_data = src_vh.read(1, window=window)

                # Compute stretches
                # Red and Blue use VV [0.14, 0.52]
                red_blue = stretch_to_8bit(vv_data, 0.14, 0.52)
                # Green uses VH [0.05, 0.259]
                green = stretch_to_8bit(vh_data, 0.05, 0.259)

                # Write bands (1=Red, 2=Green, 3=Blue)
                dst.write(red_blue, 1, window=window)
                dst.write(green, 2, window=window)
                dst.write(red_blue, 3, window=window)
                
    # Convert standard GeoTIFF to Cloud Optimized GeoTIFF (COG)
    save_gtiff_as_cog(out_path, out_path)
