import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def read_opera_metadata(output_dir: Path) -> pd.DataFrame:
    """
    Read the OPERA products metadata file (Excel or CSV) and clean the 'Start Time' column.

    Args:
        output_dir (Path): Path to the directory containing the metadata file.

    Returns:
        pd.DataFrame: DataFrame with 'Start Time' as datetime64[ns].

    Raises:
        FileNotFoundError: If the metadata file does not exist.
    """
    excel_path = output_dir / "opera_products_metadata.xlsx"
    csv_path = output_dir / "opera_products_metadata.csv"
    
    # Read the file into a Pandas DataFrame
    if excel_path.exists():
        df = pd.read_excel(excel_path)
        logger.info(f"Loaded {len(df)} rows from {excel_path}")
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from {csv_path}")
    else:
        raise FileNotFoundError(f"Metadata file not found at {output_dir}")

    # Define the two format strings (necessary as RTC has a slightly different format)
    FORMAT_MICROSECONDS = "%Y-%m-%dT%H:%M:%S.%fZ"  # For non-RTC data
    FORMAT_SECONDS_ONLY = "%Y-%m-%dT%H:%M:%SZ"     # For RTC data

    # Ensure the column is treated as a string before parsing. 
    start_times = df["Start Time"].astype(str)

    # Parse the non-RTC format (RTC dates become NaT)
    df_temp1 = pd.to_datetime(
        start_times, format=FORMAT_MICROSECONDS, errors="coerce"
    )
    
    # Parse the RTC format (Non-RTC dates become NaT)
    df_temp2 = pd.to_datetime(
        start_times, format=FORMAT_SECONDS_ONLY, errors="coerce"
    )

    # Combine differently parsed datetimes into a single column
    df["Start Time"] = df_temp1.combine_first(df_temp2)

    granule_col = 'Granule ID' if 'Granule ID' in df.columns else 'Granule'
    
    if granule_col in df.columns:
        # Create a "Scene_ID" by removing the Processing Date.
        df['Scene_ID'] = df[granule_col].str.replace(
            r'(_\d{8}T\d{6}Z)(_\d{8}T\d{6}Z)', 
            r'\1', 
            regex=True
        )

        # Sort alphabetically by Granule ID string (chronologically, newest processing date at the bottom).
        df = df.sort_values(granule_col)

        # Drop duplicates based on the Scene ID, keeping the 'last' (newest ProcessingTime).
        original_len = len(df)
        df = df.drop_duplicates(subset=['Scene_ID'], keep='last')
        dropped_count = original_len - len(df)
        
        if dropped_count > 0:
            logger.info(f"Deduplicated {dropped_count} ghost granule(s) by keeping the newest processing dates.")

        # Clean up the temp column
        df = df.drop(columns=['Scene_ID'])
    else:
        logger.warning(f"Could not find Granule ID column. Skipping deduplication.")

    return df


def fetch_missing_dems(bbox: list, local_dir: Path) -> None:
    """
    Queries Earthdata for recent DSWx-HLS granules covering the bbox 
    and downloads ONLY their _B10_DEM.tif files to the local directory.
    """
    import datetime
    import earthaccess
    import logging
    
    logger.info("[DEM Fetcher] Missing local DEMs detected. Querying Earthdata for static topography...")
    
    try:
        # Repackage our [S, N, W, E] bbox into Earthaccess format: (W, S, E, N)
        s, n, w, e = bbox
        cmr_bbox = (w, s, e, n)
        
        # Query Earthdata for recent DSWx-HLS granules covering the bbox (last 60 days)
        end_date = datetime.datetime.now(datetime.timezone.utc)
        start_date = end_date - datetime.timedelta(days=60)
        
        results = earthaccess.search_data(
            short_name="OPERA_L3_DSWX-HLS_V1",
            bounding_box=cmr_bbox,
            temporal=(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
            count=20 # Grab enough to cover the bbox footprint
        )
        
        if not results:
            logger.warning("[DEM Fetcher] No recent DSWx-HLS granules found for this BBOX.")
            return
        
        # Filter to get only the _B10_DEM URLs
        dem_urls = []
        for granule in results:
            for link in granule.data_links():
                if "_B10_DEM.tif" in link:
                    dem_urls.append(link)
                    
        if not dem_urls:
            logger.warning("[DEM Fetcher] Found granules, but no _B10_DEM.tif links.")
            return
        
        # Remove duplicates
        dem_urls = list(set(dem_urls))
        
        logger.info(f"[DEM Fetcher] Downloading {len(dem_urls)} DEM layers to {local_dir}...")
        earthaccess.download(dem_urls, local_path=str(local_dir))
        logger.info("[DEM Fetcher] Topography download complete.")
        
    except Exception as e:
        logger.error(f"[DEM Fetcher] Failed to fetch missing DEMs: {e}")


def cluster_by_time(df: pd.DataFrame, time_col: str = "Start Time", threshold_minutes: int = 120) -> list:
    """
    Groups dataframe rows by time clustering to separate passes (e.g. Ascending vs Descending).

    Args:
        df (pd.DataFrame): Dataframe to sort and group.
        time_col (str, optional): Column name containing datetime objects. Defaults to "Start Time".
        threshold_minutes (int, optional): Threshold difference to split groups. Defaults to 120.

    Returns:
        list: List of dataframe groups.
    """
    df = df.sort_values(time_col)
    groups = []
    if df.empty:
        return groups
    
    current_group = [df.iloc[0]]
    
    # Iterate through rows starting from the second one
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # Calculate time difference in minutes
        time_diff = (row[time_col] - prev_row[time_col]).total_seconds() / 60
        
        if time_diff <= threshold_minutes:
            current_group.append(row)
        else:
            # Finalize previous group and start a new one
            groups.append(pd.DataFrame(current_group))
            current_group = [row]
    
    # Append the last group
    if current_group:
        groups.append(pd.DataFrame(current_group))
        
    return groups


def get_S1_orbit_direction(urls: list, username: str = None, password: str = None) -> str:
    """
    Reads metadata of the first available OPERA DSWx-S1 URL to extract the orbit pass direction.
    Returns 'A' for ascending, 'D' for descending, or '' if not found.
    """
    if not urls: return ""
    
    from opera_utils.disp._remote import open_file
    import rasterio
    
    url = urls[0]
    try:
        if url.startswith("http") and not url.startswith("/vsi"):
            f = open_file(url, earthdata_username=username, earthdata_password=password)
            with rasterio.open(f) as ds:
                tags = ds.tags()
        else:
            with rasterio.open(url) as ds:
                tags = ds.tags()
                
        direction = tags.get("RTC_ORBIT_PASS_DIRECTION", tags.get("ORBIT_PASS_DIRECTION", "")).lower()
        
        if "ascending" in direction: return "A"
        if "descending" in direction: return "D"
    except Exception as e:
        logger.warning(f"Failed to read flight direction from {url}: {e}")
        
    return ""