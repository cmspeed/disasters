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

    return df


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