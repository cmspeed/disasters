import os
import argparse
from pathlib import Path
import pandas as pd
from osgeo import gdal
import rasterio
import rioxarray
# import leafmap

def parse_arguments():
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
        "WTR",
        "BWTR",
        "VEG-ANOM-MAX",
        "VEG-DIST-STATUS"
    ]

    valid_modes = [
        "flood",
        "fire",
        "earthquake"
    ]

    parser.add_argument(
        "-b", "--bbox", nargs=4, type=float, metavar=("S", "N", "W", "E"),
        required=True, help="Bounding box in the form: South North West East"
    )

    parser.add_argument(
        "-s", "--satellite", type=str, required = False, default="all",
        help="Which satellites to include. Use 'all' for all satellites."
    )

    parser.add_argument(
        "-o", "--output_dir", type=Path, required=True,
        help="Path to the directory where results and metadata will be saved."
    )

    parser.add_argument(
        "-sn", "--short_name", type=str, required=False, choices=valid_short_names,
        help="Short name to filter the DataFrame (must be one of the known OPERA products)"
    )

    parser.add_argument(
        "-l", "--layer_name", type=str, required=False, choices=valid_layer_names,
        help="Layer name to extract from metadata (e.g., 'WTR', 'BWTR', 'VEG-ANOM-MAX')"
    )

    parser.add_argument(
        "-d", "--date", type=str, required=False,
        help="Date string (YYYY-MM-DD) to filter rows by Start Date"
    )

    parser.add_argument(
        "-n", "--number_of_dates", required=False, type=int, default=5,
        help="Number of most recent dates to consider for OPERA products"
    )

    parser.add_argument(
        "-m", "--mode", type=str, required=False, default="flood", choices=valid_modes,
        help="Mode of operation: flood, fire, earthquake. Default is 'flood'."
    )

    return parser.parse_args()

def authenticate():
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
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Created or reused output directory: {output_dir}")
    except Exception as e:
        print(f"[ERROR] Could not create output directory: {e}")
        raise

def read_opera_metadata_csv(output_dir):
    csv_path = output_dir / "opera_products_metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} rows from {csv_path}")
    return df

def compile_and_load_data(layer_links):
    from opera_utils.disp._remote import open_file
    from collections import Counter

    username, password = authenticate()

    DS = []
    for link in layer_links:
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

    return DS

def generate_products(df_opera, mode, mode_dir):
    import sys
    sys.path.insert(0, "/u/trappist-r0/colespeed/work/opera_mosaic/")  # adjust this path
    import opera_mosaic
    from rasterio.shutil import copy

    if mode == "flood":
        short_names = ["OPERA_L3_DSWX-HLS_V1", "OPERA_L3_DSWX-S1_V1"]
        layer_names = ["WTR", "BWTR"]
    elif mode == "fire":
        short_names = ["OPERA_L3_DIST-ALERT-HLS_V1", "OPERA_L3_DIST-ALERT-S1_V1"]
        layer_names = ["VEG-ANOM-MAX", "VEG-DIST-STATUS"]
    elif mode == "earthquake":
        short_names = ["OPERA_L2_RTC-S1_V1", "OPERA_L2_CSLC-S1_V1", "OPERA_L3_DISP-S1_V1"]
        layer_names = ["DISP", "CSLC", "WTR"]  # Placeholder — update with actual layers

    df_opera['Start Time'] = pd.to_datetime(df_opera['Start Time'], errors='coerce')
    df_opera['Start Date'] = df_opera['Start Time'].dt.date.astype(str)
    unique_dates = df_opera['Start Date'].dropna().unique()

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

                # Do your processing here:
                print(f"[INFO] Processing {short_name} - {layer} on {date}")
                print(f"Found {len(urls)} URLs")

                # Compile and load data
                DS = compile_and_load_data(urls)

                # Mosaic the datasets using the appropriate method/rule
                mosaic, colormap, nodata = opera_mosaic.mosaic_opera(DS, product=short_name, merge_args={})
                image = opera_mosaic.array_to_image(mosaic, colormap=colormap, nodata=nodata)

                # Create filename and full paths
                mosaic_name = f"{short_name}_{layer}_{date}_mosaic.tif"
                mosaic_path = mode_dir / mosaic_name
                tmp_path = mode_dir / f"tmp_{mosaic_name}"

                # Save the mosaic to the mode directory
                copy(image, mosaic_path, driver='GTiff')

                # Reproject/compress using GDAL
                gdal.Warp(
                    tmp_path,
                    mosaic_path,
                    xRes=30,
                    yRes=30,
                    creationOptions=["COMPRESS=DEFLATE"]
                )

                # Overwrite original with compressed version
                os.replace(tmp_path, mosaic_path)

                print(f"[INFO] Mosaic written to: {mosaic_path}")

    return

def main():
    import next_pass

    args = parse_arguments()
    
    make_output_dir(args.output_dir)

    bbox = args.bbox
    satellite = args.satellite
    number_of_dates = args.number_of_dates

    # Call next_pass to generate csv of relevant opera products
    output_dir = next_pass.run_next_pass(bbox, satellite, number_of_dates)
    dest = args.output_dir / output_dir.name

    output_dir.rename(dest)
    print(f"[INFO] Moved next_pass output directory to {dest}")

    # Read the metadata CSV file
    df_opera = read_opera_metadata_csv(dest)
    mode = args.mode

    # Make a new directory with the mode name
    mode_dir = args.output_dir / mode
    make_output_dir(mode_dir)
    print(f"[INFO] Created mode directory: {mode_dir}")

    # Generate products based on the mode
    generate_products(df_opera, mode, mode_dir)
    
if __name__ == "__main__":
    main()