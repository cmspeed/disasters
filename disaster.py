import os
import argparse
from pathlib import Path
import pandas as pd
from osgeo import gdal
import rasterio
import rioxarray
import leafmap

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
    from opera_utils import open_file
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
    ]

    parser.add_argument(
        "-b", "--bbox", nargs=4, type=float, metavar=("S", "N", "W", "E"),
        required=True, help="Bounding box in the form: South North West East"
    )

    parser.add_argument(
        "-s", "--satellite", type=str, default="all",
        help="Which satellites to include. Use 'all' for all satellites."
    )

    parser.add_argument(
        "-o", "--output_dir", type=Path, required=True,
        help="Path to the directory where results and metadata will be saved."
    )

    parser.add_argument(
        "--short_name", type=str, required=True, choices=valid_short_names,
        help="Short name to filter the DataFrame (must be one of the known OPERA products)"
    )

    parser.add_argument(
        "--layer_name", type=str, required=True, choices=valid_layer_names,
        help="Layer name to extract from metadata (e.g., 'WTR', 'BWTR', 'VEG-ANOM-MAX')"
    )

    parser.add_argument(
        "--date", type=str, required=True,
        help="Date string (YYYY-MM-DD) to filter rows by Start Date"
    )

    return parser.parse_args()

def main():
    import next_pass

    args = parse_arguments()
    
    make_output_dir(args.output_dir)

    bbox = args.bbox
    satellite = args.satellite
    disaster_name = args.output_dir

    # Now call next_pass as a library
    output_dir = next_pass.run_next_pass(bbox, satellite)
    dest = args.output_dir / output_dir.name

    # Move or copy whole directory
    output_dir.rename(dest)
    print(f"[INFO] Moved next_pass output directory to {dest}")

    # Read the metadata CSV file
    df_opera = read_opera_metadata_csv(dest)

    short_name = args.short_name
    layer_name = args.layer_name
    date = args.date 

    df_opera['Start Time'] = pd.to_datetime(df_opera['Start Time'], format='mixed')
    df_opera['Start Date'] = df_opera['Start Time'].dt.date.astype(str)

    layers = f"Download URL {layer_name}"

    matching = df_opera[df_opera['Start Date'] == date]

    layer_links = matching[layers].dropna().tolist()

    if not layer_links:
        print(f"[WARNING] No matching links found for date {date} and layer {layer_name}.")
        return
    
    else:
        DS = compile_and_load_data(layer_links)
        print(f"[INFO] Loaded {len(DS)} datasets from matching links.")
        # Perform merge using OPERA product pixel priority rules
        mosaic, colormap, nodata = leafmap.common.mosaic_opera(DS, product=short_name, merge_args={})
        image = leafmap.common.array_to_image(mosaic, colormap=colormap, nodata=nodata)
        print("Pre-event granule mosaic generated successfully...")

        # Create filename and full paths
        mosaic_name = f"{short_name}_{layer_name}_{date}_mosaic.tif"
        mosaic_path = args.output_dir / mosaic_name
        tmp_path = args.output_dir / f"tmp_{mosaic_name}"

        # Save the mosaic
        rasterio.shutil.copy(image, mosaic_path, driver='GTiff')

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


if __name__ == "__main__":
    main()