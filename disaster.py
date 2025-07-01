import argparse
from pathlib import Path
import pandas as pd

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

def main():
    import next_pass

    args = parse_arguments()
    
    make_output_dir(args.output_dir)

    bbox = args.bbox
    satellite = args.satellite

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
    print(layer_links)

if __name__ == "__main__":
    main()