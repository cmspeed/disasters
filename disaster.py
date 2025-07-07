import os
import argparse
from pathlib import Path
import pandas as pd
from osgeo import gdal
import next_pass
import rasterio
import rioxarray

def parse_arguments():
    """
    Parse command line arguments for the disaster analysis workflow.
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
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
    """
    Authenticate with Earthdata and ASF for data access.
    Returns:
        tuple: (username, password) for Earthdata and ASF access.
    """
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
    """
    Create the output directory if it does not exist.
    Args:
        output_dir (Path): Path to the output directory.
    Raises:
        Exception: If the directory cannot be created.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Created or reused output directory: {output_dir}")
    except Exception as e:
        print(f"[ERROR] Could not create output directory: {e}")
        raise
    return

def read_opera_metadata_csv(output_dir):
    """
    Read the OPERA products metadata CSV file.
    Args:
        output_dir (Path): Path to the directory containing the CSV file.
    Returns:
        pd.DataFrame: DataFrame containing the metadata from the CSV file.
    Raises:
        FileNotFoundError: If the CSV file does not exist in the specified directory.
    """
    csv_path = output_dir / "opera_products_metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} rows from {csv_path}")
    return df

def compile_and_load_data(layer_links):
    """
    Compile and load data from the provided layer links.
    Args:
        layer_links (list): List of URLs corresponding to the OPERA data layers.
    Returns:
        list: List of rioxarray datasets loaded from the provided links.
    Raises:
        Exception: If there is an error loading any of the datasets.
    """
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
    """
    Generate products based on the provided DataFrame and mode.
    Args:
        df_opera (pd.DataFrame): DataFrame containing OPERA products metadata.
        mode (str): Mode of operation, e.g., "flood", "fire", "earthquake".
        mode_dir (Path): Path to the directory where products will be saved.
    Raises:
        Exception: If the mode is not recognized or if there are issues with data processing.
    """
    import opera_mosaic
    from rasterio.shutil import copy

    # Create data directory
    data_dir = mode_dir / "data"
    make_output_dir(data_dir)

    # Create maps directory
    maps_dir = mode_dir / "maps"
    make_output_dir(maps_dir)

    # Create layouts directory
    layouts_dir = mode_dir / "layouts"
    make_output_dir(layouts_dir)
    
    if mode == "flood":
        short_names = ["OPERA_L3_DSWX-HLS_V1", "OPERA_L3_DSWX-S1_V1"]
        layer_names = ["WTR", "BWTR"]
    elif mode == "fire":
        short_names = ["OPERA_L3_DIST-ALERT-HLS_V1", "OPERA_L3_DIST-ALERT-S1_V1"]
        layer_names = ["VEG-ANOM-MAX", "VEG-DIST-STATUS"]
    elif mode == "earthquake":
        print("Earthquake mode coming soon. Exiting...")
        return

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
                mosaic_path = data_dir / mosaic_name
                tmp_path = data_dir / f"tmp_{mosaic_name}"

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

                # Make a map with PyGMT
                map_name = make_map(maps_dir, mosaic_path, short_name, layer, date)

                # Make a layout with matplotlib
                make_layout(layouts_dir, map_name, short_name, layer, date)

    return

def expand_region(region, width_deg=15, height_deg=10):
    """
    Return a new region [xmin, xmax, ymin, ymax] of fixed size,
    centered on the centroid of the input region, with coordinates rounded
    to 0 decimal places.
    Args:
        region (list): Input region in the form [xmin, xmax, ymin, ymax].
        width_deg (float): Desired width in degrees.
        height_deg (float): Desired height in degrees.
    Returns:
        list: New region with fixed size, centered on the input region.
    Raises:
        ValueError: If the input region is not in the correct format.
    """
    xmin, xmax, ymin, ymax = region
    center_lon = (xmin + xmax) / 2
    center_lat = (ymin + ymax) / 2

    half_width = width_deg / 2
    half_height = height_deg / 2

    expanded_region = [
        round(center_lon - half_width),
        round(center_lon + half_width),
        round(center_lat - half_height),
        round(center_lat + half_height),
    ]
    return expanded_region

def expand_region_to_aspect(region, target_aspect):
    """
    Expand the input region to match a target aspect ratio.
    Args:
        region (list): Input region in the form [xmin, xmax, ymin, ymax].
        target_aspect (float): Desired aspect ratio (width / height).
    Returns:
        list: New region with adjusted aspect ratio.
    Raises:
        ValueError: If the input region is not in the correct format.
    """
    xmin, xmax, ymin, ymax = map(float, region)
    width = xmax - xmin
    height = ymax - ymin
    current_aspect = width / height

    if current_aspect > target_aspect:
        # Too wide → expand height
        new_height = width / target_aspect
        pad = (new_height - height) / 2
        ymin -= pad
        ymax += pad
    else:
        # Too tall → expand width
        new_width = height * target_aspect
        pad = (new_width - width) / 2
        xmin -= pad
        xmax += pad

    return [xmin, xmax, ymin, ymax]

def make_map(maps_dir, mosaic_path, short_name, layer, date):
    """
    Create a map using PyGMT from the provided mosaic path.
    Args:
        maps_dir (Path): Directory where the map will be saved.
        mosaic_path (Path): Path to the mosaic file.
        short_name (str): Short name of the product.
        layer (str): Layer name to be used in the map.
        date (str): Date string in the format YYYY-MM-DD.
    Returns:
        map_name (Path): Path to the saved map image.
    Raises:
        ImportError: If required libraries are not installed.
    """
    import pygmt
    import rioxarray
    from pyproj import Geod
    import math

    mosaic_wgs84 = Path(str(mosaic_path).replace(".tif", "_WGS84.tif"))

    gdal.Warp(
        mosaic_wgs84,
        mosaic_path,
        dstSRS='EPSG:4326',
        resampleAlg='near',
        creationOptions=['COMPRESS=DEFLATE']
    )

    # === Load Mosaic ===
    grd = rioxarray.open_rasterio(mosaic_wgs84).squeeze()
    grd = grd.where(grd != 255)  # Remove nodata values

    # === Region ===
    bounds = grd.rio.bounds()  # xmin, ymin, xmax, ymax
    region = [bounds[0], bounds[2], bounds[1], bounds[3]]  # [xmin, xmax, ymin, ymax]

    # === Define Target Aspect Ratio ===
    # To match matplotlib layout: extent=[0, 60, 0, 100] → aspect ratio = width / height
    target_aspect = 60 / 100

    # Pad region to match target aspect ratio
    region_padded = expand_region_to_aspect(region, target_aspect)

    # === Define Projection ===
    center_lon = (region_padded[0] + region_padded[1]) / 2
    projection_width_cm = 15  # can be any fixed value; output height adjusts
    projection = f"M{center_lon}/{projection_width_cm}c"

    # === Create PyGMT Figure ===
    fig = pygmt.Figure()

    # Base coast layer (optional)
    fig.coast(
        region=region_padded,
        projection=projection,
        borders="2/thin",
        shorelines="thin",
        land="grey",
        water="lightblue",
        frame="a",
    )

    # === Add Grid Image ===
    if layer == 'WTR':
        color_palette = 'palettes/WTR.cpt'
        fig.grdimage(
            grid=grd,
            region=region_padded,
            projection=projection,
            cmap=color_palette,
            frame=["WSne", 'xaf', 'yaf'],
            nan_transparent=True
        )
        fig.colorbar(
            cmap=color_palette,
            equalsize=1.5,
        )

    elif layer == 'BWTR':
        color_palette = 'palettes/BWTR.cpt'
        fig.grdimage(
            grid=grd,
            region=region_padded,
            projection=projection,
            cmap=color_palette,
            frame=["WSne", 'xaf', 'yaf'],
            nan_transparent=True
        )
        fig.colorbar(
            cmap=color_palette,
            equalsize=1.5,
        )

    elif layer == "VEG-ANOM-MAX":
        color_palette = 'palettes/VEG-ANOM-MAX.cpt'
        fig.grdimage(
            grid=grd,
            region=region_padded,
            projection=projection,
            cmap=color_palette,
            frame=["WSne", 'xaf', 'yaf'],
            nan_transparent=True
        )
        fig.colorbar(
            cmap=color_palette,
            frame="xaf+lVEG-ANOM-MAX(%)",
        )

    elif layer == "VEG-DIST-STATUS":
        color_palette = 'palettes/VEG-DIST-STATUS.cpt'
        fig.grdimage(
            grid=grd,
            region=region_padded,
            projection=projection,
            cmap=color_palette,
            frame=["WSne", 'xaf', 'yaf'],
            nan_transparent=True
        )
        fig.colorbar(
            cmap=color_palette,
            equalsize=1.5,
        )

    # === Add Scale Bar and Compass ===
    xmin, xmax, ymin, ymax = region_padded
    center_lat = (ymin + ymax) / 2
    geod = Geod(ellps="WGS84")
    _, _, distance_m = geod.inv(xmin, center_lat, xmax, center_lat)

    # Set scalebar to ~25% of region width
    raw_length_km = distance_m * 0.25 / 1000  # 25% of map width in km
    exponent = math.floor(math.log10(raw_length_km))
    base = 10 ** exponent

    for factor in [1, 2, 5, 10]:
        scalebar_length_km = base * factor
        if scalebar_length_km >= raw_length_km:
            break
        
    fig.basemap(map_scale=f"jBR+w{scalebar_length_km:.0f}k+o0.5c/0.5c")
    fig.basemap(rose="jTR+o0.5c/0.5c+w1.5c")

    bounds = grd.rio.bounds()
    region = [bounds[0], bounds[2], bounds[1], bounds[3]]
    region_expanded = expand_region(region, width_deg=25, height_deg=15)

    # === Add Inset Map ===
    with fig.inset(
        position="jBL+o0.2c/0.2c",
        box="+pblack",
        region=region_expanded,
        projection="M5c",
    ):
        # Use a plotting method to create a figure inside the inset.
        fig.coast(
            land="gray",
            borders=[1, 2],
            shorelines="1/thin",
            water="white",
        )
        # Coordinates for rectangular outline of the main region
        xmin, xmax, ymin, ymax = region_padded
        rectangle = [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
            [xmin, ymin],  # Close the loop
        ]

        # Plot the rectangle on the inset
        fig.plot(
            x=[pt[0] for pt in rectangle],
            y=[pt[1] for pt in rectangle],
            pen="2p,red"
        )

    # # === Export Map with Consistent Aspect Ratio ===
    map_name = maps_dir / f"{short_name}_{layer}_{date}_map.png"
    fig.savefig(map_name, dpi=900)

    return map_name

def make_layout(layout_dir, map_name, short_name, layer, date):
    """
    Create a layout using matplotlib for the provided map.
    Args:
        layout_dir (Path): Directory where the layout will be saved.
        map_name (Path): Path to the map image.
        short_name (str): Short name of the product.
        layer (str): Layer name to be used in the layout.
        date (str): Date string in the format YYYY-MM-DD.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.image as mpimg
    import textwrap
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from matplotlib.image import imread


    # Create blank figure
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_axis_off()

    # Set background
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # === Add Main Map ===
    map_img = mpimg.imread(map_name)
    ax.imshow(map_img, extent=[0, 60, 0, 100])  # Main map on left 60% of layout

    # === Add Logos ===
    logo_opera = imread("logos/OPERA_logo.png")
    logo_new = imread("logos/ARIA_logo.png") 

    # Create a new axes for logos in the bottom-right corner
    logo_ax = fig.add_axes([0.82, 0.02, 0.06, 0.08], anchor='SE', zorder=10)
    logo_ax.imshow(logo_opera)
    logo_ax.axis('off')

    logo_ax2 = fig.add_axes([0.89, 0.02, 0.06, 0.08], anchor='SE', zorder=10)
    logo_ax2.imshow(logo_new)
    logo_ax2.axis('off')

    # Layout control
    x_pos = 0.65  # Just right of the map
    line_spacing = 0.04  # vertical spacing between blocks

    # Define wrap width (in characters)
    wrap_width = 50

    # Map text elements
    title = "Placeholder Title"

    if short_name == "OPERA_L3_DSWX-S1_V1":
        subtitle = "OPERA Dynamic Surface Water eXtent from Sentinel-1 (DSWx-S1)"
        map_information = (
            f"The ARIA/OPERA water extent map is derived from an OPERA DSWx-S1 mosaicked "
            f"product from Copernicus Sentinel-1 data."
            f"This map depicts regions of full surface water and inundated surface water. "
        )
        data_source = "Copernicus Sentinel-1"
        
    elif short_name == "OPERA_L3_DSWX-HLS_V1":
        subtitle = "OPERA Dynamic Surface Water eXtent from HLS (DSWx-HLS)"
        map_information = (
            f"The ARIA/OPERA water extent map is derived from an OPERA DSWx-HLS mosaicked " 
            f"product from Harmonized Landsat and Sentinel-2 data."
            f"This map depicts regions of vegetation disturbance."
        )
        data_source = "Copernicus Harmonized Landsat and Sentinel-2"
        
    elif short_name == "OPERA_L3_DIST-ALERT-S1_V1":
        subtitle = "OPERA Surface Disturbance Alert from Sentinel-1 (DIST-ALERT-S1)"
        map_information = (
            f"The ARIA/OPERA surface disturbance alert map is derived from an OPERA DIST-ALERT-S1 mosaicked "
            f"product from Copernicus Sentinel-1 data."
            f"This map depicts regions of surface disturbance."
        )
        data_source = "Copernicus Sentinel-1"

    elif short_name == "OPERA_L3_DIST-ALERT-HLS_V1":
            subtitle = "OPERA Surface Disturbance Alert from Harmonized Landsat and Sentinel-2 (DIST-ALERT-HLS)"
            map_information = (
            f"The ARIA/OPERA surface disturbance alert map is derived from an OPERA DIST-ALERT-HLS mosaicked "
            f"product from Harmonized Landsat and Sentinel-2 data."
            f"This map depicts regions of vegetation disturbance."
            )
            data_source = "Copernicus Harmonized Landsat and Sentinel-2"

    acquisitions = f"{date}"

    data_sources = textwrap.dedent(f"""\
        Product: {short_name}

        Layer: {layer}

        Data Source: {data_source}

        Resolution: 30 meters
    """)

    data_availability = textwrap.dedent(f"""\
        This product is available at: https://aria-share.jpl.nasa.gov/

        Visit the OPERA website: https://www.jpl.nasa.gov/go/opera/
    """)


    disclaimer = ("The results posted here are preliminary and unvalidated, "
                "intended to aid field response and provide a first look at the disaster-affected region.")


    # Wrapping text
    title_wrp = textwrap.fill(title, width=40)
    subtitle_wrp = textwrap.fill(subtitle, width=wrap_width)
    acquisitions_wrp = textwrap.fill(acquisitions, width=wrap_width)
    map_information_wrp = textwrap.fill(map_information, width=wrap_width)
    data_sources_wrp = textwrap.fill(data_sources, width=wrap_width)
    data_availability_wrp = textwrap.fill(data_availability, width=wrap_width)
    disclaimer_wrp = textwrap.fill(disclaimer, width=wrap_width)

    # Starting y-position (top of the figure)
    y_start = 0.98

    # === Title ===
    ax.text(x_pos, y_start, title_wrp,
            fontsize=14, weight='bold',
            ha='left', va='top', transform=ax.transAxes)

    # === Subtitle ===
    ax.text(x_pos, y_start - line_spacing * 1, subtitle_wrp,
            fontsize=8, fontweight='bold', ha='left', va='top', transform=ax.transAxes)

    # === Acquisition Heading ===
    ax.text(x_pos, y_start - line_spacing * 3.5, "Data Acquisitions:",
            fontsize=8, fontweight='bold', ha='left', va='top', transform=ax.transAxes)

    # === Acquisition Dates ===
    ax.text(x_pos, y_start - line_spacing * 4, acquisitions,
            fontsize=8, ha='left', va='top', transform=ax.transAxes)

    # === Map Information Heading ===
    ax.text(x_pos, y_start - line_spacing * 6, "Map Information:",
            fontsize=8, fontweight='bold', ha='left', va='top', transform=ax.transAxes)

    # === Map Information Text ===
    ax.text(x_pos, y_start - line_spacing * 6.5, map_information_wrp,
            fontsize=8, ha='left', va='top', transform=ax.transAxes)

    # === Data Sources Heading ===
    ax.text(x_pos, y_start - line_spacing * 10, "Data Sources:",
            fontsize=8, fontweight='bold', ha='left', va='top', transform=ax.transAxes)

    # === Data Sources Text ===
    ax.text(x_pos, y_start - line_spacing * 10.5, data_sources,
            fontsize=8, ha='left', va='top', transform=ax.transAxes,linespacing=1, wrap=True
    )
    # === Data Availability Heading ===
    ax.text(x_pos, y_start - line_spacing * 15, "Product Availability:",
            fontsize=8, fontweight='bold', ha='left', va='top', transform=ax.transAxes)

    # === Data Availability Text ===
    ax.text(x_pos, y_start - line_spacing * 15.5, data_availability,
            fontsize=8, ha='left', va='top', linespacing=1, transform=ax.transAxes, wrap=True)

    # === Disclaimer Heading ===
    ax.text(x_pos, y_start - line_spacing * 18, "Disclaimer:",
            fontsize=8, fontweight='bold', ha='left', va='top', transform=ax.transAxes)

    # === Disclaimer (bottom-left) ===
    ax.text(x_pos, y_start - line_spacing * 18.5, disclaimer_wrp,
            fontsize=8, ha='left', va='top', transform=ax.transAxes)

    plt.tight_layout()

    layout_name = layout_dir / f"{short_name}_{layer}_{date}_layout.pdf"
    plt.savefig(layout_name, format="pdf", bbox_inches="tight", dpi=400)
    return 

def main():
    """
    Main entry point for the disaster analysis workflow.
    This function parses command line arguments, sets up the output directory,
    authenticates with Earthdata and ASF, and runs the next_pass module to generate
    disaster products based on the specified mode (flood, fire, earthquake).
    Raises:
        Exception: If there are issues with directory creation, CSV reading, or product generation.
    """
    args = parse_arguments()
    
    # Terminate if user selects 'earthquake' mode, for now
    if args.mode == "earthquake":
        print("Earthquake mode coming soon. Exiting...")
        return

    output_dir = next_pass.run_next_pass(
        bbox=args.bbox,
        number_of_dates=args.number_of_dates,
        date=args.date,
    )

    make_output_dir(args.output_dir)
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
    
    return

if __name__ == "__main__":
    main()