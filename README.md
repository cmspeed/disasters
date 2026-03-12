![Python](https://img.shields.io/badge/python-3.12-blue)

# disasters

An automated workflow for generating disaster response maps and layouts over a user-defined AOI using NASA-JPL OPERA products.

## Overview

This tool streamlines the generation of data to support disaster response efforts using NASA-JPL OPERA products. It automates the discovery, download, mosaicking, differencing (if applicable), and visualization of products related to:

- Flooding (e.g., DSWx-HLS, DSWx-S1)
- Wildfires (e.g., DIST-ALERT-HLS)
- Landslides (e.g., DIST-ALERT-HLS, RTC)
- SAR Backscatter Visualizations (e.g., RTC RGB Composites)
- Earthquakes (e.g., CSLC, DISP) *(coming soon)*

The output includes ready-to-share maps and analysis-ready GeoTIFFs for any user-defined region and event type. Currently `flood`, `fire`, `landslide`, and `rtc-rgb` are supported (`earthquake` is exposed in the CLI but exits early because implementation is not complete yet).

## Development setup

### Requirements

- Python 3.12
- [Mamba](https://mamba.readthedocs.io/en/latest/installation.html) or [Conda](https://docs.conda.io/en/latest/miniconda.html)
- Git

We recommend using **Mamba** for faster environment setup.

### Prerequisite installs
1. Download source code:
```bash
git clone https://github.com/OPERA-Cal-Val/disasters.git
```

2. Navigate to the repository, install dependencies using the included environment.yml file, and install the package:

```bash
cd disasters
mamba env create --file environment.yml
mamba activate disasters
pip install -e .
```
The provided `environment.yml` file sets up a fully functional environment, and the pip install command makes the opera-disaster CLI tool available system-wide.

`environment.yml` is the source of truth for the working environment (including conda-native/system packages like GDAL and notebook/tooling dependencies). `pyproject.toml` pulls package runtime dependencies dynamically from `requirements.txt` for packaging/install metadata.

### Commands

The CLI exposes three commands:

- `opera-disaster run`: end-to-end workflow for search/download, mosaicking, filtering, differencing, maps, and layouts.
- `opera-disaster download`: download raw OPERA GeoTIFFs for later local processing.
- `opera-disaster mosaic`: build stitched mosaics from a local directory without generating maps or layouts.

### Usage

#### `run`

You can define your temporal search window in two ways: by providing a single target date (`-d YYYY-MM-DD`) to retrieve the `N` most recent satellite passes defined by the `-n` argument, or by providing a strict date range (`-d YYYY-MM-DD/YYYY-MM-DD`) to automatically query all products within that exact window (which ignores the `-n` argument).

For `run`, the `-b/--bbox` argument accepts:

- `"S N W E"` bounding coordinates
- WKT geometry such as `"POLYGON((...))"`
- A local geometry file path such as `/path/to/aoi.kml`
- A web URL pointing to an AOI file such as `https://example.com/aoi.geojson`

When `-ld/--local_dir` is used, the pipeline skips cloud search and scans local `.tif` files recursively. In local mode, the implementation currently supports `-b` as either `S N W E` coordinates or WKT geometry.

#### Example: Generate flood maps over Lake Mead, Nevada using the two most recent OPERA products
```bash
opera-disaster run -b "35 37 -115 -113" -m flood -n 2 -o LakeMead -lt "Lake Mead Floods"
```

#### Example: Query a strict date range (Fire Mode)
Generate products within a specific multi-week window over Los Angeles corresponding to wildfires. Disturbance prior to the fires is removed with `-fd`.
```bash
opera-disaster run -b "33.5 35 -119.5 -117" -m fire -d 2025-01-01/2025-01-15 -fd 2025-01-01 -o LA_fires -lt "Los Angeles Fires, January 2025"
```

#### Example: Generate fire impact maps over New Mexico using the five (default) most recent OPERA products
```bash
opera-disaster run -b "32 34 -106.5 -104" -m fire -o NM_Fires -lt "New Mexico Fires, June 2025"
```

#### Example: Generate fire impact maps over a wildfire in Quebec using the most recent 30 OPERA products (prior to 07-31-2023), filtered to remove disturbance prior to 05-15-2023
```bash
opera-disaster run -b "48 49.5 -77.5 -74.4" -m fire -d 2023-07-31 -n 30 -fd 2023-05-15 -o QuebecFires -lt "Quebec Wildfire, Summer 2023"
```

#### Example: Generate landslide impact maps over a landslide in Brazil in February 2023
This command uses the `-st` argument to retain only pixels with slopes greater than 15 degrees (slope filtering is optional)
```bash
opera-disaster run -b "-24 -23.5 -45.75 -45.5" -m landslide -d 2023-02-01/2023-03-01 -fd 2023-02-01 -zb "-23.783 -23.733 -45.733 -45.683" -st 15 -o brazil_landslides -lt "Brazil Landslides, Feb. 2023"
```

#### Example: Generate flood maps over Rio Grande do Sol, Brazil in April-May 2024
Sediment-rich water pixels are often flagged as snow/ice. This command reclassifies snow/ice pixels to water and computes cloudiness from the HLS granules.
```bash
opera-disaster run -b "-30.5 -29.5 -52 -51" -m flood -d 2024-04-20/2024-05-08 -o RioGrandeDoSulFloods2024 -lt "Rio Grande Do Sul Floods, Brazil, 2024" -rc -c
```

#### Example: Generate flood maps over Jamaica for Hurricane Melissa in October 2025
In this example, misclassified snow/ice classified pixels (likely sediment-rich water) are reclassified to open water using the DIST-HLS Confidence layer.
Note: All snow/ice/sediment-rich water pixels are not reclassified using this approach.
```bash
opera-disaster run -b "17.3 18.8 -78.6 -75.6" -m flood -d 2025-10-17/2025-11-04 -o hurricane_melissa_Oct2025 -rc -lt "Hurricane Melissa, Oct. 2025"
```

#### Example: Generate disturbance maps over Jamaica for Hurricane Melissa in October 2025
In this example, a filter data (`-fd`) of October 28, 2025 (coinciding with hurricane landfall) is applied. All disturbance prior to this date is filtered out of the final mosaics.
```bash
opera-disaster run -b "17.3 18.8 -78.6 -75.6" -m fire -d 2025-10-17/2025-11-04 -fd 2025-10-28 -o hurricane_melissa_Oct2025 -lt "Hurricane Melissa, Oct. 2025"
```

#### Example: Generate an RTC RGB composite visualization

Create an 8-bit RGB composite from Sentinel-1 RTC backscatter data to visualize surface features using the 5 most recent passes.
```bash
opera-disaster run -b "35.5 36.5 -107 -106" -m rtc-rgb -d 2026-02-01/2026-02-15 -o VallesCaldera_RTC -lt "Valles Caldera SAR Backscatter"
```

#### Example: Add a zoomed inset and disable coastal masking
```bash
opera-disaster run -b "34.8 35.4 -120.0 -119.2" -zb "34.95 35.15 -119.75 -119.45" -m flood -o SantaBarbara -lt "Santa Barbara Flooding" --no-mask
```

#### Example: Benchmark the pipeline
```bash
opera-disaster run -b "35 37 -115 -113" -m flood -o LakeMeadBench -lt "Lake Mead Floods" --benchmark
```

### Running with Local Data

You can process pre-downloaded OPERA GeoTIFFs stored on your local machine by using the `-ld` / `--local_dir` argument. When this flag is set, the tool skips the cloud search/download step and processes all valid files found in the specified directory.

#### Basic Local Usage
```bash
opera-disaster run -b "35 37 -115 -113" -m flood -ld /path/to/my/data -o LocalOutput -lt "Local Test"
```

**Note:** The bounding box (`-b`) is still required to define the map extent and master grid alignment.

#### File Organization
The tool scans recursively, so file organization does not matter. You can:
* Dump all files into one folder.
* Organize them by date (e.g., `data/20231005/`).
* Organize them by product type.

**Important:** The script relies on standard OPERA naming conventions (e.g., `OPERA_L3_DSWx-HLS_...`) to identify products, dates, and tile IDs. Renaming files arbitrarily may cause them to be skipped.

#### Supporting Advanced Features Locally
To use features like **Snow/Ice Reclassification** or **Temporal Filtering** locally, you must ensure the required auxiliary files are present in the directory alongside the main data files. The script automatically pairs them based on the Tile ID and Date.

* **Snow/Ice Reclassification (`-rc`):** Requires the `_CONF` layer file (e.g., `..._B03_CONF.tif`) for DSWx-HLS products.
* **Fire/Disturbance Filtering (`-fd`):** Requires the `VEG-DIST-DATE` (e.g., `..._VEG-DIST-DATE.tif`) and `VEG-DIST-CONF` (e.g., `..._VEG-DIST-CONF.tif`) files.

If auxiliary files are missing, the script will log a warning and proceed with standard processing (skipping the advanced step).

### Standalone Download Workflow

Use `download` when you want to fetch OPERA files without running the full analysis pipeline.

```bash
opera-disaster download -b "35 37 -115 -113" -o LakeMead_downloads -m flood -d 2026-02-01/2026-02-15
```

Notes:

- Downloads are written under `<output_dir>/data`.
- If `-m/--mode` is omitted, the command downloads all available OPERA products and layers returned by the catalog.
- Mode-filtered downloads include required auxiliary layers such as `CONF`, `VEG-DIST-DATE`, and `VEG-DIST-CONF` when needed.

### Standalone Mosaic Workflow

Use `mosaic` when you already have local GeoTIFFs and only want stitched outputs.

```bash
opera-disaster mosaic -i /path/to/local/data -o mosaics_out
```

You can optionally crop the mosaics with `-b`:

```bash
opera-disaster mosaic -i /path/to/local/data -o mosaics_out -b "35 37 -115 -113"
```

If `-b` is omitted, the command computes the geographic union of the local inputs automatically.

### Command-line Arguments

| Argument             | Required | Description                                                                                   |
|----------------------|----------|-----------------------------------------------------------------------------------------------|
| `-b`, `--bbox`        | Yes      | AOI for `run`; accepts `South North West East`, WKT, a local geometry path, or a remote geometry URL. Must be enclosed in quotes. |
| `-o`, `--output_dir`  | Yes      | Output directory or prefix for storing results |
| `-m`, `--mode`        | No       | Mode: `flood`, `fire`, `landslide`, `rtc-rgb`, or `earthquake`; defaults to `flood` for `run` |
| `-ld`, `--local_dir`  | No       | Path to a local directory containing pre-downloaded OPERA GeoTIFFs. If provided, cloud search is skipped. |
| `-d`, `--date`        | No       | Date string. Can be a single end date (`YYYY-MM-DD`) OR a strict date range (`YYYY-MM-DD/YYYY-MM-DD`). If a range is provided, the script automatically queries the exact window. Defaults to `today`. (Ignored if `-ld` is used) |
| `-n`, `--number_of_dates` | No   | Number of most recent dates to process (default: `5`). (Ignored if `-ld` is used) |
| `-f`, `--functionality` | No    | Search mode passed to `next_pass`; defaults to `opera_search` |
| `-sn`, `--short_name` | No      | Compatibility flag retained in the CLI; currently not used by the main pipeline logic |
| `-l`, `--layer_name` | No       | Compatibility flag retained in the CLI; currently not used by the main pipeline logic |
| `-lt`, `--layout_title` | Yes     | Title of PDF layout generated for each product |
| `-fd`, `--filter_date` | No     | Date to use as a disturbance filter in `fire` and DIST-based `landslide` processing |
| `-rc`, `--reclassify_snow_ice` | No     | Flag to reclassify false snow/ice positives as water in DSWx-HLS products ONLY. (Default: False)|
| `-st`, `--slope_threshold` | No     | Slope threshold in degrees (0-100); applied to `landslide` and `rtc-rgb` outputs when DEM-derived slope data can be built |
| `--benchmark` | No | Reports timing comparisons for loading, plotting, and differencing |
| `--no-mask` | No | Skips the global coastal masking step |
| `-c`, `--compute_cloudiness` | No | Flag to enable HLS cloud cover calculation during the search step. |
| `-zb`, `--zoom_bbox`  | No       | Optional inset extent for `run`, as `South North West East`. Must be enclosed in quotes.|

### Disaster Modes

The `-m / --mode` argument determines which NASA OPERA products and data layers are used.

| Mode         | OPERA Products                     | Layer(s)                              | Description                                                                 |
|--------------|------------------------------------|----------------------------------------|-----------------------------------------------------------------------------|
| `flood`      | `DSWx-HLS`, `DSWx-S1` | `WTR`, `BWTR` | Detects surface water using optical (HLS) and SAR (S1) observations         |
| `fire`       | `OPERA_L3_DIST-ALERT-HLS_V1`, `OPERA_L3_DIST-ALERT-S1_V1` | `VEG-ANOM-MAX`, `VEG-DIST-STATUS` | Identifies vegetation disturbance and anomalies from wildfire events        |
| `landslide`       | `OPERA_L3_DIST-ALERT-HLS_V1`, `OPERA_L2_RTC-S1_V1` | `VEG-ANOM-MAX`, `VEG-DIST-STATUS`, `RTC-VV`, `RTC-VH` | Identifies vegetation disturbance and anomalies from landslides events        |
| `rtc-rgb`       | `OPERA_L2_RTC-S1_V1` | `RTC-VV`, `RTC-VH` | Generates 8-bit RGB composite visualizations from Sentinel-1 RTC backscatter data
| `earthquake` | *(coming soon)* | *(coming soon)* | Placeholder CLI mode; the current implementation logs a message and exits | 

### Output
For the `run` command, outputs are written under `<output_dir>/<mode>/` in subdirectories such as `data/`, `maps/`, and `layouts/`.

For each valid product and pass, the pipeline can generate:
- Mosaicked GeoTIFF file
- Spatially synchronized auxiliary products where needed (for example `CONF` mosaics in flood mode)
- Difference maps for supported pairings:
  - Flood: water gain products across pass pairs
  - Landslide: RTC log-difference products across pass pairs
- Quicklook PNG map with legend and colorbar
- Layout in PDF format including PNG map and explanation

Special cases:

- `rtc-rgb` creates RGB GeoTIFF outputs but does not generate the standard map/layout products.
- `mosaic` writes stitched GeoTIFFs only.
- `download` writes raw source files and copied metadata only.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.
