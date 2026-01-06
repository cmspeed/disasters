![Python](https://img.shields.io/badge/python-3.9%2B-blue)

# disasters

An automated workflow for generating disaster response maps and layouts over a user-defined AOI using NASA-JPL OPERA products.

## Overview

This tool streamlines the generation of data to support disaster response efforts using NASA-JPL OPERA products. It automates the discovery, download, mosaicking, differencing (if applicable), and visualization of products related to:

- Flooding (e.g., DSWx-HLS, DSWx-S1)
- Wildfires (e.g., DIST-ALERT-HLS)
- Landslides (e.g., DIST-ALERT-HLS, RTC)
- Earthquakes (e.g., CSLC, DISP) *(coming soon)*

The output includes ready-to-share maps and analysis-ready GeoTIFFs for any user-defined region and event type. Currently `flood`, `wildfire`, and `landslide` are supported (*earthquakes coming soon*).

## Development setup

### Requirements

- Python 3.9+
- [Mamba](https://mamba.readthedocs.io/en/latest/installation.html) or [Conda](https://docs.conda.io/en/latest/miniconda.html)
- Git

We recommend using **Mamba** for faster environment setup.

### Prerequisite installs
1. Download source code:
```bash
git clone https://github.com/cmspeed/disasters
```

2. Navigate to and install dependencies, using the included `environment.yml` file:

```bash
cd disasters
mamba env create --file environment.yml
mamba activate disasters
```
The provided `environment.yml` file sets up a fully functional environment with all required packages.

### Usage

#### Example: Generate flood maps over Lake Mead, Nevada using the two most recent OPERA products
```bash
python disaster.py -b 35 37 -115 -113 -o LakeMead -m flood -n 2 -lt "Lake Mead Floods"
```
#### Example: Generate fire impact maps over New Mexico using the five (default) most recent OPERA products
```bash
python disaster.py -b 32 34 -106.5 -104 -o NM_Fires -m fire -lt "New Mexico Fires, June 2025"
```

#### Example: Generate fire impact maps over a wildfire in Quebec using the most recent 30 OPERA products (prior to 07-31-2023), filtered to remove disturbance prior to 05-15-2023
```bash
python disaster.py -b 48 49.5 -77.5 -74.4 -o QuebecFires -m fire -d 2023-07-31 -n 30 -lt "Quebec Wildfire, Summer 2023" -fd 2023-05-15
```

#### Example: Generate landslide impact maps over a landslide in Brazil in February 2023, retaining only pixels with slopes greater than 15 degrees (slope filtering is optional)
```bash
python disaster.py -b -24 -23.5 -45.75 -45.5 -o brazil_landslides -m landslide -lt "Brazil Landslides, Feb. 2023" -fd 2023-02-01 -d 2023-03-01 -zb -23.783 -23.733 -45.733 -45.683 -st 15

```

#### Example: Generate flood maps over Jamaica for Hurricane Melissa in October 2025
In this example, misclassified snow/ice classified pixels (likely sediment-rich water) are reclassified to open water using the DIST-HLS Confidence layer.
Note: All snow/ice/sediment-rich water pixels are not reclassified using this approach.
```bash
python disaster.py -b 17.3 18.8 -78.6 -75.6 -m flood -o hurricane_melissa_Oct2025 -rc -lt "Hurricane Melissa, Oct. 2025"
```

#### Example: Generate disturbance maps over Jamaica for Hurricane Melissa in October 2025
In this example, a filter data (`-fd`) of October 28, 2025 (coinciding with hurricane landfall) is applied. All disturbance prior to this date is filtered out of the final mosaics.
```bash
python disaster.py -b 17.3 18.8 -78.6 -75.6 -m fire -o hurricane_melissa_Oct2025 -lt "Hurricane Melissa, Oct. 2025" -fd 2025-10-28
```

### Running with Local Data

You can process pre-downloaded OPERA GeoTIFFs stored on your local machine by using the `-ld` / `--local_dir` argument. When this flag is set, the tool skips the cloud search/download step and processes all valid files found in the specified directory.

#### Basic Local Usage
```bash
python disaster.py -b 35 37 -115 -113 -ld /path/to/my/data -o LocalOutput -m flood -lt "Local Test"
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

### Command-line Arguments

| Argument             | Required | Description                                                                                   |
|----------------------|----------|-----------------------------------------------------------------------------------------------|
| `-b`, `--bbox`        | Yes      | Bounding box: `South North West East` (space-separated floats) |
| `-o`, `--output_dir`  | Yes      | Output directory or prefix for storing results |
| `-m`, `--mode`        | Yes      | Disaster mode: `flood`, `fire`, `landslide`, or `earthquake`|
| `-ld`, `--local_dir`  | No       | Path to a local directory containing pre-downloaded OPERA GeoTIFFs. If provided, cloud search is skipped. |
| `-d`, `--date`        | No       | Specifies the end date (YYYY-MM-DD) for the OPERA product search. The script will find the 'N' most recent products available on or before this date. Defaults to 'today'. (Ignored if `-ld` is used) |
| `-n`, `--number_of_dates` | No   | Number of most recent dates to process (default: `5`). (Ignored if `-ld` is used) |
| `-lt`, `--layout_title` | Yes     | Title of PDF layout generated for each product |
| `-fd`, `--filter_date` | No     | Date to use as filter in `fire` mode to remove all disturbance preceding `filter_date` |
| `-rc`, `--reclassify_snow_ice` | No     | Flag to reclassify false snow/ice positives as water in DSWx-HLS products ONLY. (Default: False)|
| `-st`, `--slope_threshold` | No     | Specifies a slope threshold in degrees (0-100) to be used only in `landslide` mode. When set, the tool calculates a slope map using the DEM (Band 10) from available DSWx-HLS products. Pixels in the final mosaics (DIST and RTC) corresponding to slope values less than this threshold will be filtered out (set to nodata). This is useful for pinpointing potential landslide candidates. |
| `-zb`, `--zoom_bbox`  | No     | Zoom bounding box: `South North West East` (space-separated floats) |

### Disaster Modes

The `-m / --mode` argument determines which NASA OPERA products and data layers are used.

| Mode         | OPERA Products                     | Layer(s)                              | Description                                                                 |
|--------------|------------------------------------|----------------------------------------|-----------------------------------------------------------------------------|
| `flood`      | `DSWx-HLS`, `DSWx-S1` | `WTR`, `BWTR` | Detects surface water using optical (HLS) and SAR (S1) observations         |
| `fire`       | `OPERA_L3_DIST-ALERT-HLS_V1`, `DIST-ALERT-S1` *(coming soon)* | `VEG-ANOM-MAX`, `VEG-DIST-STATUS` | Identifies vegetation disturbance and anomalies from wildfire events        |
| `landslide`       | `OPERA_L3_DIST-ALERT-HLS_V1`, `OPERA_L2_RTC-S1_V1` | `VEG-ANOM-MAX`, `VEG-DIST-STATUS`, `RTC-VV`, `RTC-VH` |Identifies vegetation disturbance and anomalies from landslides events        |
| `earthquake` | `CSLC`, `DISP`, `RTC-S1` *(coming soon)* | *(coming soon)* | Maps surface displacement and SAR backscatter changes related to seismic activity | 

### Output
For each valid product and date:
- Mosaicked GeoTIFF file
- Reprojected WGS84 version of the mosaic
- Difference maps for water products for all available date pairs (Water gain for DSWx, Log-ratio for RTC)
- Quicklook PNG map with legend and colorbar
- Layout in PDF format including PNG map and explanation

Products are organized in a timestamped subdirectory under your specified `--output_dir` (or directly in the output directory if running locally).

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.