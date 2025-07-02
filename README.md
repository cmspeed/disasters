![Python](https://img.shields.io/badge/python-3.9%2B-blue)

# disasters

An automated workflow for generating disaster response maps and layouts using NASA-JPL OPERA products.

## Overview

This tool streamlines the generation of data to support disaster response efforts using NASA-JPL OPERA products. It automates the discovery, download, mosaicking, and visualization of products related to:

- Flooding (e.g., DSWx-HLS)
- Wildfires (e.g., DIST-ALERT-HLS)
- Earthquakes (e.g., CSLC, DISP) *(coming soon)*

The output includes ready-to-share maps and analysis-ready GeoTIFFs for any user-defined region and event type. Currently flood and wildfires (*earthquakes coming soon*) are supported.

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

#### Example: Generate flood maps over Lake Mead, Nevada using the two most recent dates of available OPERA data
```bash
python disaster.py -b 35 37 -115 -113 -o LakeMead -m flood -n 2
```
#### Example: Generate fire impact maps over New Mexico using the fire most recent dates of available OPERA data (default)
```bash
python disaster.py -b 32 34 -106.5 -104 -o NM_Fires -m fire
```
### Command-line Arguments

| Argument             | Required | Description                                                                                   |
|----------------------|----------|-----------------------------------------------------------------------------------------------|
| `-b`, `--bbox`        | Yes      | Bounding box: `South North West East` (space-separated floats)                              |
| `-o`, `--output_dir`  | Yes      | Output directory or prefix for storing results                                               |
| `-m`, `--mode`        | Yes      | Disaster mode: `flood`, `fire`, or `earthquake`                                              |
| `-n`, `--number_of_dates` | No  | Number of most recent dates to process (default: `5`)                                        |

### Disaster Modes

The `-m / --mode` argument determines which disaster type to process and which NASA OPERA products and data layers are used.

| Mode         | OPERA Products                     | Layer(s)                              | Description                                                                 |
|--------------|------------------------------------|----------------------------------------|-----------------------------------------------------------------------------|
| `flood`      | `DSWx-HLS`, `DSWx-S1` | `WTR`, `BWTR` | Detects surface water using optical (HLS) and SAR (S1) observations         |
| `fire`       | `DIST-ALERT-HLS`, `DIST-ALERT-S1` *(coming soon)* | `VEG-ANOM-MAX`, `VEG-DIST-STATUS` | Identifies vegetation disturbance and anomalies from wildfire events        |
| `earthquake` | `CSLC`, `DISP`, `RTC-S1` *(coming soon)* | *(coming soon)* | Maps surface displacement and SAR backscatter changes related to seismic activity |

### Output
For each valid product and date:
- Mosaicked GeoTIFF file
- Reprojected WGS84 version of the mosaic
- Quicklook PNG map with legend and colorbar
- Layout in PDF format including PNG map and explanation

Products are organized in a timestamped subdirectory under your specified `--output_dir`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.