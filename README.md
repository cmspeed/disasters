# disasters

An automated workflow for generating disaster response maps using NASA OPERA products.  
Supports event types such as floods, fires, and earthquakes, and produces mosaicked, georeferenced outputs based on user-defined areas of interest.

## Development setup

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

### Usage

```bash
python disaster.py -b 35 37 -115 -113 -o LakeMead -m flood -n 2
```

### Arguments
`-b / --bbox` Required. The bounding box for the area of interest, specified as four space-separated float values: South North West East
Example:
`-b 35 37 -115 -113`
This selects a region from 35°N to 37°N latitude and 115°W to 113°W longitude.

`-o / --output_dir` Required. Name or path of the directory where output data and maps will be saved.
Example:
`-o LakeMead`

`-m / --mode` Required. Type of disaster event to process. Determines which OPERA product(s) and layers are downloaded and mosaicked.
Options:
`flood`: Water detection products (e.g., DSWx-HLS, DSWx-S1)
`fire` : Vegetation anomaly and disturbance layers (e.g., DIST-ALERT-HLS)
`earthquake` : Displacement products (e.g., CSLC, DISP, RTC-S1)
Default: `flood`

`-n / --number_of_dates` Optional. Number of most recent dates to search for and generate OPERA products. 
Default: `5`

### Output
For each valid product and date:
- Mosaicked GeoTIFF file
- Reprojected WGS84 version of the mosaic
- Quicklook PNG map with legend and colorbar
- Layout including PNG map and explanation

Products are organized in a timestamped subdirectory under your specified `--output_dir`.