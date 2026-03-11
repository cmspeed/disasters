from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import click

from .pipeline import PipelineConfig, run_pipeline

logger = logging.getLogger(__name__)

# Keep the same valid values as in the original disaster.py parser
VALID_SHORT_NAMES = [
    "OPERA_L3_DSWX-HLS_V1",
    "OPERA_L3_DSWX-S1_V1",
    "OPERA_L3_DIST-ALERT-HLS_V1",
    "OPERA_L3_DIST-ANN-HLS_V1",
    "OPERA_L2_RTC-S1_V1",
    "OPERA_L2_CSLC-S1_V1",
    "OPERA_L3_DISP-S1_V1",
]

VALID_LAYER_NAMES = ["WTR", "BWTR", "VEG-ANOM-MAX", "VEG-DIST-STATUS"]
VALID_MODES = ["flood", "fire", "landslide", "earthquake", "rtc-rgb"]
VALID_FUNCTIONS = ["opera_search", "both"]


@click.group()
def cli() -> None:
    """Disaster products pipeline CLI."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@cli.command(name="run")
@click.option(
    "-b",
    "--bbox",
    type=str,
    required=True,
    help=(
        "Bounding box or area of interest. MUST be enclosed in double quotes if it contains spaces. "
        "Accepted formats: "
        "1) 4 floats: \"S N W E\" | "
        "2) WKT string: \"POLYGON((...))\" | "
        "3) Local path: \"/path/to/file.kml\" | "
        "4) Web URL: \"https://example.com/AOI.geojson\""
    ),
)
@click.option(
    "-zb",
    "--zoom-bbox",
    type=str,
    default=None,
    help="Optional bounding box for the zoom-in inset map. MUST be 4 floats enclosed in double quotes (e.g., \"S N W E\").",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    required=True,
    help="Directory where results and metadata will be saved.",
)
@click.option(
    "-ld",
    "--local-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    required=False,
    default=None,
    help="Path to a local directory containing pre-downloaded OPERA geotiffs. If provided, cloud search is skipped.",
)
@click.option(
    "-sn",
    "--short-name",
    type=click.Choice(VALID_SHORT_NAMES),
    required=False,
    help=(
        "Short name to filter the DataFrame (must be one of the known OPERA "
        "products). Currently not used by the pipeline logic but kept for "
        "CLI compatibility."
    ),
)
@click.option(
    "-l",
    "--layer-name",
    type=click.Choice(VALID_LAYER_NAMES),
    required=False,
    help=(
        "Layer name to extract from metadata (e.g., 'WTR', 'BWTR', "
        "'VEG-ANOM-MAX'). Currently not used by the pipeline logic but kept "
        "for CLI compatibility."
    ),
)
@click.option(
    "-d",
    "--date",
    type=str,
    required=False,
    help=(
        "Date string. Can be a single end date (YYYY-MM-DD) to find the N most recent products, "
        "OR a date range (YYYY-MM-DD/YYYY-MM-DD). If a range is provided, the script calculates "
        "the required number of passes automatically."
    ),
)
@click.option(
    "-n",
    "--number-of-dates",
    type=int,
    default=5,
    show_default=True,
    help="Number of most recent dates to consider for OPERA products. (Overridden if a date range is provided in -d).",
)
@click.option(
    "-m",
    "--mode",
    type=click.Choice(VALID_MODES),
    default="flood",
    show_default=True,
    help="Mode of operation: flood, fire, landslide, earthquake, rtc-rgb.",
)
@click.option(
    "-f",
    "--functionality",
    type=click.Choice(VALID_FUNCTIONS),
    default="opera_search",
    show_default=True,
    help="Functionality to run: 'opera_search' or 'both'.",
)
@click.option(
    "-lt",
    "--layout-title",
    type=str,
    required=True,
    help="Title for the PDF layout(s). Enclose in quotes if it has spaces.",
)
@click.option(
    "-fd",
    "--filter-date",
    type=str,
    required=False,
    default=None,
    help=(
        "Date string (YYYY-MM-DD) to filter by date in the date filtering step "
        "in 'fire' and 'landslide' mode."
    ),
)
@click.option(
    "-rc",
    "--reclassify-snow-ice",
    is_flag=True,
    default=False,
    help=("Reclassify false snow/ice positives as water in DSWx-HLS products ONLY."),
)
@click.option(
    "-st", 
    "--slope-threshold", 
    type=int, 
    metavar="DEG", 
    default=None, 
    required=False,
    help="Slope threshold in degrees (0-100). Pixels with slope < threshold will be masked in Landslide mode.",
)
@click.option(
    "--benchmark", 
    is_flag=True, 
    default=False,
    help="If set, runs data loading in both sequential and concurrent modes to compare performance.",
)
@click.option(
    "--no-mask",
    is_flag=True,
    default=False,
    help="If set, skips the coastal masking step.",
)
@click.option(
    "-c",
    "--compute_cloudiness",
    is_flag=True,
    default=False,
    help="Enable HLS cloud cover calculation. This may significantly increase runtime, especially for large AOIs or wide date ranges."
)

def run(
    bbox: str,
    zoom_bbox: Optional[str],
    output_dir: Path,
    local_dir: Optional[Path],
    short_name: Optional[str],
    layer_name: Optional[str],
    date: Optional[str],
    number_of_dates: int,
    mode: str,
    functionality: str,
    layout_title: str,
    filter_date: Optional[str],
    reclassify_snow_ice: bool,
    slope_threshold: Optional[int],
    benchmark: bool,
    no_mask: bool,
    compute_cloudiness: bool
) -> None:
    """Run the disaster pipeline (end-to-end)."""
    # Ensure slope values are between 0 and 100 degrees, if provided
    if slope_threshold is not None and not (0 <= slope_threshold <= 100):
        raise click.BadParameter("Slope threshold must be between 0 and 100.", param_hint="--slope-threshold")

    # Process bbox tokens into a list of floats OR a single WKT/path string
    bbox_parts = bbox.replace(",", " ").split()
    
    if len(bbox_parts) == 4:
        try:
            bbox_arg = [float(x) for x in bbox_parts]
        except ValueError:
            bbox_arg = bbox
    else:
        # Keep as WKT string or file path
        bbox_arg = bbox

    # Process zoom_bbox if provided
    zoom_bbox_arg = None
    if zoom_bbox is not None:
        zoom_parts = zoom_bbox.replace(",", " ").split()
        if len(zoom_parts) == 4:
            try:
                zoom_bbox_arg = [float(x) for x in zoom_parts]
            except ValueError:
                raise click.BadParameter("Zoom bounding box must contain exactly 4 valid numbers.", param_hint="--zoom-bbox")
        else:
            raise click.BadParameter("Zoom bounding box must contain exactly 4 valid numbers.", param_hint="--zoom-bbox")

    # Build the PipelineConfig object
    cfg = PipelineConfig(
        bbox=bbox_arg,
        zoom_bbox=zoom_bbox_arg,
        output_dir=output_dir,
        local_dir=local_dir,
        short_name=short_name,
        layer_name=layer_name,
        date=date,
        number_of_dates=number_of_dates,
        mode=mode,
        functionality=functionality,
        layout_title=layout_title,
        filter_date=filter_date,
        reclassify_snow_ice=reclassify_snow_ice,
        slope_threshold=slope_threshold,
        benchmark=benchmark,
        no_mask=no_mask,
        compute_cloudiness=compute_cloudiness
    )

    mode_dir = run_pipeline(cfg)
    if mode_dir is not None:
        logger.info(f"Pipeline completed. Mode outputs in: {mode_dir}")
    else:
        logger.info("Pipeline exited without running (e.g., earthquake mode).")


@cli.command(name="download")
@click.option(
    "-b",
    "--bbox",
    type=str,
    required=True,
    help=(
        "Bounding box or area of interest. MUST be enclosed in double quotes if it contains spaces. "
        "Accepted formats: \"S N W E\" | \"POLYGON((...))\" | \"/path/to/file.kml\""
    ),
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    required=True,
    help="Directory where the 'data' folder and metadata will be saved.",
)
@click.option(
    "-d",
    "--date",
    type=str,
    required=False,
    help="Date string (YYYY-MM-DD) OR a date range (YYYY-MM-DD/YYYY-MM-DD).",
)
@click.option(
    "-n",
    "--number-of-dates",
    type=int,
    default=5,
    show_default=True,
    help="Number of most recent dates to consider for OPERA products.",
)
@click.option(
    "-m",
    "--mode",
    type=click.Choice(VALID_MODES),
    default=None,
    required=False,
    help="Optional: Filter downloads to only include products and layers relevant to a specific mode.",
)
@click.option(
    "-c",
    "--compute_cloudiness",
    is_flag=True,
    default=False,
    help="Enable HLS cloud cover calculation. This may significantly increase runtime, especially for large AOIs or wide date ranges."
)

def download(
    bbox: str,
    output_dir: Path,
    date: Optional[str],
    number_of_dates: int,
    mode: Optional[str],
    compute_cloudiness: bool
) -> None:
    """Download OPERA granules over an AOI/time window for local use."""
    
    # Process bbox tokens
    bbox_parts = bbox.replace(",", " ").split()
    if len(bbox_parts) == 4:
        try:
            bbox_arg = [float(x) for x in bbox_parts]
        except ValueError:
            bbox_arg = bbox
    else:
        bbox_arg = bbox

    from .pipeline import run_download_only
    
    logger.info("Starting standalone download pipeline...")
    out_dir = run_download_only(
        bbox=bbox_arg,
        output_dir=output_dir,
        date=date,
        number_of_dates=number_of_dates,
        mode=mode,
        compute_cloudiness=compute_cloudiness
    )
    
    if out_dir:
        logger.info(f"Download complete. Files saved to: {out_dir}")
    else:
        logger.warning("Download pipeline exited without producing outputs.")


@cli.command(name="mosaic")
@click.option(
    "-i",
    "--input-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True, exists=True),
    required=True,
    help="Path to a local directory containing pre-downloaded OPERA geotiffs.",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    required=True,
    help="Directory where the stitched GeoTIFF mosaics will be saved.",
)
@click.option(
    "-b",
    "--bbox",
    type=str,
    required=False,
    default=None,
    help=(
        "Optional bounding box to crop the output. If omitted, the pipeline computes the geographic union of all inputs. "
        "MUST be enclosed in double quotes if it contains spaces. "
        "Accepted formats: \"S N W E\" | \"POLYGON((...))\" | \"/path/to/file.kml\""
    ),
)
@click.option(
    "--benchmark", 
    is_flag=True, 
    default=False,
    help="If set, tracks performance metrics during the mosaicking process.",
)

def mosaic(
    input_dir: Path,
    output_dir: Path,
    bbox: Optional[str],
    benchmark: bool
) -> None:
    """Stitch local OPERA granules into analysis-ready mosaics (No analysis/layouts)."""
    
    # Process optional bbox using the same parsing as the run command
    bbox_arg = None
    if bbox is not None:
        bbox_parts = bbox.replace(",", " ").split()
        if len(bbox_parts) == 4:
            try:
                coords = [float(x) for x in bbox_parts]
                # Auto-swap S/N if flipped
                if coords[0] > coords[1]: coords[0], coords[1] = coords[1], coords[0]
                # Auto-swap W/E if flipped
                if coords[2] > coords[3]: coords[2], coords[3] = coords[3], coords[2]
                bbox_arg = coords
            except ValueError:
                bbox_arg = bbox
        else:
            bbox_arg = bbox

    # Import the dedicated mosaic pipeline (we will build this next)
    from .pipeline import run_mosaic_only
    
    logger.info("Starting mosaic pipeline...")
    output_path = run_mosaic_only(
        input_dir=input_dir,
        output_dir=output_dir,
        bbox=bbox_arg,
        benchmark=benchmark
    )
    
    if output_path:
        logger.info(f"Mosaicking complete. Outputs saved to: {output_path}")
    else:
        logger.warning("Mosaic pipeline exited without producing outputs.")

if __name__ == "__main__":
    cli()