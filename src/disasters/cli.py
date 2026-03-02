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
    type=float,
    nargs=4,
    required=True,
    metavar="S N W E",
    help="Bounding box in the form: South North West East.",
)
@click.option(
    "-zb",
    "--zoom-bbox",
    type=float,
    nargs=4,
    metavar="S N W E",
    default=None,
    help="Optional bounding box for the zoom-in inset map.",
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
def run(
    bbox: Sequence[float],
    zoom_bbox: Optional[Sequence[float]],
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
) -> None:
    """Run the disaster pipeline (end-to-end)."""
    # Ensure slope values are between 0 and 100 degrees, if provided
    if slope_threshold is not None and not (0 <= slope_threshold <= 100):
        raise click.BadParameter("Slope threshold must be between 0 and 100.", param_hint="--slope-threshold")

    # Build the PipelineConfig object
    cfg = PipelineConfig(
        bbox=list(bbox),
        zoom_bbox=list(zoom_bbox) if zoom_bbox is not None else None,
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
        benchmark=benchmark
    )

    mode_dir = run_pipeline(cfg)
    if mode_dir is not None:
        logger.info(f"Pipeline completed. Mode outputs in: {mode_dir}")
    else:
        logger.info("Pipeline exited without running (e.g., earthquake mode).")


if __name__ == "__main__":
    cli()