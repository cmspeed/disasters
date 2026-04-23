import atexit
import netrc
from pathlib import Path

import boto3
import earthaccess
import rasterio
from osgeo import gdal
from rasterio.session import AWSSession

_RIO_ENV = None
_RIO_ENV_SIGNATURE = None


def _close_rasterio_env() -> None:
    """Close the cached Rasterio environment when it is replaced or on exit."""
    global _RIO_ENV, _RIO_ENV_SIGNATURE

    if _RIO_ENV is None:
        return

    try:
        _RIO_ENV.__exit__(None, None, None)
    except Exception:
        pass
    finally:
        _RIO_ENV = None
        _RIO_ENV_SIGNATURE = None


def authenticate() -> tuple[str, str]:
    """
    Authenticate with Earthdata (EDL) and set up Rasterio/GDAL environment.
    Returns (username, password) from ~/.netrc for endpoints that need it.
    """

    # Ensure Earthdata session (creates/updates ~/.netrc)
    try:
        earthaccess.login(strategy="netrc")
    except Exception as e:
        print(f"[WARNING] earthaccess.login() failed; continuing: {e}")

    # Use a hidden cookie file in the home directory to avoid local permission issues
    cookie_path = str(Path.home() / ".earthdata_cookies.txt")

    # Try to obtain S3 temporary creds for ASF (OPERA RTC Host)
    temp_creds_req = None
    try:
        temp_creds_req = earthaccess.get_s3_credentials(daac="ASF")
    except Exception as e:
        print(f"[WARNING] Could not get S3 credentials; continuing without AWS session: {e}")

    # Setup GLOBAL GDAL configuration (For raw osgeo.gdal tools like Warp)
    gdal.SetConfigOption("GDAL_HTTP_COOKIEFILE", cookie_path)
    gdal.SetConfigOption("GDAL_HTTP_COOKIEJAR", cookie_path)
    gdal.SetConfigOption("GDAL_HTTP_NETRC", "YES")
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", "tif,tiff,xml,json")

    # Build Rasterio environment (For xarray/rioxarray tools)
    env_kwargs = dict(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS="tif,tiff,xml,json",
        GDAL_HTTP_COOKIEFILE=cookie_path,
        GDAL_HTTP_COOKIEJAR=cookie_path,
    )

    env_signature = (
        cookie_path,
        temp_creds_req["accessKeyId"],
        temp_creds_req["secretAccessKey"],
        temp_creds_req["sessionToken"],
    ) if isinstance(temp_creds_req, dict) and all(
        k in temp_creds_req for k in ("accessKeyId", "secretAccessKey", "sessionToken")
    ) else (
        cookie_path,
        None,
        None,
        None,
    )

    if isinstance(temp_creds_req, dict) and all(k in temp_creds_req for k in ("accessKeyId", "secretAccessKey", "sessionToken")):
        # Apply AWS Creds to Rasterio
        session = boto3.Session(
            aws_access_key_id=temp_creds_req["accessKeyId"],
            aws_secret_access_key=temp_creds_req["secretAccessKey"],
            aws_session_token=temp_creds_req["sessionToken"],
            region_name="us-west-2",
        )
        rio_env = rasterio.Env(AWSSession(session), **env_kwargs)
        
        # Apply AWS Creds to raw GDAL
        gdal.SetConfigOption("AWS_ACCESS_KEY_ID", temp_creds_req["accessKeyId"])
        gdal.SetConfigOption("AWS_SECRET_ACCESS_KEY", temp_creds_req["secretAccessKey"])
        gdal.SetConfigOption("AWS_SESSION_TOKEN", temp_creds_req["sessionToken"])
        gdal.SetConfigOption("AWS_REGION", "us-west-2")
    else:
        if temp_creds_req is not None:
            print("[WARNING] S3 credentials missing expected keys; continuing without AWS session.")
        rio_env = rasterio.Env(**env_kwargs)

    # Keep one Rasterio env open per process and refresh it only when credentials change.
    global _RIO_ENV, _RIO_ENV_SIGNATURE
    if _RIO_ENV_SIGNATURE != env_signature:
        _close_rasterio_env()
        rio_env.__enter__()
        _RIO_ENV = rio_env
        _RIO_ENV_SIGNATURE = env_signature

    # Parse credentials from ~/.netrc (used for xarray endpoints)
    netrc_file = Path.home() / ".netrc"
    try:
        auths = netrc.netrc(netrc_file)
        username, _, password = auths.authenticators("urs.earthdata.nasa.gov")
    except Exception:
        print("[WARNING] Could not parse ~/.netrc file.")
        username, password = None, None

    return username, password


atexit.register(_close_rasterio_env)
