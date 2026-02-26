import netrc
from pathlib import Path

import boto3
import earthaccess
import rasterio
from rasterio.session import AWSSession


def authenticate() -> tuple:
    """
    Authenticate with Earthdata and ASF for data access.

    Returns:
        tuple: (username, password) for Earthdata and ASF access.
    """
    temp_creds_req = earthaccess.get_s3_credentials(daac="PODAAC")
    session = boto3.Session(
        aws_access_key_id=temp_creds_req["accessKeyId"],
        aws_secret_access_key=temp_creds_req["secretAccessKey"],
        aws_session_token=temp_creds_req["sessionToken"],
        region_name="us-west-2",
    )
    
    cookie_path = str(Path.home() / "cookies.txt")
    rio_env = rasterio.Env(
        AWSSession(session),
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS="TIF, TIFF",
        GDAL_HTTP_COOKIEFILE=cookie_path,
        GDAL_HTTP_COOKIEJAR=cookie_path,
    )
    rio_env.__enter__()

    # Parse credentials from the netrc file for ASF access
    netrc_file = Path.home() / ".netrc"
    auths = netrc.netrc(netrc_file)
    username, _, password = auths.authenticators("urs.earthdata.nasa.gov")
    
    return username, password