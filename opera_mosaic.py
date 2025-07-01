import rasterio
import rioxarray
from rioxarray.merge import merge_arrays
import xarray as xr

def mosaic_opera(DS, product="OPERA_L3_DSWX-S1_V1", merge_args={}):
    """Mosaics a list of OPERA product granules into a single image (in memory).
    
    Args:
        DS (list): A list of OPERA product granules opened as xarray.DataArray objects.
        product (str): OPERA product short name. Used to define pixel prioritization scheme in regions of OPERA granule overlap.
            Options include: "OPERA_L3_DSWX-HLS_V1","OPERA_L3_DSWX-S1_V1", "OPERA_L3_DIST-ALERT-HLS_V1", "OPERA_L3_DIST-ANN-HLS_V1", "OPERA_L2_RTC-S1_V1" 
            Default: "OPERA_L3_DSWX-S1_V1"
        merge_args (dict, optional): A dictionary of arguments to pass to the rioxarray.merge_arrays function. Defaults to {}.
    
    Returns:
        da_mosaic: An xarray.DataArray containing the mosaic of the individual OPERA product granule DataArrays.
        colormap: A colormap for the mosaic, if in the original OPERA metadata, otherwise None.
        nodata: The nodata value for the mosaic corresponding to the original OPERA product granule metadata.
    """
    from rioxarray.merge import merge_arrays
    import numpy as np
    
    DA = []
    for ds in DS:
        nodata = ds.rio.nodata
        da = ds.fillna(nodata)
        DA.append(da)

    # Define 'valid' values for each product type
    if product.startswith("OPERA_L3_DSWX"):
        priority = {1: 100, 2:95, 3: 90, 0: 50, 250: 20, 251: 15, 252: 10, 253:5, 254:1, 255: 0}
    elif product.startswith("OPERA_L3_DIST"):
        priority = {1: 100, 2:100, 3: 100, 4:100, 5:100, 6:100, 7:100, 8:100, 9:100, 10:100, 0:10, 255: 0}
    # elif product.startswith("OPERA_L2_RTC"):
    #     priority = {}
    else:
        priority = {}

    valid_values = set(priority.keys())

    # Check if any DataArray contains non-valid values, if so fall back to defaul rasterio.merge method
    if contains_unexpected_values(DA, valid_values):
        method = "first"
    elif product.startswith("OPERA_L3_DIST") or product.startswith("OPERA_L2_RTC"):
        method = "first"
    else:
        method = opera_rules(product=product, nodata=nodata)

    merged_arr = merge_arrays(DA, method=method)

    try:
        colormap = get_image_colormap(DS[0])
    except Exception as e:
        colormap = None
    return merged_arr, colormap, nodata

def opera_rules(product="OPERA_L3_DSWX-S1_V1", nodata=255):
    """Returns a custom callabale rasterio.merge method for OPERA products using pixel priority rules.
    Args:
        product (str): OPERA product short name, used to determine pixel prioritization in regions of OPERA granule overlap.
            Options include: "OPERA_L3_DSWX-HLS_V1","OPERA_L3_DSWX-S1_V1", "OPERA_L3_DIST-ALERT-HLS_V1", "OPERA_L3_DIST-ANN-HLS_V1", "OPERA_L2_RTC-S1_V1" 
            Default: "OPERA_L3_DSWX-S1_V1"
        nodata (int): The nodata value for the OPERA product. Default is 255.
    Returns:
        method (function): A function that implements the custom merge method for the specified OPERA product.
    """

    if product in ("OPERA_L3_DSWX-HLS_V1", "OPERA_L3_DSWX-S1_V1"):
        priority = {
            1: 100,   # Open water (DSWx-HLS, DSWx-S1)
            2: 95,    # Partial surface water (DSWx-HLS)
            3: 95,    # Inundated vegetation (DSWx-S1)
            0: 50,    # Not water (DSWx-HLS, DSWx-S1)
            250: 20,  # Height Above Nearest Drainage (HAND) masked (DSWx-S1)
            251: 15,  # Layover/shadow masked (DSWx-S1)
            252: 10,  # Snow/Ice (DSWx-HLS)
            253: 5,   # Cloud/Cloud shadow (DSWx-HLS)
            254: 1,   # Ocean masked (DSWx-HLS)
            255: 0    # Fill value (no data) (DSWx-HLS, DSWx-S1)
        }
    elif product in ("OPERA_L3_DIST-ALERT-HLS_V1", "OPERA_L3_DIST-ANN-HLS_V1"):
        priority = {
            1:100, # first <50% 
            2:100, # provisional <50% 
            3:100, # confirmed <50% 
            4:100, # first ≥50% 
            5:100, # provisional ≥50% 
            6:100, # confirmed ≥50% 
            7:100, # confirmed <50%, finished 
            8:100, # confirmed ≥50%, finished 
            9:100, # confirmed previous year <50% 
            10:100, # confirmed previous year ≥50%
            0:10, # No disturbance 
            255:0 # No data
        }
    elif product == 'OPERA_L2_RTC-S1_V1':
        priority = {
            1: 100,   # Open water (DSWx-HLS, DSWx-S1)
            2: 95,    # Partial surface water (DSWx-HLS)
            3: 90,    # Inundated vegetation (DSWx-S1)
            0: 50,    # Not water (DSWx-HLS, DSWx-S1)
            250: 20,  # Height Above Nearest Drainage (HAND) masked (DSWx-S1)
            251: 15,  # Layover/shadow masked (DSWx-S1)
            252: 10,  # Snow/Ice (DSWx-HLS)
            253: 5,   # Cloud/Cloud shadow (DSWx-HLS)
            254: 1,   # Ocean masked (DSWx-HLS)
            255: 0    # Fill value (no data) (DSWx-HLS, DSWx-S1)
        }

    else:
        raise ValueError(f"Unknown product type: {product}. Supported products are DSWx, DIST, RTC.")

    def method(old_data, new_data, old_nodata=None, new_nodata=None, index=None, roff=None, coff=None):
        """
        Custom merge method for OPERA products using pixel priority rules.

        Args:
            old_data (numpy.ndarray): The existing data array.
            new_data (numpy.ndarray): The new data array to merge.
            old_nodata (int, optional): The nodata value for the existing data. Defaults to None. Required by rasterio.merge.
            new_nodata (int, optional): The nodata value for the new data. Defaults to None. Required by rasterio.merge.
            index (tuple, optional): The index of the pixel being merged. Defaults to None. Required by rasterio.merge.
            roff (int, optional): Row offset. Defaults to None. Required by rasterio.merge.
            coff (int, optional): Column offset. Defaults to None. Required by rasterio.merge.
        
        Returns:
            numpy.ndarray: The merged data array.
        """
        import numpy as np

        max_val = max(priority.keys()) + 1
        priority_array = np.full(max_val, -1, dtype=np.int16)
        for val, pri in priority.items():
            priority_array[val] = pri

        valid_mask = new_data != nodata

        for i in range(old_data.shape[0]):
            new_vals = new_data[i]
            old_vals = old_data[i]

            new_priorities = priority_array[new_vals]
            old_priorities = priority_array[old_vals]

            update_mask = (valid_mask[i]) & (new_priorities > old_priorities)

            # Apply the update
            old_vals[update_mask] = new_vals[update_mask]

        return old_data
    return method

def contains_unexpected_values(DA, valid_values):
    import numpy as np
    for da in DA:
        unique_vals = np.unique(da.values)
        if not set(unique_vals).issubset(valid_values):
            return True
    return False

def get_image_colormap(image, index=1):
    """
    Retrieve the colormap from an image.

    Args:
        image (str, rasterio.io.DatasetReader, rioxarray.DataArray):
            The input image. It can be:
            - A file path to a raster image (string).
            - A rasterio dataset.
            - A rioxarray DataArray.
        index (int): The band index to retrieve the colormap from (default is 1).

    Returns:
        dict: A dictionary representing the colormap (value: (R, G, B, A)), or None if no colormap is found.

    Raises:
        ValueError: If the input image type is unsupported.
    """
    import rasterio
    import rioxarray
    import xarray as xr

    dataset = None

    if isinstance(image, str):  # File path
        with rasterio.open(image) as ds:
            return ds.colormap(index) if ds.count > 0 else None
    elif isinstance(image, rasterio.io.DatasetReader):  # rasterio dataset
        dataset = image
    elif isinstance(image, xr.DataArray) or isinstance(image, xr.Dataset):
        source = image.encoding.get("source")
        if source:
            with rasterio.open(source) as ds:
                return ds.colormap(index) if ds.count > 0 else None
        else:
            raise ValueError(
                "Cannot extract colormap: DataArray does not have a source."
            )
    else:
        raise ValueError(
            "Unsupported input type. Provide a file path, rasterio dataset, or rioxarray DataArray."
        )

    if dataset:
        return dataset.colormap(index) if dataset.count > 0 else None

def array_to_memory_file(
    array,
    source: str = None,
    dtype: str = None,
    compress: str = "deflate",
    transpose: bool = True,
    cellsize: float = None,
    crs: str = None,
    transform: tuple = None,
    driver="COG",
    colormap: dict = None,
    **kwargs,
):
    """Convert a NumPy array to a memory file.

    Args:
        array (numpy.ndarray): The input NumPy array.
        source (str, optional): Path to the source file to extract metadata from. Defaults to None.
        dtype (str, optional): The desired data type of the array. Defaults to None.
        compress (str, optional): The compression method for the output file. Defaults to "deflate".
        transpose (bool, optional): Whether to transpose the array from (bands, rows, columns) to (rows, columns, bands). Defaults to True.
        cellsize (float, optional): The cell size of the array if source is not provided. Defaults to None.
        crs (str, optional): The coordinate reference system of the array if source is not provided. Defaults to None.
        transform (tuple, optional): The affine transformation matrix if source is not provided.
            Can be rio.transform() or a tuple like (0.5, 0.0, -180.25, 0.0, -0.5, 83.780361). Defaults to None.
        driver (str, optional): The driver to use for creating the output file, such as 'GTiff'. Defaults to "COG".
        colormap (dict, optional): A dictionary defining the colormap (value: (R, G, B, A)).
        **kwargs: Additional keyword arguments to be passed to the rasterio.open() function.

    Returns:
        rasterio.DatasetReader: The rasterio dataset reader object for the converted array.
    """
    import rasterio
    import numpy as np
    import xarray as xr
    from rasterio.transform import Affine

    if isinstance(array, xr.DataArray):
        coords = [coord for coord in array.coords]
        if coords[0] == "time":
            x_dim = coords[1]
            y_dim = coords[2]
            array = (
                array.isel(time=0).rename({y_dim: "y", x_dim: "x"}).transpose("y", "x")
            )
        if hasattr(array, "rio"):
            if hasattr(array.rio, "crs"):
                if array.rio.crs is not None:
                    crs = array.rio.crs
            if transform is None and hasattr(array.rio, "transform"):
                transform = array.rio.transform()
        elif source is None:
            if hasattr(array, "encoding"):
                if "source" in array.encoding:
                    source = array.encoding["source"]
        array = array.values

    if array.ndim == 3 and transpose:
        array = np.transpose(array, (1, 2, 0))
    if source is not None:
        with rasterio.open(source) as src:
            crs = src.crs
            transform = src.transform
            if compress is None:
                compress = src.compression
    else:
        if crs is None:
            raise ValueError(
                "crs must be provided if source is not provided, such as EPSG:3857"
            )

        if transform is None:
            if cellsize is None:
                raise ValueError("cellsize must be provided if source is not provided")
            # Define the geotransformation parameters
            xmin, ymin, xmax, ymax = (
                0,
                0,
                cellsize * array.shape[1],
                cellsize * array.shape[0],
            )
            # (west, south, east, north, width, height)
            transform = rasterio.transform.from_bounds(
                xmin, ymin, xmax, ymax, array.shape[1], array.shape[0]
            )
        elif isinstance(transform, Affine):
            pass
        elif isinstance(transform, (tuple, list)):
            transform = Affine(*transform)

        kwargs["transform"] = transform

    if dtype is None:
        # Determine the minimum and maximum values in the array
        min_value = np.min(array)
        max_value = np.max(array)
        # Determine the best dtype for the array
        if min_value >= 0 and max_value <= 1:
            dtype = np.float32
        elif min_value >= 0 and max_value <= 255:
            dtype = np.uint8
        elif min_value >= -128 and max_value <= 127:
            dtype = np.int8
        elif min_value >= 0 and max_value <= 65535:
            dtype = np.uint16
        elif min_value >= -32768 and max_value <= 32767:
            dtype = np.int16
        else:
            dtype = np.float64

    # Convert the array to the best dtype
    array = array.astype(dtype)
    # Define the GeoTIFF metadata
    metadata = {
        "driver": driver,
        "height": array.shape[0],
        "width": array.shape[1],
        "dtype": array.dtype,
        "crs": crs,
        "transform": transform,
    }

    if array.ndim == 2:
        metadata["count"] = 1
    elif array.ndim == 3:
        metadata["count"] = array.shape[2]
    if compress is not None:
        metadata["compress"] = compress

    metadata.update(**kwargs)

    # Create a new memory file and write the array to it
    memory_file = rasterio.MemoryFile()
    dst = memory_file.open(**metadata)

    if array.ndim == 2:
        dst.write(array, 1)
        if colormap:
            dst.write_colormap(1, colormap)
    elif array.ndim == 3:
        for i in range(array.shape[2]):
            dst.write(array[:, :, i], i + 1)
            if colormap:
                dst.write_colormap(i + 1, colormap)

    dst.close()
    # Read the dataset from memory
    dataset_reader = rasterio.open(dst.name, mode="r")

    return dataset_reader

def array_to_image(
    array,
    output: str = None,
    source: str = None,
    dtype: str = None,
    compress: str = "deflate",
    transpose: bool = True,
    cellsize: float = None,
    crs: str = None,
    transform: tuple = None,
    driver: str = "COG",
    colormap: dict = None,
    **kwargs,
) -> str:
    """Save a NumPy array as a GeoTIFF using the projection information from an existing GeoTIFF file.

    Args:
        array (np.ndarray): The NumPy array to be saved as a GeoTIFF.
        output (str): The path to the output image. If None, a temporary file will be created. Defaults to None.
        source (str, optional): The path to an existing GeoTIFF file with map projection information. Defaults to None.
        dtype (np.dtype, optional): The data type of the output array. Defaults to None.
        compress (str, optional): The compression method. Can be one of the following: "deflate", "lzw", "packbits", "jpeg". Defaults to "deflate".
        transpose (bool, optional): Whether to transpose the array from (bands, rows, columns) to (rows, columns, bands). Defaults to True.
        cellsize (float, optional): The resolution of the output image in meters. Defaults to None.
        crs (str, optional): The CRS of the output image. Defaults to None.
        transform (tuple, optional): The affine transformation matrix, can be rio.transform() or a tuple like (0.5, 0.0, -180.25, 0.0, -0.5, 83.780361).
            Defaults to None.
        driver (str, optional): The driver to use for creating the output file, such as 'GTiff'. Defaults to "COG".
        colormap (dict, optional): A dictionary defining the colormap (value: (R, G, B, A)).
        **kwargs: Additional keyword arguments to be passed to the rasterio.open() function.
    """

    import numpy as np
    import rasterio
    import xarray as xr
    import rioxarray
    from rasterio.transform import Affine

    if output is None:
        return array_to_memory_file(
            array,
            source,
            dtype,
            compress,
            transpose,
            cellsize,
            crs=crs,
            transform=transform,
            driver=driver,
            colormap=colormap,
            **kwargs,
        )

    if isinstance(array, xr.DataArray):
        if (
            hasattr(array, "rio")
            and (array.rio.crs is not None)
            and (array.rio.transform() is not None)
        ):

            if "latitude" in array.dims and "longitude" in array.dims:
                array = array.rename({"latitude": "y", "longitude": "x"})
            elif "lat" in array.dims and "lon" in array.dims:
                array = array.rename({"lat": "y", "lon": "x"})

            if array.ndim == 2 and ("x" in array.dims) and ("y" in array.dims):
                array = array.transpose("y", "x")
            elif array.ndim == 3 and ("x" in array.dims) and ("y" in array.dims):
                dims = list(array.dims)
                dims.remove("x")
                dims.remove("y")
                array = array.transpose(dims[0], "y", "x")
            if "long_name" in array.attrs:
                array.attrs.pop("long_name")

            array.rio.to_raster(
                output, driver=driver, compress=compress, dtype=dtype, **kwargs
            )
            if colormap:
                write_image_colormap(output, colormap, output)
            return output

    if array.ndim == 3 and transpose:
        array = np.transpose(array, (1, 2, 0))

    out_dir = os.path.dirname(os.path.abspath(output))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ext = os.path.splitext(output)[-1].lower()
    if ext == "":
        output += ".tif"
        driver = "COG"
    elif ext == ".png":
        driver = "PNG"
    elif ext == ".jpg" or ext == ".jpeg":
        driver = "JPEG"
    elif ext == ".jp2":
        driver = "JP2OpenJPEG"
    elif ext == ".tiff":
        driver = "GTiff"
    else:
        driver = "COG"

    if source is not None:
        with rasterio.open(source) as src:
            crs = src.crs
            transform = src.transform
            if compress is None:
                compress = src.compression
    else:
        if cellsize is None:
            raise ValueError("resolution must be provided if source is not provided")
        if crs is None:
            raise ValueError(
                "crs must be provided if source is not provided, such as EPSG:3857"
            )

        if transform is None:
            # Define the geotransformation parameters
            xmin, ymin, xmax, ymax = (
                0,
                0,
                cellsize * array.shape[1],
                cellsize * array.shape[0],
            )
            transform = rasterio.transform.from_bounds(
                xmin, ymin, xmax, ymax, array.shape[1], array.shape[0]
            )
        elif isinstance(transform, Affine):
            pass
        elif isinstance(transform, (tuple, list)):
            transform = Affine(*transform)

        kwargs["transform"] = transform

    if dtype is None:
        # Determine the minimum and maximum values in the array
        min_value = np.min(array)
        max_value = np.max(array)
        # Determine the best dtype for the array
        if min_value >= 0 and max_value <= 1:
            dtype = np.float32
        elif min_value >= 0 and max_value <= 255:
            dtype = np.uint8
        elif min_value >= -128 and max_value <= 127:
            dtype = np.int8
        elif min_value >= 0 and max_value <= 65535:
            dtype = np.uint16
        elif min_value >= -32768 and max_value <= 32767:
            dtype = np.int16
        else:
            dtype = np.float64

    # Convert the array to the best dtype
    array = array.astype(dtype)

    # Define the GeoTIFF metadata
    metadata = {
        "driver": driver,
        "height": array.shape[0],
        "width": array.shape[1],
        "dtype": array.dtype,
        "crs": crs,
        "transform": transform,
    }

    if array.ndim == 2:
        metadata["count"] = 1
    elif array.ndim == 3:
        metadata["count"] = array.shape[2]
    if compress is not None and (driver in ["GTiff", "COG"]):
        metadata["compress"] = compress

    metadata.update(**kwargs)
    # Create a new GeoTIFF file and write the array to it
    with rasterio.open(output, "w", **metadata) as dst:
        if array.ndim == 2:
            dst.write(array, 1)
            if colormap:
                dst.write_colormap(1, colormap)
        elif array.ndim == 3:
            for i in range(array.shape[2]):
                dst.write(array[:, :, i], i + 1)
                if colormap:
                    dst.write_colormap(i + 1, colormap)
    return output
