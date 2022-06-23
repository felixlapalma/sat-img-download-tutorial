import numpy as np

#
import planetary_computer as pc

#
import rasterio as rio
import rasterio.mask as rasterio_mask
import rasterio.merge as rasterio_merge
from rasterio import Affine, MemoryFile, warp, windows
from rasterio.enums import Resampling
from shapely.geometry import box


def write_raster(file_path, data, **profile):
    with rio.open(file_path, "w", **profile) as dataset:  # Open as DatasetWriter
        dataset.write(data)

    return rio.open(file_path)


def write_mem_raster(data, **profile):
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:  # Open as DatasetWriter
            dataset.write(data)
        return memfile.open()


def merge_raster_list(raster_list):
    """merge raster list

    Parameters
    ----------
        raster_list: list
            list of rasters open instances
    Returns
    -------
        raster: rasterio dataset
            merged raster

    """
    data, data_transform = rasterio_merge.merge(raster_list)

    # assemble raster
    profile = raster_list[0].profile.copy()
    profile.update(
        {
            "transform": data_transform,
            "width": data.shape[2],
            "height": data.shape[1],
        }
    )
    return write_mem_raster(data, **profile)


def _get_band_cog(url, bbox, **kwargs):
    """
    Get Single Band data with the required resolution

    Parameters
    ----------
        url: str
            path to cog band required
        bbox: list
            bounding box as [left, bottom, right, top] (return from rasterio.features.bounds)
            in epsg:4326
        kwargs: dict
            dict for further customization of rasterio.mask.mask

    Returns
    -------
        band_data: numpy flattened array
            band data
        profile: dict
            raster profile (transforms,driver,etc)

    """
    crop = kwargs.get("crop", True)
    filled = kwargs.get("filled", False)
    all_touched = kwargs.get("all_touched", True)
    nodata = kwargs.get("nodata", 0)

    with rio.open(url) as ds:
        profile = ds.profile.copy()
        warped_aoi_bounds = warp.transform_bounds("epsg:4326", ds.crs, *bbox)
        # make shape
        shape_bounds = [get_polygon_from_bbox(warped_aoi_bounds)]
        band_data, data_transform = rasterio_mask.mask(
            ds,
            shape_bounds,
            crop=crop,
            nodata=nodata,
            all_touched=all_touched,
            filled=filled,
        )
        profile.update(
            {
                "height": band_data.shape[1],
                "width": band_data.shape[2],
                "transform": data_transform,
            }
        )
    return band_data, profile


def _get_bands_data(item, bands_list, bbox, sign_query=True):
    """purpose

    Parameters
    ----------
        band_list

    """
    bands_data = {}
    bands_profile = {}
    aoi_bounds = bbox  # rasterio.features.bounds

    for b in bands_list:
        asset_href = item.assets[b].href
        if sign_query:
            signed_href = pc.sign(asset_href)
        else:
            # Auth may be through other method
            signed_href = asset_href
        #
        with rio.open(signed_href) as ds:
            profile = ds.profile.copy()
            warped_aoi_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)
            # get window
            aoi_window = windows.from_bounds(transform=ds.transform, *warped_aoi_bounds)
            # get data for window - Notice boundless Flag
            band_data = ds.read(window=aoi_window, boundless=True)
            #
            window_transform = rio.windows.transform(aoi_window, ds.transform)
            profile.update(
                {
                    "height": band_data.shape[1],
                    "width": band_data.shape[2],
                    "transform": window_transform,
                }
            )
        bands_data[b] = band_data.squeeze()
        bands_profile[b] = profile
    return bands_data, bands_profile


def _get_bands_data_rmask(
    item, bands_list, bbox, sign_query=True, crop=True, filled=False
):
    """purpose

    Parameters
    ----------
        band_list

    """
    bands_data = {}
    bands_profile = {}
    aoi_bounds = bbox  # rasterio.features.bounds

    for b in bands_list:
        asset_href = item.assets[b].href
        if sign_query:
            signed_href = pc.sign(asset_href)
        else:
            # Auth may be through other method
            signed_href = asset_href
        # Call _get_band_cog
        band_data, profile = _get_band_cog(
            signed_href, aoi_bounds, **{"crop": crop, "filled": filled}
        )
        #
        bands_data[b] = band_data.squeeze()
        bands_profile[b] = profile
    return bands_data, bands_profile


def raster_resample_check(raster_res, target_resolution):
    """
    Check if resampling is required.

    Parameters
    ----------
    raster_res: tuple
        raster resolution as tuple (raster.res for instance, if raster is rasterio.open case)
    target_resolution: float
        Required resolution in meters.

    Returns
    -------
    resample: bool
        True if resample is required
    scale_factor: float
        Calculated scale factor
    true_resolution: float
        Original pixel size
    """
    assert raster_res[0] == raster_res[1]  # square pixel
    true_resolution = raster_res[0]
    if not np.isclose(true_resolution, target_resolution):
        scale_factor = target_resolution / true_resolution
        resample_ = True
    else:
        scale_factor = 1
        resample_ = False
    return resample_, scale_factor, true_resolution


def raster_resample(raster, scale, close=False):
    """
    Resample a raster dataset by multiplying the pixel size by the scale factor,
    and update the dataset shape.
    For example, given a pixel size of 250m, dimensions of (1024, 1024) and
    a scale of 2, the resampled raster will have an output pixel size of 500m
    and dimensions of (512, 512) given a pixel size of 250m, dimensions of
    (1024, 1024) and a scale of 0.5, the resampled raster would have an output
    pixel size of 125m and dimensions of (2048, 2048) returns a `DatasetReader`
    instance in MemoryFile.

    Parameters
    ----------
        raster: raster.open instance
        scale: float
            scale to apply to raster
        close: bool
            close provided raster dataset

    Returns
    -------
        resampled raster: opened rasterio.MemoryFile
    """
    t = raster.transform
    transform = Affine(t.a * scale, t.b, t.c, t.d, t.e * scale, t.f)
    height = int(raster.height / scale)
    width = int(raster.width / scale)

    profile = raster.profile
    profile.update(transform=transform, driver="GTiff", height=height, width=width)

    data = raster.read(
        out_shape=(raster.count, height, width),
        resampling=Resampling.nearest,
    )

    if close:
        raster.close()
    return write_mem_raster(data, **profile)


def _assemble_raster(bands_dict_data, profiles_dict_data, bands_req):

    _base_bands_data = []
    for band in bands_req:
        if bands_dict_data[band].ndim == 2:
            _base_bands_data.append(bands_dict_data[band])
        else:
            _base_bands_data.append(bands_dict_data[band][0, :, :])
    _base_profiles_data = [profiles_dict_data[band] for band in bands_req]
    #
    data = np.stack(_base_bands_data, axis=0)
    #
    profile = _base_profiles_data[0].copy()
    profile.update({"count": len(_base_profiles_data)})
    return write_mem_raster(data, **profile)


def get_polygon_from_bbox(bounds, ccw=True):
    """
    Return box polygon from bounds

    Parameters
    ----------
    bounds: bound as list (as returned by rasterio.features.bounds)
    ccw: bool
        Counter-clockwise order (True) or not (False)

    Returns
    -------
    Polygon: shapely.geometry.box
    """
    return box(*bounds, ccw=ccw)


def get_contains_intersect(item_bbox, roi_bbox):
    """ """
    contains_ = False
    intersect_ = False
    scene_bbox = get_polygon_from_bbox(item_bbox)
    contains_ = scene_bbox.contains(roi_bbox)
    intersect_ = scene_bbox.intersects(roi_bbox)

    if intersect_ and contains_:
        full_match = True
    elif intersect_:
        full_match = False
    else:
        full_match = None

    return full_match


def get_stac_items(kwargs_not_unwrap):
    """
    Parameters
    ----------
        catalog: STAC client to be queryed
        bbox: bound as tuple or list
        time_of_interest: date range as str,e.g., "2021-01-01/2021-01-04"
        cloud_cover: int, maximum cloud cover accepted

    Returns
    -------
        list of stac items (if any)
    """
    catalog = kwargs_not_unwrap.get("catalog", None)
    bbox = kwargs_not_unwrap.get("bbox", None)
    time_of_interest = kwargs_not_unwrap.get("time_of_interest", None)
    cloud_cover = kwargs_not_unwrap.get("cloud_cover", None)

    # we require catalog, area and time of interest
    if catalog is None or bbox is None or time_of_interest is None:

        return []

    # Clouds
    if cloud_cover is None:
        cloud_cover = 100

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=time_of_interest,
        query={"eo:cloud_cover": {"lt": cloud_cover}},
    )
    return list(search.get_items())


def _get_bands_data_from_planetary(
    item, bands_list, bbox, resolution=10, sign_query=True, crop=True, filled=False
):
    """purpose

    Parameters
    ----------
        band_list

    """
    bands_data = {}
    bands_profile = {}
    aoi_bounds = bbox  # rasterio.features.bounds

    for b in bands_list:
        asset_href = item.assets[b].href
        if sign_query:
            signed_href = pc.sign(asset_href)
        else:
            # Auth may be through other method
            signed_href = asset_href
        # call _get_band_cog
        band_data, profile = _get_band_cog(
            signed_href, aoi_bounds, **{"crop": crop, "filled": filled}
        )
        # Temporal
        rasteri = write_mem_raster(band_data, **profile)
        # check resample
        resample_, scale_factor, true_resolution = raster_resample_check(
            rasteri.res, resolution
        )
        # resample if needed (this not assure size consistency between bands)
        if resample_:
            rasteri = raster_resample(rasteri, scale_factor, close=True)
            band_data, profile = rasteri.read(), rasteri.profile
        bands_data[b] = band_data
        bands_profile[b] = profile

    # Now assure consistency
    # Arbitrarily take one as base
    b_base = bands_list[0]
    template_ds = write_mem_raster(bands_data[b_base], **bands_profile[b_base])
    for b in bands_list[1:]:
        src = write_mem_raster(bands_data[b], **bands_profile[b])
        raster_rep = reproject_with_raster_template(src, template_ds)
        # update data
        bands_data[b], bands_profile[b] = raster_rep.read(), raster_rep.profile
        raster_rep.close()
    template_ds.close()

    return bands_data, bands_profile


def reproject_with_raster_template(src, template_ds, **kwargs):
    """reproject raster with template"""

    src_nodata = kwargs.get("src_nodata", 0)
    dst_nodata = kwargs.get("dst_nodata", 0)

    out_kwargs = template_ds.profile.copy()

    dst_crs = template_ds.crs
    dst_transform = template_ds.transform
    dst_height = template_ds.height
    dst_width = template_ds.width

    out_kwargs.update(
        crs=dst_crs,
        transform=dst_transform,
        width=dst_width,
        height=dst_height,
        nodata=dst_nodata,
    )

    # Adjust block size if necessary.
    if "blockxsize" in out_kwargs and dst_width < int(out_kwargs["blockxsize"]):
        del out_kwargs["blockxsize"]
    if "blockysize" in out_kwargs and dst_height < int(out_kwargs["blockysize"]):
        del out_kwargs["blockysize"]

    with MemoryFile() as memfile:
        with memfile.open(**out_kwargs) as out_raster:
            warp.reproject(
                source=rio.band(src, list(range(1, src.count + 1))),
                destination=rio.band(out_raster, list(range(1, src.count + 1))),
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=out_kwargs["transform"],
                dst_crs=out_kwargs["crs"],
                dst_nodata=dst_nodata,
                resampling=Resampling.nearest,
            )
        return memfile.open()


def calibrate_s2_scl(
    raster,
    filter_values,
    init_value=0,
    nodata=0,
    close=False,
):
    """
    Calibrate Sentinel2 SCL band.
    Parameters
    ----------
    raster:
        raster instance opened by rasterio
    filter_values: list or tuple
        S2-SCL: [0,1,2,4,5,6,7,11]
    close: bool
        Close the input raster dataset before returning the calibrated raster.
    Returns
    -------
    returns a DatasetReader instance from a MemoryFile.
    REF: https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm
    0 	NO_DATA
    1 	SATURATED_OR_DEFECTIVE
    2 	DARK_AREA_PIXELS
    3 	CLOUD_SHADOWS
    4 	VEGETATION
    5 	NOT_VEGETATED
    6 	WATER
    7 	UNCLASSIFIED
    8 	CLOUD_MEDIUM_PROBABILITY
    9 	CLOUD_HIGH_PROBABILITY
    10 	THIN_CIRRUS
    11 	SNOW
    """
    profile = raster.profile
    profile.update({"dtype": rio.ubyte, "driver": "GTiff"})
    data = raster.read()
    mask_nodata = data == nodata
    assert len(filter_values) > 0, "At least one value should be provided for filtering"
    mask_ = data != filter_values[0]
    for value in filter_values[1:]:
        mask_ = mask_ & (data != value)
    data_cloud = np.where(mask_, True, init_value)
    data_cloud = np.where(mask_nodata, False, data_cloud)
    if close:
        raster.close()
    return write_mem_raster(data_cloud.astype(rio.ubyte), **profile)
