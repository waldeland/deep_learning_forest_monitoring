import datetime
import warnings
import fnmatch
import os
import time
import numpy as np
import rasterio
import shutil
from rasterio._warp import Resampling
from rasterio.warp import reproject
from s2cloudless import S2PixelCloudDetector


_bands_10m= ["b02", "b03", "b04", "b08"]
_bands_20m= ["b05", "b06", "b07", "b8a", "b11", "b12"]
_bands_60m= ["b01", "b09", "b10"]
_all_bands = _bands_10m + _bands_20m + _bands_60m
_bands_used_in_cloud_detection = ['b01', 'b02', 'b04', 'b05', 'b08', 'b8a', 'b09', 'b10', 'b11', 'b12']

def convert_sentinel2(path_to_SAFE_folder,
                      output_resolution=10,
                      cloud_detection_resolution = 60,
                      n_threads = 10,
                      delete_safe_file=True):

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    if path_to_SAFE_folder[-1] == '/':
        path_to_SAFE_folder = path_to_SAFE_folder[:-1]
    tile_name = path_to_SAFE_folder.split('/')[-1].replace('.SAFE','')

    print('Processing:', tile_name )

    #Explore content in folder to find image-folder:
    input_dir = os.path.join(path_to_SAFE_folder, "GRANULE")
    sub_directories = [name for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name))]
    image_dir = os.path.join(input_dir, sub_directories[0], "IMG_DATA")

    #Read all band
    scale_factor = 10000
    band_data = {}
    t = time.time()
    time_left = 0


    for img_dir, dirnames, filenames in os.walk(image_dir):
        for band_ind, band_name in enumerate(_all_bands):
            if band_name in _bands_10m:
                resolution = 10
            elif band_name in _bands_20m:
                resolution = 20
            elif band_name in _bands_60m:
                resolution = 60



            for filename in filenames:

                #L2a products have multiple resolutions for each band.
                l2a_prod_res = filename.split('.jp2')[0].split('_')[-1]
                if l2a_prod_res in ['10m', '20m','60m']:
                    l2a_prod_res = int(l2a_prod_res.split('m')[0])
                    if l2a_prod_res > output_resolution and l2a_prod_res > resolution:
                        continue

                if fnmatch.filter([filename], "*" + band_name.upper() + "*.jp2"):
                    path_to_jp2 = os.path.join(img_dir, filename)

                    # Assuming that os.walk loop through folders alphabetically, this ensure that the higest resolution is chosen if there exist multiple resolutions pr file
                    if band_name not in band_data:
                        print(' - Reading', band_name, '-', len(band_data)+1,'of', len(_all_bands), '-', 'Estimated time left:', str(datetime.timedelta(seconds=time_left)))


                        with rasterio.open(path_to_jp2) as ds:
                            band_data[band_name] = {'data': ds.read()/scale_factor,
                                                    'transform':ds.transform,
                                                    'crs':ds.crs,
                                                    'res':resolution}

                        time_left = (time.time() - t) / (len(band_data)) * (len(_all_bands) - len(band_data));

    #Collect sentral parameters for each resolution
    transforms = {
        10: band_data[_bands_10m[0]]['transform'],
        20: band_data[_bands_20m[0]]['transform'],
        60: band_data[_bands_60m[0]]['transform'],
    }
    crss = {
        10: band_data[_bands_10m[0]]['crs'],
        20: band_data[_bands_20m[0]]['crs'],
        60: band_data[_bands_60m[0]]['crs'],
    }
    shapes = {
        10: band_data[_bands_10m[0]]['data'].shape,
        20: band_data[_bands_20m[0]]['data'].shape,
        60: band_data[_bands_60m[0]]['data'].shape,
    }
    target_transform =  transforms[output_resolution]
    target_crs= crss[output_resolution]

    #Reprojections
    def reproject_to_resolution(data, res_from, res_to):

        if res_to==res_from:
            return data
        tmp_array = np.zeros(shapes[res_to])
        reproject(data, tmp_array,
                  src_transform=transforms[res_from],
                  dst_transform=transforms[res_to],
                  src_crs=crss[res_from],
                  dst_crs=crss[res_to],
                  num_threads=n_threads,
                  resampling=Resampling.bilinear)

        return tmp_array

    ####################################################################################################################
    # Perform cloud detection using
    # https://github.com/sentinel-hub/sentinel2-cloud-detector/blob/master/examples/sentinel2-cloud-detector-example.ipynb

    #Convert all bands to 60m for cloud-detection
    print(" - Cloud detection running")
    bands_for_cloud_detection = [
        reproject_to_resolution(band_data[band_name]['data'], band_data[band_name]['res'], cloud_detection_resolution) for band_name in _bands_used_in_cloud_detection
    ]
    bands_for_cloud_detection = np.concatenate([np.expand_dims(np.squeeze(b),-1) for b in bands_for_cloud_detection], -1)
    bands_for_cloud_detection = np.expand_dims(bands_for_cloud_detection,0)
    cloud_detector = S2PixelCloudDetector(threshold=0.4, all_bands=False, average_over=4, dilation_size=2)
    cloud_mask = cloud_detector.get_cloud_masks(bands_for_cloud_detection)[0]

    #Save to output resolution
    cloud_mask = reproject_to_resolution(cloud_mask, cloud_detection_resolution, output_resolution)


    print(" - Cloud detection finished")

    # Save memory maps
    print(" - Creating 10m resolution data-cube")
    bands = ["b02", "b03", "b04", "b08", "b05", "b06", "b07", "b8a", "b11", "b12", "b01", "b09", "b10"]
    bands_data = []
    for band_name in bands:
        data_at_res = reproject_to_resolution( band_data[band_name]['data'], band_data[band_name]['res'], output_resolution)
        data_at_res[data_at_res==0] = np.nan #Insert np.nan as nodata value
        bands_data.append(data_at_res.squeeze().astype('float16'))

    bands_data = [d[:, :, None] for d in bands_data]
    bands_data = np.concatenate(bands_data, -1)

    if delete_safe_file:
        print(' - Deleting SAFE file', path_to_SAFE_folder)
        shutil.rmtree(path_to_SAFE_folder)

    warnings.filterwarnings("default", category=DeprecationWarning)

    return bands_data, cloud_mask, target_transform, target_crs






