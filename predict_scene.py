import argparse

import os
import rasterio
import torch

import numpy as np
from rasterio._base import Affine
from rasterio.crs import CRS

from utils.data_download import download_file_from_google_drive, download_sentinel_data
from utils.process_safe_file import convert_sentinel2
from utils.tiled_prediction import tiled_prediction
from utils.unet import UNet

#Download trained model
_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pt')
if not os.path.isfile(_model_path):
    download_file_from_google_drive('1Y7PQepK36D2mSUIg0GhF-V1E7ZA-_dZS', _model_path)




def predict_scene(product_id, output_path, subcrop=None):
    """

    Args:
        product_id:
        output_path:
        slice:

    Returns:

    """

    model = UNet(n_classes=1, in_channels=13, start_filts=32, use_bn=True, partial_conv=True)
    model.load_state_dict(torch.load(_model_path, map_location=lambda storage, loc: storage))
    model = model.cuda()
    model.eval()


    path_to_safe_folder = download_sentinel_data(product_id, output_path)
    data_cube, cloud_mask, transform, crs = convert_sentinel2(path_to_safe_folder)

    no_data_mask = np.isnan(data_cube)
    data_cube[no_data_mask] = 0

    if subcrop is not None:
        data_cube = data_cube[subcrop[0]:subcrop[1], subcrop[2]:subcrop[3], :]
        no_data_mask = no_data_mask[subcrop[0]:subcrop[1], subcrop[2]:subcrop[3], :]
        transform = np.array(transform)
        transform[2] = transform[2] + subcrop[0]*transform[0]
        transform[5] = transform[5] + subcrop[2]*transform[4]
        transform = Affine(*transform[:6])

    print('Predicting')
    tree_height = tiled_prediction(data_cube, model, [512, 512], [64, 64]).squeeze()
    tree_height = np.clip(tree_height, 0, np.inf)
    tree_height[np.sum(no_data_mask,2)>0] = -1

    print('Exporting to', product_id + '.tif')
    if output_path is not None:
        with rasterio.open(
            os.path.join(output_path, product_id + '.tif'),
            "w",
            driver="GTiff",
            compress="lzw",
            bigtiff="YES",
            height=tree_height.shape[0],
            width=tree_height.shape[1],
            dtype=np.float32,
            count=1,
            crs=crs,
            transform=transform,
            nodata=-1,
        ) as out_file:
            out_file.write(tree_height.astype('float32'), indexes=1)

    return tree_height


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-p', '--product',default='S2A_MSIL1C_20170126T073151_N0204_R049_T37MDQ_20170126T074339', type=str ,help='Sentinel-2 product identifier')
    p.add_argument('-o', '--output', type=str, default='.' , help='Path to folder for output')
    p = p.parse_args()
    predict_scene(p.product, p.output)




