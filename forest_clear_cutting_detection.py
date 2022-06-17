import skimage.morphology
import numpy as np
from tqdm import tqdm



def clear_cutting_detection(data, t_threshold=11.7, min_change = 5, filter_on_morphology=True):
    """
    Proposed clear cutting detection scheme
    Args:
        data: numpy array with estimated vegetation height, dimensions are (H x W x N) where N is the number of time-stamps
        t_threshold (float): threshold
        min_change (float): minimum height difference before and after change point for detections

    """

    #Small helping function
    def _take_inds_along_axis_2(data, inds):
        out = data[:, :, 0] * 0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                out[i, j] = data[i, j, inds[i, j]]
        return out


    #Vecotrizes coputation of t-value (vecotrizing over pixels)
    N = np.isfinite(data)

    N_a = np.cumsum(N, -1)
    N_b = np.sum(N, -1, keepdims=True) - N_a

    cumsum_a = np.nancumsum(data, -1)
    cumsum_b = np.flip(np.nancumsum(np.flip(data, -1), -1), -1)

    # Removing index 0 which has no valid mean both before and after
    N_a = N_a[:, :, :-1]
    N_b = N_b[:, :, :-1]
    cumsum_a = cumsum_a[:, :, :-1]
    cumsum_b = cumsum_b[:, :, 1:]

    MU_a = cumsum_a / N_a
    MU_b = cumsum_b / N_b

    VAR_a = np.zeros_like(MU_a, 'float32') * np.nan
    VAR_b = np.zeros_like(MU_a, 'float32') * np.nan

    for ti in tqdm(range(data.shape[-1] - 1), 'Computing sigma', data.shape[-1] - 1):
        VAR_a[:, :, ti] = np.nansum((data[:, :, :ti + 1] - MU_a[:, :, ti:ti + 1]) ** 2, -1) / (N_a[:, :, ti] - 1)
        VAR_b[:, :, ti] = np.nansum((data[:, :, ti + 1:] - MU_b[:, :, ti:ti + 1]) ** 2, -1) / (N_b[:, :, ti] - 1)

    t = (MU_a - MU_b) / (np.sqrt(VAR_a / N_a + VAR_b / N_b))

    #Finding point in time with max t-value
    t = np.concatenate([t, np.zeros_like(t[:,:,0:1])-1],-1) #START: Hack to make code robust to "ValueError: All-NaN slice encountered"
    index_with_max_t = np.nanargmax(t, -1) - 1 # Subtract 1 due to hack
    max_t = np.nanmax(t, -1)

    #Computing vegetation height before and after
    MU_before =  _take_inds_along_axis_2(MU_a, index_with_max_t)
    MU_after =  _take_inds_along_axis_2(MU_b, index_with_max_t)

    #Filtering detections
    detections = max_t > t_threshold
    height_change = MU_before - MU_after
    detections = np.bitwise_and(detections, height_change > min_change)

    if filter_on_morphology:
        detections = skimage.morphology.remove_small_objects(detections, min_size=10, connectivity=1, in_place=False)

    return detections, (max_t, height_change, index_with_max_t)

if __name__ == '__main__':
    from predict_scene import predict_scene
    import os, rasterio

    ########################################
    ##########      INPUTS      ############
    ########################################
    # Replace this list with your own list of S2 product identifiers, ordered by sensing-date and all covering the same tileid (e.g. T35MNR)
    s2_ids = [
        "S2A_MSIL1C_20170401T082001_N0204_R121_T35MNR_20170401T083654",
        "S2A_MSIL1C_20170411T081601_N0204_R121_T35MNR_20170411T083418",
        "S2A_MSIL1C_20170421T082011_N0204_R121_T35MNR_20170421T083236",
        "S2A_MSIL1C_20170501T081611_N0205_R121_T35MNR_20170501T083422",
        "S2A_MSIL1C_20170511T082011_N0205_R121_T35MNR_20170511T083238",
        "S2A_MSIL1C_20170521T081611_N0205_R121_T35MNR_20170521T083423",
        "S2A_MSIL1C_20170531T082011_N0205_R121_T35MNR_20170531T083237",
        "S2A_MSIL1C_20170610T081601_N0205_R121_T35MNR_20170610T083432",
        "S2A_MSIL1C_20170710T082011_N0205_R121_T35MNR_20170710T083235",
        "S2B_MSIL1C_20170715T081609_N0205_R121_T35MNR_20170715T083438",
        "S2A_MSIL1C_20170730T082011_N0205_R121_T35MNR_20170730T083232",
        "S2B_MSIL1C_20170814T082009_N0205_R121_T35MNR_20170814T083232",
        "S2B_MSIL1C_20170824T081559_N0205_R121_T35MNR_20170824T083416",
        "S2A_MSIL1C_20170918T081601_N0205_R121_T35MNR_20170918T083613",
        "S2B_MSIL1C_20170923T081959_N0205_R121_T35MNR_20170923T083322",
        "S2A_MSIL1C_20171008T081831_N0205_R121_T35MNR_20171008T083804",
        "S2B_MSIL1C_20171112T082149_N0206_R121_T35MNR_20171112T103922",
        "S2B_MSIL1C_20171222T082329_N0206_R121_T35MNR_20171222T105215",
        "S2A_MSIL1C_20180116T082251_N0206_R121_T35MNR_20180116T120855",
        "S2A_MSIL1C_20180307T081801_N0206_R121_T35MNR_20180307T102747",
        "S2B_MSIL1C_20180322T081619_N0206_R121_T35MNR_20180322T111859",
        "S2A_MSIL1C_20180327T081601_N0206_R121_T35MNR_20180327T120752",
        "S2B_MSIL1C_20180401T081559_N0206_R121_T35MNR_20180401T103327",
        "S2B_MSIL1C_20180421T081559_N0206_R121_T35MNR_20180421T103900",
        "S2A_MSIL1C_20180426T081701_N0206_R121_T35MNR_20180426T102334",
        "S2B_MSIL1C_20180501T081559_N0206_R121_T35MNR_20180501T115720",
        "S2A_MSIL1C_20180526T081601_N0206_R121_T35MNR_20180526T120942",
        "S2B_MSIL1C_20180531T081559_N0206_R121_T35MNR_20180531T103214",
        "S2A_MSIL1C_20180605T081601_N0206_R121_T35MNR_20180605T102730",
        "S2B_MSIL1C_20180610T081559_N0206_R121_T35MNR_20180610T103202",
        "S2A_MSIL1C_20180615T081601_N0206_R121_T35MNR_20180615T103511",
        "S2B_MSIL1C_20180620T081859_N0206_R121_T35MNR_20180620T134542",
        "S2A_MSIL1C_20180625T081601_N0206_R121_T35MNR_20180625T150304",
        "S2B_MSIL1C_20180630T081559_N0206_R121_T35MNR_20180630T124029",
        "S2A_MSIL1C_20180705T081601_N0206_R121_T35MNR_20180705T103349",
        "S2B_MSIL1C_20180710T081559_N0206_R121_T35MNR_20180710T115338",
        "S2A_MSIL1C_20180715T081601_N0206_R121_T35MNR_20180715T103432",
        "S2B_MSIL1C_20180720T081559_N0206_R121_T35MNR_20180720T121127",
        "S2B_MSIL1C_20180730T081559_N0206_R121_T35MNR_20180730T141111",
        "S2A_MSIL1C_20180804T081601_N0206_R121_T35MNR_20180804T103644",
        "S2A_MSIL1C_20180814T081601_N0206_R121_T35MNR_20180814T114118",
        "S2A_MSIL1C_20180923T081641_N0206_R121_T35MNR_20180923T122158",
        "S2B_MSIL1C_20181008T081819_N0206_R121_T35MNR_20181008T140253",
        "S2A_MSIL1C_20181023T082001_N0206_R121_T35MNR_20181023T103940",
        "S2B_MSIL1C_20190126T082219_N0207_R121_T35MNR_20190126T111237",
        "S2A_MSIL1C_20190131T082201_N0207_R121_T35MNR_20190131T103522",
        "S2B_MSIL1C_20190225T081919_N0207_R121_T35MNR_20190225T120920",
        "S2A_MSIL1C_20190322T081621_N0207_R121_T35MNR_20190322T110719",
        "S2B_MSIL1C_20190406T081609_N0207_R121_T35MNR_20190406T120341",
        "S2B_MSIL1C_20190416T081609_N0207_R121_T35MNR_20190416T120731",
        "S2B_MSIL1C_20190506T081609_N0207_R121_T35MNR_20190506T120436",
        "S2A_MSIL1C_20190511T081611_N0207_R121_T35MNR_20190511T103549",
        "S2B_MSIL1C_20190516T081609_N0207_R121_T35MNR_20190516T120542",
        "S2A_MSIL1C_20190521T081611_N0207_R121_T35MNR_20190521T120310",
        "S2B_MSIL1C_20190526T081609_N0207_R121_T35MNR_20190526T120530",
        "S2B_MSIL1C_20190605T081609_N0207_R121_T35MNR_20190605T120937",
        "S2A_MSIL1C_20190610T081611_N0207_R121_T35MNR_20190610T102045",
        "S2B_MSIL1C_20190615T081609_N0207_R121_T35MNR_20190615T120344",
        "S2A_MSIL1C_20190620T081611_N0207_R121_T35MNR_20190620T120102",
        "S2B_MSIL1C_20190625T081609_N0207_R121_T35MNR_20190625T120443",
        "S2B_MSIL1C_20190705T081609_N0207_R121_T35MNR_20190705T110801",
        "S2B_MSIL1C_20190715T081609_N0208_R121_T35MNR_20190715T121541",
        "S2B_MSIL1C_20190725T081609_N0208_R121_T35MNR_20190725T120407",
        "S2B_MSIL1C_20190804T081609_N0208_R121_T35MNR_20190804T111436",
        "S2B_MSIL1C_20190814T081609_N0208_R121_T35MNR_20190814T120408",
        "S2A_MSIL1C_20190829T081601_N0208_R121_T35MNR_20190829T110655",
        "S2A_MSIL1C_20190908T081601_N0208_R121_T35MNR_20190908T102009",
        "S2A_MSIL1C_20200116T082301_N0208_R121_T35MNR_20200116T102258",
        "S2A_MSIL1C_20200306T081811_N0209_R121_T35MNR_20200306T101910",
        "S2B_MSIL1C_20200331T081559_N0209_R121_T35MNR_20200331T111713",
        "S2B_MSIL1C_20200410T081559_N0209_R121_T35MNR_20200410T120328",
        "S2A_MSIL1C_20200415T081601_N0209_R121_T35MNR_20200415T120410",
        "S2B_MSIL1C_20200430T081559_N0209_R121_T35MNR_20200430T111404",
        "S2A_MSIL1C_20200505T081611_N0209_R121_T35MNR_20200505T101953",
        "S2B_MSIL1C_20200510T081559_N0209_R121_T35MNR_20200510T120518",
        "S2B_MSIL1C_20200520T081609_N0209_R121_T35MNR_20200520T120627",
        "S2A_MSIL1C_20200525T081611_N0209_R121_T35MNR_20200525T104047",
        "S2B_MSIL1C_20200530T081609_N0209_R121_T35MNR_20200530T112043",
        "S2A_MSIL1C_20200604T081611_N0209_R121_T35MNR_20200604T101958",
        "S2B_MSIL1C_20200609T081609_N0209_R121_T35MNR_20200609T120443",
        "S2A_MSIL1C_20200614T081611_N0209_R121_T35MNR_20200614T111228",
        "S2A_MSIL1C_20200624T081611_N0209_R121_T35MNR_20200624T103958",
        "S2B_MSIL1C_20200629T081609_N0209_R121_T35MNR_20200629T111600",
        "S2A_MSIL1C_20200704T081611_N0209_R121_T35MNR_20200704T101923",
        "S2B_MSIL1C_20200709T081609_N0209_R121_T35MNR_20200709T112259",
        "S2B_MSIL1C_20200719T081609_N0209_R121_T35MNR_20200719T120434",
        "S2A_MSIL1C_20200803T081611_N0209_R121_T35MNR_20200803T095004",
        "S2B_MSIL1C_20200917T081609_N0209_R121_T35MNR_20200917T103015",
        "S2B_MSIL1C_20201116T082219_N0209_R121_T35MNR_20201116T121219",
    ]

    # Speed up computations by selecting a subcrop
    # y_start, y_stop, x_start, x_stop
    subcrop = [9000,9300, 5500,5800]

    # Where to put vegetation height predictions + clear cutting detections
    output_path = '.'
    ########################################

    # Estimate vegetation height
    for pid in s2_ids:
        if not os.path.isfile(os.path.join(output_path, pid+'.tif')):
            predict_scene(pid, output_path, subcrop=subcrop)

    # Load vegetation height predictions
    data_cube = []
    for pid in s2_ids:
        with rasterio.open(os.path.join(output_path, pid+'.tif')) as f:
            data_cube.append(np.squeeze(f.read())[:,:,None])
    data_cube = np.concatenate(data_cube,-1)
    data_cube[data_cube==-1] = np.nan

    # Run change detection
    detections, _ = clear_cutting_detection(data_cube, t_threshold=11.7, min_change=5)

    # Export:
    with rasterio.open(os.path.join(output_path,  s2_ids[0]+'.tif'), 'r') as src:
        with rasterio.open(
            os.path.join(output_path,  'clear_cutting_mask.tif'),
            "w",
            driver="GTiff",
            compress="lzw",
            bigtiff="YES",
            height=detections.shape[0],
            width=detections.shape[1],
            dtype=np.float32,
            count=1,
            crs=src.crs,
            transform=src.transform,
        ) as out_file:
            out_file.write(detections.astype('float32'), indexes=1)
