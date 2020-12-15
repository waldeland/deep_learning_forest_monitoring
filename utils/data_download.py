import numpy as np
import binascii
import os
import multiprocessing
import wget
import zipfile
import requests

def download_sentinel_data(product_identifier, output_location, verbose=1, timeout=10*60, debug=False):
    """
    Download and unzip a tile with satelite data.

    Args:
        product_identifier (string): Product identifier
        output_location (string): Location of where to put unzip output data
        verbose (int): Print progress (verbose==1)
        timeout (int): seconds to timeout wget. On timeout download is restarted up to 10 times
        debug (bool): run on main thread if set to True (timeout is ignored)

    Returns:
        string: Path to SAFE-folder
    """

    tmp_zip_file = os.path.join(output_location,str(os.getpid())+ '_'+ str(binascii.hexlify(os.urandom(16)))+ '_tmp.zip')
    safe_file = os.path.join(output_location, product_identifier+'.SAFE')


    if verbose:
        print('Downloading', product_identifier,get_eocloud_url(product_identifier))

    # Download
    # We need to do this async as it sometimes freezes
    def _download(n_retries=0):
        try:
            wget.download(get_eocloud_url(product_identifier), out=tmp_zip_file, bar=wget.bar_thermometer if verbose else None)
        except Exception as e:
            if n_retries:
                _download(n_retries-1)
            else:
                raise e

    n_retries = 5
    if not debug:
        i = 0
        completed = False
        while i < n_retries and not completed:
            i += 1

            p = multiprocessing.Process(target=_download, daemon=True)
            p.start()
            p.join(timeout=timeout)
            if p.is_alive():
                p.terminate()
                p.join()
                print('Retrying download.',n_retries- i,'retries left.')
                continue
            completed = True

        if not completed:
            raise TimeoutError('Download reached timeout ten times.')

    else:
        _download(n_retries)

    if verbose:
        print('\n')

    if not os.path.isdir(output_location):

        if verbose:
            print('Making directory:', output_location)

        os.makedirs(output_location)

    if verbose:
        print('Unziping', product_identifier)

    with zipfile.ZipFile(tmp_zip_file) as f:
        f.extractall(safe_file)

    os.remove(tmp_zip_file)

    return safe_file

def get_eocloud_url(product_identifier):
    """
    Returns an URL to SAFE-file on eocloud.eu given a product identifier
    Args:
        product_identifier (string): product identifier given by ESA (eg. S2B_MSIL1C_20180516T072619_N0206_R049_T37LCJ_20180516T102848)

    .. Todo:: maybe double check that the base URL is working for all cases
    .. Todo:: write tests

    Returns:

    """

    tile = parse_product_identifier(product_identifier)
    product_level = tile['product_level']
    year = tile['datetime'].astype(object).year
    month = tile['datetime'].astype(object).month
    day = tile['datetime'].astype(object).day
    return "http://185.48.233.249/Sentinel-2/MSI/{}/{}/{:02d}/{:02d}/{}.SAFE".format(product_level, year, month, day, product_identifier)


def download_file_from_google_drive(google_drive_id, destination):
    """
    download files from google drive
    Args:
        google_drive_id (string): for example, given the url:
            https://drive.google.com/uc?id=1YZp2PUR1NYKPlBIVoVRO0Tg1ECDmrnC3&export=download,
            the id is 1YZp2PUR1NYKPlBIVoVRO0Tg1ECDmrnC3
        destination (string): output file


    """
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : google_drive_id}, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : google_drive_id, 'confirm' : token}
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)



def parse_product_identifier(product_identifier):
    """
    Parse the product identifier
    :param product_identifier: (string)
    :return: (dict)
    """

    # Remove extension
    product_identifier = product_identifier.split('.')[0]

    # Add product_identifier
    out = {'product_identifier': product_identifier}

    # Split name into different parts
    name_parts = product_identifier.split('_')

    # Figure out which sentinel (1, 2, or 3)
    out['sentinel_type'] = int(name_parts[0][1])

    if out['sentinel_type'] == 2:
        """
        Sentinel 2: (after  6th of December, 2016)

        MMM_MSIL1C_YYYYMMDDHHMMSS_Nxxyy_ROOO_Txxxxx_<Product Discriminator>.SAFE

        The products contain two dates.

        The first date (YYYYMMDDHHMMSS) is the datatake sensing time.
        The second date is the "<Product Discriminator>" field, which is 15 characters in length, and is used to distinguish between different end user products from the same datatake. Depending on the instance, the time in this field can be earlier or slightly later than the datatake sensing time.

        The other components of the filename are:

        MMM: is the mission ID(S2A/S2B)
        MSIL1C: denotes the Level-1C product level
        YYYYMMDDHHMMSS: the datatake sensing start time
        Nxxyy: the Processing Baseline number (e.g. N0204)
        ROOO: Relative Orbit number (R001 - R143)
        Txxxxx: Tile Number field
        SAFE: Product Format (Standard Archive Format for Europe)"""

        out['product_discriminator'] = _sentinel_datetime_2_np_datetime(name_parts[6])

        # We only support the new format
        # TODO: add support for older sentinel 1 name formats
        if not out['product_discriminator']>np.datetime64('2016-12-06T00:00:00'):
            raise NotImplementedError('parse_eodata_folder_name() does not support sentinel-2 data earlier than 6th of December 2016')

        out['misson_id'] = name_parts[0]
        out['product_level'] = name_parts[1][3:]
        out['datetime'] = _sentinel_datetime_2_np_datetime(name_parts[2])
        out['processing_baseline_number'] = int(name_parts[3][1:])
        out['relative_orbit_number'] = int(name_parts[4][1:])
        out['tile_id'] = name_parts[5]

    elif out['sentinel_type'] == 1:
        """ https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/naming-conventions """
        out['misson_id'] = name_parts[0]

        out['mode'] = name_parts[1]

        out['product_type'] = name_parts[2][0:3]
        out['resolution_class'] = name_parts[2][-1]

        out['processing_level'] = int(name_parts[3][0])
        out['product_class'] = name_parts[3][1]
        out['polarization'] = name_parts[3][2:]

        out['datetime'] = _sentinel_datetime_2_np_datetime(name_parts[4])
        out['start_date'] = _sentinel_datetime_2_np_datetime(name_parts[4])
        out['end_date'] = _sentinel_datetime_2_np_datetime(name_parts[5])

        out['absolute_orbit_number'] = int(name_parts[6][1:])
        out['mission_data_take_id'] = name_parts[7]
        out['product_unique_id'] = name_parts[8]

    elif out['sentinel_type'] == 3:
        # TODO: add support for sentinel 3 name formats
        raise NotImplementedError('parse_eodata_folder_name() does not support sentinel-3 yet')

    return out



def _sentinel_datetime_2_np_datetime(sentinel_datetime_string):
    date, time = sentinel_datetime_string.split('T')
    year = date[0:4]
    month = date[4:6]
    day = date[6:8]
    hour = time[0:2]
    min = time[2:4]
    sec = time[4:6]

    np_datetime_str = year + '-' + month + '-' + day + 'T' + hour + ':' + min + ':' + sec
    return np.datetime64(np_datetime_str)

