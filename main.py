import os
import czifile
import xmltodict
import numpy as np
from tqdm import tqdm
from tifffile import imwrite
from skimage.util import img_as_ubyte, img_as_uint
from skimage.exposure import rescale_intensity, adjust_gamma

SHORTNAME_MAPPING = {'AF350': 1,
                   'AF405': 2,
                   'AF430': 3,
                   'AF480': 4,
                   'AF546': 5,
                   'AF594': 6,
                   'AF647': 7,
                   'PCP55': 8,
                   'AF700': 9,
                   'I800r': 10,
                   'PhaCo': 11}


def hex_to_rgb(hex):
    return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))


def export_czi(czi_filename, output_dir=None, round_number=None, dtype='default'):
    """"
    Export CZI file to TIFF files
    :param czi_filename: path to CZI file
    :param output_dir: path to output directory
    :param round_number: round number to export
    :param dtype: data type of exported TIFF files (default | uint8 | uint16)
    """

    # output directory
    if output_dir is None:
        output_dir = os.path.dirname(czi_filename)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # output filename prefix
    if round_number is None:
        base_name = os.path.splitext(os.path.basename(czi_filename))[0]
    else:
        base_name = "R" + str(round_number)

    # read CZI file
    czi = czifile.CziFile(czi_filename)

    # get dtype
    if dtype == 'default':
        dtype = czi.dtype
    elif dtype == 'uint8':
        dtype = np.uint8
    elif dtype == 'uint16':
        dtype = np.uint16

    # get metadata
    metadata = xmltodict.parse(czi.metadata())

    # get image dimensions
    array = np.squeeze(czi.asarray())

    # for each channel
    for ch in tqdm(range(array.shape[0]), leave=False):

        image = array[ch, :, :]
        channel_metadata = metadata['ImageDocument']['Metadata']['DisplaySetting']['Channels']['Channel'][ch]

        # get display histogram inforamtion
        display_min = float(channel_metadata.get('Low', 0))
        display_max = float(channel_metadata.get('High', 1))
        gamma = float(channel_metadata.get('Gamma', 1))

        # rescale intensity
        if display_min != 0.0 or display_max != 1.0:

            # convert display min/max accordingly to the dtype
            if image.dtype == np.uint8:
                display_min = int(np.round(display_min * 255))
                display_max = int(np.round(display_max * 255))
            elif image.dtype == np.uint16:
                display_min = int(np.round(display_min * 65535))
                display_max = int(np.round(display_max * 65535))

            image = rescale_intensity(image, in_range=(display_min, display_max))

        # adjust gamma
        if gamma != 1:
            image = adjust_gamma(image, gamma)

        # set dtype
        if dtype == np.uint8:
            image = img_as_ubyte(image)
        elif dtype == np.uint16:
            image = img_as_uint(image)

        # create colormap
        if dtype == np.uint8:                   # colormap only works for uint8
            color = hex_to_rgb(channel_metadata.get('Color', "#FFFFFFFF").lstrip("#")[2:])   # first two chars are alpha
            colormap = np.zeros((3, 256), dtype=np.uint8)
            colormap[0, :] = np.linspace(0, color[0], dtype=np.uint8, num=256)
            colormap[1, :] = np.linspace(0, color[1], dtype=np.uint8, num=256)
            colormap[2, :] = np.linspace(0, color[2], dtype=np.uint8, num=256)
        else:
            colormap = None

        # get channel number
        channel_number = SHORTNAME_MAPPING[channel_metadata['ShortName']]

        # save image
        imwrite(os.path.join(output_dir, "{}C{}.tif".format(base_name, channel_number)), image,
                        colormap=colormap)


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description='Export CZI file to TIFF files')
    parser.add_argument('-i', '--input_dir', default=None,  help='path to input directory containing CZI files')
    parser.add_argument('-o', '--output_dir', default=None, help='path to output directory')
    parser.add_argument('-d', '--dtype', default='default', help='data type of exported TIFF files (default | uint8 | uint16)')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    for file in tqdm(os.listdir(args.input_dir)):
        if file.endswith(".czi"):

            round_number = int(file[-6])
            export_czi(os.path.join(args.input_dir, file), output_dir=args.output_dir, dtype=args.dtype, round_number=round_number)
