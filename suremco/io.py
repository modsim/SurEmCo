# SurEmCo - IO routines

import os

import numpy
import pandas


def read_magic_number(filename):
    with open(filename, 'rb') as fp:
        return fp.read(4)


def is_png(magic_number):
    return magic_number == b'\x89PNG'


def is_gif(magic_number):
    return magic_number == b'GIF8'  # GIF89a GIF87a


def is_jpeg(magic_number):
    return magic_number[:3] == b'\xff\xd8\xff'


def is_tiff(magic_number):
    return (
        (magic_number == b'II\x2a\x00') or
        (magic_number == b'MM\x00\x2a')
    )


def is_image_file(magic_number=None, filename=None):
    if filename is not None:
        magic_number = read_magic_number(filename)
    return is_png(magic_number) or is_gif(magic_number) or is_jpeg(magic_number) or is_tiff(magic_number)


def load_dataset(filename):
    if os.path.splitext(filename)[1].lower() == '.ascii':
        return pandas.read_table(filename, sep=',')
    else:
        return pandas.read_table(filename, skiprows=0, header=1, sep=' ')


def prepare_dataset(data):
    lookup = tuple(data.columns)

    mapping_table = {
        # new!
        (
            '#amplitude(photoelectrons),',
            'x0(pixels),', 'y0(pixels),',
            'simga_x(pixels),',
            'sigma_y(pixels),',
            'background(photoelectrons),',
            'z_position(pixels),',
            'quality,',
            'CNR,',
            'localization_precision_x(pixels),',
            'localization_precision_y(pixels),',
            'localization_precision_z(pixels),',
            'correlation_coefficient,',
            'frame'
        ): {
            '#amplitude(photoelectrons),': 'amp',
            'x0(pixels),': 'x',
            'y0(pixels),': 'y',
            'simga_x(pixels),': 'sigma_x',
            'sigma_y(pixels),': 'sigma_y',
            'background(photoelectrons),': 'back',
            'z_position(pixels),': 'z',
            'quality,': 'quality',
            'CNR,': 'cnr',
            'localization_precision_x(pixels),': 'locprec_x',
            'localization_precision_y(pixels),': 'locprec_y',
            'localization_precision_z(pixels),': 'locprec_z',
            'correlation_coefficient,': 'corr_coeff',
            'frame': 'frame'
        },
        # old
        (
            '#amplitude(photonelectrons),',
            'x0(pixels),',
            'y0(pixels),',
            'simga_x(pixels),',
            'sigma_y(pixels),',
            'backgroud(photonelectrons),',
            'z_position(pixels),',
            'quality,',
            'CNR,',
            'localiztion_precision_x(pixels),',
            'localiztion_precision_y(pixels),',
            'localiztion_precision_z(pixels),',
            'correlation_coefficient,',
            'frame'
        ): {
            '#amplitude(photonelectrons),': 'amp',
            'x0(pixels),': 'x',
            'y0(pixels),': 'y',
            'simga_x(pixels),': 'sigma_x',
            'sigma_y(pixels),': 'sigma_y',
            'backgroud(photonelectrons),': 'back',
            'z_position(pixels),': 'z',
            'quality,': 'quality',
            'CNR,': 'cnr',
            'localiztion_precision_x(pixels),': 'locprec_x',
            'localiztion_precision_y(pixels),': 'locprec_y',
            'localiztion_precision_z(pixels),': 'locprec_z',
            'correlation_coefficient,': 'corr_coeff',
            'frame': 'frame'
        },
        # ## EXPERIMENTAL SUPPORT FOR TOTALLY DIFFERENT FORMATS
        (
            '#stackID(UINT32)',
            ' frameID(UINT32)',
            ' eventID(UINT32)',
            ' x0(float)',
            ' y0(float)',
            ' photon_count(float)',
            ' z0(float)',
            ' type(UINT32)',
            ' n_raw(UINT32) '
        ): {
            '#stackID(UINT32)': 'stack',
            ' frameID(UINT32)': 'frame',
            ' eventID(UINT32)': 'event',
            ' x0(float)': 'x',
            ' y0(float)': 'y',
            ' photon_count(float)': 'amp',
            ' z0(float)': 'z',
            ' type(UINT32)': 'type_',
            ' n_raw(UINT32) ': 'raw'
        }
    }

    expected = ['amp', 'x', 'y', 'sigma_x', 'sigma_y', 'back', 'z', 'quality', 'cnr', 'locprec_x', 'locprec_y',
                'locprec_z', 'corr_coeff', 'frame']

    try:
        mapping = mapping_table[lookup]
    except IndexError:
        print("The super resolution tabular format is stored in an unsupported way.")
        raise
    data.rename(columns=mapping, inplace=True)

    ###
    for expected_column in expected:
        if expected_column not in data.columns:
            data[expected_column] = numpy.zeros(len(data))
    ###

    if not (data.locprec_x == data.locprec_y).all():
        print("Warning! Localization precision is not x/y identical!")
        data['locprec'] = (data.locprec_x + data.locprec_y) / 2.0
    else:
        data['locprec'] = data.locprec_x

    if not (data.sigma_x == data.sigma_y).all():
        print("Warning! Sigma is not x/y identical!")
        data['sigma'] = (data.sigma_x + data.sigma_x) / 2.0
    else:
        data['sigma'] = data.sigma_x

    return data
