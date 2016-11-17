#!/usr/bin/env python
##
# @package cwfs
# @file cwfsPlots.py
##
# @authors: Bo Xin & Chuck Claver
# @       Large Synoptic Survey Telescope

import sys

import numpy as np
import matplotlib.pyplot as plt

from . import errors


def plotImage(image, title=None, mask=None, show=True):
    if mask is not None:
        image = np.where(mask == 0, np.nan, image)

    plt.imshow(image, origin='lower')
    plt.colorbar()
    if title:
        plt.title(title)

    if show:
        plt.show()


def plotZer(z, unit):
    try:
        if unit == 'm':
            z = z * 1e-9
        elif unit == 'nm':
            pass
        elif unit == 'um':
            z = z * 1e-3
        else:
            raise Exception(errors.unknownUnitError)
    except errors.unknownUnitError:
        print('Unknown unit: %s' % unit)
        print('Known options are: m, nm, um')
        sys.exit()

    x = range(4, len(z) + 4)
    plt.plot(x, z,  # label='',
             marker='o', color='r', markersize=10)
    plt.xlabel('Zernike Index')
    plt.ylabel('Zernike coefficient (%s)' % unit)
    plt.grid()
    plt.show()
