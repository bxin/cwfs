#!/usr/bin/env python
##
# @package cwfs
# @file cwfsInstru.py
##
# @authors: Bo Xin & Chuck Claver
# @       Large Synoptic Survey Telescope

import os
import numpy as np


class cwfsInstru(object):

    def __init__(self, instruFile, sensorSamples):
        self.filename = os.path.join('data/lsst/', (instruFile + '.param'))
        fid = open(self.filename)
        iscomment = False
        for line in fid:
            line = line.strip()
            if (line.startswith('###')):
                iscomment = ~iscomment
            if (not(line.startswith('#')) and
                    (not iscomment) and len(line) > 0):
                if (line.startswith('Obscuration')):
                    self.obscuration = float(line.split()[-1])
                if (line.startswith('Focal_length')):
                    self.focalLength = float(line.split()[-1])
                if (line.startswith('Aperture_diameter')):
                    self.apertureDiameter = float(line.split()[-1])
                if (line.startswith('Offset')):
                    self.offset = float(line.split()[-1])
                if (line.startswith('Pixel_size')):
                    self.pixelSize = float(line.split()[-1])
                if (line.startswith('Mask_param')):
                    self.maskParam = os.path.join(
                        'data/lsst/', line.split()[-1])
                if (line.startswith('Marginal_fl')):
                    self.marginalFL = float(line.split()[-1])
        fid.close()
        self.fno = self.focalLength / self.apertureDiameter

        # the below need to be instrument parameters, b/c it is not specific
        # for I1 or I2
        self.sensorSamples = sensorSamples
        self.sensorFactor = self.sensorSamples / \
            (self.offset * self.apertureDiameter /
             self.focalLength / self.pixelSize)
        self.sensorWidth = (self.apertureDiameter *
                            self.offset / self.focalLength) * self.sensorFactor
        self.donutR = self.pixelSize * \
            (self.sensorSamples / self.sensorFactor) / 2

        self.ySensor, self.xSensor = \
            np.mgrid[
                -(self.sensorSamples / 2 - 0.5):(self.sensorSamples / 2 + 0.5),
                -(self.sensorSamples / 2 - 0.5):(self.sensorSamples / 2 + 0.5)]
        self.xSensor = self.xSensor / \
            (self.sensorSamples / 2 / self.sensorFactor)
        self.ySensor = self.ySensor / \
            (self.sensorSamples / 2 / self.sensorFactor)
        r2Sensor = self.xSensor**2 + self.ySensor**2
        idx = (r2Sensor > 1) | (r2Sensor < self.obscuration**2)
        self.xoSensor = self.xSensor.copy()  # o indicates annulus
        self.yoSensor = self.ySensor.copy()
        self.xoSensor[idx] = np.nan
        self.yoSensor[idx] = np.nan
