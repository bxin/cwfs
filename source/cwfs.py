#!/usr/bin/env python
##
# @package cwfs
# @file cwfs.py
# @brief main script to run cwfs
##
# @authors: Bo Xin & Chuck Claver
# @       Large Synoptic Survey Telescope

import os
import argparse

from cwfsInstru import cwfsInstru
from cwfsAlgo import cwfsAlgo
from cwfsImage import cwfsImage
from cwfsTools import outParam, outZer4Up

# main function


def main():

    parser = argparse.ArgumentParser(
        description='-----This is cwfs (Curvature Wavefront Sensing) code----')

    parser.add_argument('intra', help='intra focal image file name (no path)')
    parser.add_argument('extra', help='extra focal image file name (no path)')

    parser.add_argument('-dir', dest='imgDir',
                        help='relative or absolute path for input images')
    parser.add_argument('-ixy', dest='intra_xy', nargs=2,
                        type=float, default=[0, 0],
                        help='intra focal field (x,y) in deg, default=[0 0]')
    parser.add_argument('-exy', dest='extra_xy', nargs=2,
                        type=float, default=[0, 0],
                        help='extra focal field (x,y) in deg, default=[0 0]')
    parser.add_argument('-i', dest='instruFile', default='lsst',
                        help='instrument parameter file, default=lsst,\
                         ".param" is appended automatically \
                        default path is data/lsst/')
    parser.add_argument('-a', dest='algoFile', default='fft',
                        help='algorithm parameter file, default=fft,\
                         ".algo" is appended automatically\
                        default path is data/algo/')
    parser.add_argument('-m', dest='model',
                        choices=('paraxial', 'onAxis', 'offAxis'),
                        default='paraxial',
                        help='Optical model to be used, default=paraxial')
    parser.add_argument('-op', dest='outputParam', default='',
                        help='file name for dumping all parameters, \
                        default=no output')
    parser.add_argument('-oz', dest='outputZerFile', default='',
                        help='file name for output Zernikes (in unit of nm), \
                        default=no output')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s 1.0')
    parser.add_argument('-d', dest='debugLevel', type=int,
                        default=0, choices=(-1, 0, 1, 2, 3),
                        help='debug level, -1=quiet, 0=Zernikes, \
                        1=operator, 2=expert, 3=everything, default=0')
    args = parser.parse_args()

    if args.debugLevel >= 1:
        print(args)

    # load intra and extra focal images
    intraFile = os.path.join(args.imgDir, args.intra)
    extraFile = os.path.join(args.imgDir, args.extra)

    I1 = cwfsImage(intraFile, args.intra_xy, 'intra')
    I2 = cwfsImage(extraFile, args.extra_xy, 'extra')

    # load instrument and algorithm parameters
    inst = cwfsInstru(args.instruFile, I1.sizeinPix)
    algo = cwfsAlgo(args.algoFile, inst, args.debugLevel)

    # run it
    algo.runIt(inst, I1, I2, args.model)

    # output Zernikes 4 and up
    if not(args.outputZerFile == '') or args.debugLevel >= 0:
        outZer4Up(algo.zer4UpNm,'nm', args.outputZerFile)

    # output parameters
    if not(args.outputParam == '') or args.debugLevel >= 1:
        outParam(args.outputParam, algo, inst, I1, I2, args.model)

if __name__ == "__main__":
    main()
