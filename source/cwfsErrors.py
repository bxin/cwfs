#!/usr/bin/env python
##
# @package cwfs
# @file cwfsImage.py
##
# @authors: Bo Xin & Chuck Claver
# @       Large Synoptic Survey Telescope


class nonSquareImageError(Exception):

    def __init__(self):
        pass


class imageDiffSizeError(Exception):

    def __init__(self):
        pass


class unknownUnitError(Exception):

    def __init__(self):
        pass


class oddNumPixError(Exception):

    def __init__(self):
        pass
