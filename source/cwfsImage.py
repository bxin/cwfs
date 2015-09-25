#!/usr/bin/env python
##
# @package cwfs
# @file cwfsImage.py
##
# @authors: Bo Xin & Chuck Claver
# @       Large Synoptic Survey Telescope

import sys

import numpy as np
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
from astropy.io import fits

from cwfsTools import ZernikeAnnularGrad
from cwfsTools import ZernikeGrad
from cwfsTools import ZernikeAnnularJacobian
from cwfsTools import ZernikeJacobian
from cwfsTools import padArray
from cwfsTools import extractArray

from cwfsErrors import nonSquareImageError
from cwfsErrors import oddNumPixError


class cwfsImage(object):

    def __init__(self, filename, fieldXY, type):

        fieldX = fieldXY[0]
        fieldY = fieldXY[1]

        if (filename.endswith(".txt")):
            self.image = np.loadtxt(filename)
            # this assumes this txt file is in the format
            # I[0,0]   I[0,1]
            # I[1,0]   I[1,1]
            self.image = self.image[::-1, :]
        else:
            IHDU = fits.open(filename)
            self.image = IHDU[0].data
            IHDU.close()
        self.fieldX = fieldX
        self.fieldY = fieldY
        self.fldr = np.sqrt(self.fieldX**2 + self.fieldY**2)
        self.type = type
        self.sizeinPix = self.image.shape[0]
        self.filename = filename
        try:
            if self.image.shape[0] != self.image.shape[1]:
                raise(nonSquareImageError)
            if self.image.shape[0] % 2 == 1:
                raise(oddNumPixError)
        except nonSquareImageError:
            print('%s image filename = %s ' % (type, filename))
            print('%s image size = (%d, %d)' % (
                type, self.image.shape[0], self.image.shape[1]))
            print('Error: Only square image stamps are accepted.')
            sys.exit()
        except oddNumPixError:
            print('%s image filename = %s ' % (type, filename))
            print('%s image size = (%d, %d)' % (
                type, self.image.shape[0], self.image.shape[1]))
            print('Error: number of pixels cannot be odd numbers')
            sys.exit()

    # if we pass inst.maskParam, a try: catch: is needed in cwfs.py
    def makeMaskList(self, inst):
        if (self.fieldX == 0 and self.fieldY == 0):
            if inst.obscuration == 0:
                self.masklist = np.array([0, 0, 1, 1])
            else:
                self.masklist = np.array([[0, 0, 1, 1],
                                          [0, 0, inst.obscuration, 0]])
        else:
            self.maskCa, self.maskRa, self.maskCb, self.maskRb = \
                interpMaskParam(self.fieldX, self.fieldY, inst.maskParam)
            cax, cay, cbx, cby = rotateMaskParam(  # only change the center
                self.maskCa, self.maskCb, self.fieldX, self.fieldY)
            self.masklist = np.array(
                [[0, 0, 1, 1], [0, 0, inst.obscuration, 0],
                 [cax, cay, self.maskRa, 1], [cbx, cby, self.maskRb, 0]])

    def makeMask(self, inst, boundaryT, maskScalingFactor):

        self.pMask = np.ones(inst.sensorSamples, dtype=int)
        self.cMask = self.pMask

        rMask = inst.apertureDiameter / (2 * inst.focalLength / inst.offset)\
            * maskScalingFactor

        for ii in range(self.masklist.shape[0]):

            r = np.sqrt((inst.xSensor - self.masklist[ii, 0])**2 +
                        (inst.ySensor - self.masklist[ii, 1])**2)

            # Initialize both mask elements to the opposite of the pass/block
            # boolean
            pMaskii = (1 - self.masklist[ii, 3]) * np.ones(
                (inst.sensorSamples, inst.sensorSamples), dtype=int)
            cMaskii = (1 - self.masklist[ii, 3]) * np.ones(
                (inst.sensorSamples, inst.sensorSamples), dtype=int)

            # Find the indices that correspond to the mask element, set them to
            # the pass/block boolean
            idx = r <= self.masklist[ii, 2]
            if (self.masklist[ii, 3] >= 1):
                # make a mask >r so that we can keep a larger area of S
                aidx = np.nonzero(r <= self.masklist[ii, 2] *
                                  (1 + boundaryT * inst.pixelSize / rMask))
            else:
                aidx = np.nonzero(r <= self.masklist[ii, 2] *
                                  (1 - boundaryT * inst.pixelSize / rMask))
            pMaskii[idx] = self.masklist[ii, 3]
            cMaskii[aidx] = self.masklist[ii, 3]

            # Multiplicatively add the current mask elements to the model masks
            # padded mask - for use at the offset planes
            self.pMask = self.pMask * pMaskii
            # non-padded mask corresponding to aperture
            self.cMask = self.cMask * cMaskii

    def getOffAxisCorr(self, order):
        self.offAxis_coeff = np.zeros((4, (order + 1) * (order + 2) / 2))
        self.offAxis_coeff[0, :] = getOffAxisCorr_single(
            'data/lsst/offAxis_cxin_poly%d.txt' % (order), self.fldr)
        self.offAxis_coeff[1, :] = getOffAxisCorr_single(
            'data/lsst/offAxis_cyin_poly%d.txt' % (order), self.fldr)
        self.offAxis_coeff[2, :] = getOffAxisCorr_single(
            'data/lsst/offAxis_cxex_poly%d.txt' % (order), self.fldr)
        self.offAxis_coeff[3, :] = getOffAxisCorr_single(
            'data/lsst/offAxis_cyex_poly%d.txt' % (order), self.fldr)

    def upResolution(self, oversample, lm, ln):

        # lm and ln are dimensions after upResolution
        sm = lm / oversample
        sn = ln / oversample

        newI = np.zeros((lm, ln))
        for i in range(sm):
            for j in range(sn):
                for k in range(oversample):
                    for l in range(oversample):
                        newI[i * oversample + k, j * oversample + l] = \
                            self.image[i, j] / oversample / oversample
        self.image = newI

    def downResolution(self, oversample, sm, sn):

        # sm and sn are dimensions after downResolution

        newI = np.zeros((sm, sn))
        for i in range(sm):
            for j in range(sn):
                subI = self.image[i * oversample:(i + 1) * oversample,
                                  j * oversample:(j + 1) * oversample]
                idx = (~np.isnan(subI)).nonzero()
                if np.sum(np.sum(subI)) > 0:
                    newI[i, j] = np.sum(subI(idx))
                    newI[i, j] = newI[i, j] / np.sum(np.sum(idx))
                else:
                    newI[i, j] = np.nan

        self.image = newI

    def imageCoCenter(self, inst, algo):

        x1, y1, tmp = getCenterAndR_ef(self.image)
        if algo.debugLevel >= 3:
            print('imageCoCenter: (x1,y1)=(%8.2f,%8.2f)\n' % (x1, y1))

        stampCenterx1 = inst.sensorSamples / 2. + 0.5
        stampCentery1 = inst.sensorSamples / 2. + 0.5
        radialShift = 3.5 * algo.upReso * \
            (inst.offset / 1e-3) * (10e-6 / inst.pixelSize)

        radialShift = radialShift * self.fldr / 1.75
        if (self.fldr > 1.75):
            radialShift = 0

        if self.fldr != 0:
            I1c = self.fieldX / self.fldr
            I1s = self.fieldY / self.fldr
        else:
            I1c = 0
            I1s = 0

        stampCenterx1 = stampCenterx1 + radialShift * I1c
        stampCentery1 = stampCentery1 + radialShift * I1s

        self.image = np.roll(self.image, int(
            np.round(stampCentery1 - y1)), axis=0)
        self.image = np.roll(self.image, int(
            np.round(stampCenterx1 - x1)), axis=1)

    def compensate(self, inst, algo, zcCol, oversample, model):

        if ((zcCol.ndim == 1) and (len(zcCol) != algo.numTerms)):
            raise Exception(
                'input:size', 'zcCol in compensate needs to be a %d \
                row column vector\n' % algo.numTerms)

        sm, sn = self.image.shape

        projSamples = sm * oversample

        # Let us create a look-up table for x -> xp first.
        luty, lutx = np.mgrid[
            -(projSamples / 2 - 0.5):(projSamples / 2 + 0.5),
            -(projSamples / 2 - 0.5):(projSamples / 2 + 0.5)]
        lutx = lutx / (projSamples / 2 / inst.sensorFactor)
        luty = luty / (projSamples / 2 / inst.sensorFactor)

        # set up the mapping
        lutxp, lutyp, J = aperture2image(
            self, inst, algo, zcCol, lutx, luty, projSamples, model)
        #    print "J",J.shape

        show_lutxyp = showProjection(
            lutxp, lutyp, inst.sensorFactor, projSamples, 0)

        realcx, realcy, tmp = getCenterAndR_ef(self.image)
        show_lutxyp = padArray(show_lutxyp, projSamples + 20)

        struct0 = ndimage.generate_binary_structure(2, 1)
        struct = ndimage.iterate_structure(struct0, 4)
        struct = ndimage.morphology.binary_dilation(struct, structure=struct0)
        struct = ndimage.morphology.binary_dilation(
            struct, structure=struct0).astype(int)
        show_lutxyp = ndimage.morphology.binary_dilation(
            show_lutxyp, structure=struct)
        show_lutxyp = ndimage.morphology.binary_erosion(
            show_lutxyp, structure=struct)
        show_lutxyp = extractArray(show_lutxyp, projSamples)

        projcx, projcy, tmp = getCenterAndR_ef(show_lutxyp.astype(float))
        projcx = projcx / (oversample)
        projcy = projcy / (oversample)

        # +(-) means we need to move image to the right (left)
        shiftx = (projcx - realcx)
        # +(-) means we need to move image upward (downward)
        shifty = (projcy - realcy)
        self.image = np.roll(self.image, int(np.round(shifty)), axis=0)
        self.image = np.roll(self.image, int(np.round(shiftx)), axis=1)

        # let's construct the interpolant,
        # to get the intensity on (x',p') plane
        # that corresponds to the grid points on (x,y)
        yp, xp = np.mgrid[-(sm / 2 - 0.5):(sm / 2 + 0.5), -
                          (sm / 2 - 0.5):(sm / 2 + 0.5)]
        xp = xp / (sm / 2 / inst.sensorFactor)
        yp = yp / (sm / 2 / inst.sensorFactor)

        # xp = reshape(xp,sm^2,1);
        # yp = reshape(yp,sm^2,1);
        # self.image = reshape(self.image,sm^2,1);
        #
        # FIp = TriScatteredInterp(xp,yp,self.image,'nearest');
        # lutIp = FIp(lutxp, lutyp);

        lutxp[np.isnan(lutxp)] = 0
        lutyp[np.isnan(lutyp)] = 0

        #    lutIp=interp2(xp,yp,self.image,lutxp,lutyp)
        #    print xp.shape, yp.shape, self.image.shape
        #    print lutxp.ravel()
        #    print xp[:,0],yp[0,:]
        ip = interpolate.RectBivariateSpline(
            yp[:, 0], xp[0, :], self.image, kx=1, ky=1)

        #    ip = interpolate.interp2d(xp, yp, self.image)
        #    ip = interpolate.interp2d(xp, yp, self.image)
        #    print lutxp.shape, lutyp.shape
        #    lutIp = ip(0.5, -0.5)
        #    print lutIp, 'lutIp1'
        #    lutIp = ip([-0.1],[-0.1])
        #    print lutIp, 'lutIp2'
        #    lutIp = ip(np.array(0.5,-0.1), np.array(-0.5, -0.1))
        #    print lutIp, 'lutIp12',lutxp.ravel()[0:10]
        lutIp = np.zeros(lutxp.shape[0] * lutxp.shape[1])
        for i, (xx, yy) in enumerate(zip(lutxp.ravel(), lutyp.ravel())):
            lutIp[i] = ip(yy, xx)
        lutIp = lutIp.reshape(lutxp.shape)

        self.image = lutIp * J

        if (self.type == 'extra'):
            self.image = np.rot90(self.image, k=2)

        # if we want the compensator to drive down tip-tilt
        # self.image = offsetImg(-shiftx, -shifty, self.image);
        # self.image=circshift(self.image,[round(-shifty) round(-shiftx)]);

        self.image[np.isnan(self.image)] = 0
        # self.image < 0 will not be physical, remove that region
        # x(self.image<0) = NaN;
        self.caustic = 0
        if (self.image.min() < 0):
            print(
                'WARNING: negative scale parameter, \
            image is within caustic, zcCol (in um)=\n')

        #    for i in range(len(zcCol)):
        #        print zcCol[i]
        #        print('%5.2f '%(zcCol[i]*1.e6))
        #    print('\n');
            self.caustic = 1

        self.image[self.image < 0] = 0
        if (oversample > 1):
            self.downResolution(self, oversample, sm, sn)


def getOffAxisCorr_single(confFile, fldr):
    c = np.loadtxt(confFile)

    ruler = np.sqrt(c[:, 0]**2 + c[:, 1]**2)
#    print ruler, fldr, (ruler >= fldr).argmax(), (ruler >= fldr).argmin()
    step = ruler[1] - ruler[0]

    p2 = (ruler >= fldr)
#    print "FINE",p2, p2.shape
    if (np.count_nonzero(p2) == 0):  # fldr is too large to be in range
        p2 = c.shape()[0]
        p1 = p2
        w1 = 1
        w2 = 0
    elif (p2[0]):  # fldr is too small to be in range
        p2 = 0  # this is going to be used as index
        p1 = 0  # this is going to be used as index
        w1 = 1
        w2 = 0
    else:
        p1 = p2.argmax() - 1
        p2 = p2.argmax()
        w1 = (ruler[p2] - fldr) / step
        w2 = (fldr - ruler[p1]) / step

#    print c[p1,2:], c[p2,2:], w1,p1, w2, p2
    corr_coeff = np.dot(w1, c[p1, 2:]) + np.dot(w2, c[p2, 2:])

    return corr_coeff


def interpMaskParam(fieldX, fieldY, maskParam):
    fldr = np.sqrt(fieldX**2 + fieldY**2)

    c = np.loadtxt(maskParam)
    ruler = np.sqrt(2 * c[:, 0]**2)
    step = ruler[1] - ruler[0]

    p2 = (ruler >= fldr)
    if (np.count_nonzero(p2) == 0):  # fldr is too large to be in range
        p2 = c.shape()[0]
        p1 = p2
        w1 = 1
        w2 = 0
    elif (p2[0]):  # fldr is too small to be in range
        p2 = 0  # this is going to be used as index
        p1 = 0  # this is going to be used as index
        w1 = 1
        w2 = 0
    else:
        p1 = p2.argmax() - 1
        p2 = p2.argmax()
        w1 = (ruler[p2] - fldr) / step
        w2 = (fldr - ruler[p1]) / step

    param = np.dot(w1, c[p1, 1:]) + np.dot(w2, c[p2, 1:])
    ca = param[0]
    ra = param[1]
    cb = param[2]
    rb = param[3]

    return ca, ra, cb, rb


def rotateMaskParam(ca, cb, fieldX, fieldY):

    fldr = np.sqrt(fieldX**2 + fieldY**2)
    c = fieldX / fldr
    s = fieldY / fldr

    cax = c * ca
    cay = s * ca
    cbx = c * cb
    cby = s * cb

    return cax, cay, cbx, cby


def getCenterAndR_ef(oriArray, readRand=1):
    # centering finding code based northcott_ef_bundle/ef/ef/efimageFunc.cc
    # this is the modified version of getCenterAndR_ef.m 6/25/14

    stepsize = 20
    nwalk = 400
    slide = 220

    histogram_len = 256

    array = oriArray.copy()
    # deblending can make over-subtraction in some area, a lower tail could
    # form; temperary solution.
    array[array < 0] = 0
    m, n = array.shape

    pmin = array.min()
    pmax = array.max()
    if (pmin == pmax):
        print('image has min=max=%f', pmin)
    array1d = np.reshape(array, m * n, 1)

    phist, cen = np.histogram(array1d, bins=histogram_len)
    startidx = range(25, 175 + 25, 25)
    # startidx=fliplr(startidx)
    foundvalley = 0

    # validating code against Matlab:
    # to avoid differences in random number generator between NumPy and
    # Matlab, read in these random numbers generated from Matlab
    if readRand:
        iRand = 0
        myRand = np.loadtxt('data/algo/testRand.txt')
        myRand = np.tile(np.reshape(myRand, (1000, 1)), (10, 1))

    for istartPoint in range(len(startidx)):
        minind = startidx[istartPoint]
        if ((minind <= 0) or (max(phist[minind - 1:]) == 0)):
            continue
        minval = phist[minind - 1]

        # try nwalk=2000 times and see if it rolls to > slide.
        # if it does roll off, let's change starting point (minind) to 25 less
        # (minind starts at 175, then goes to 150, then 125
        for i in range(nwalk + 1):
            if (minind <= slide):
                while (minval == 0):
                    minind = minind + 1
                    minval = phist[minind - 1]
                if readRand:
                    ind = np.round(-stepsize + 2 * stepsize * myRand[iRand, 0])
                    iRand += 1
                    thermal = 1 + 0.5 * myRand[iRand, 0] * \
                        np.exp(-(1.0 * i / (nwalk * 0.3)))
                    iRand += 1
                else:
                    ind = np.round(stepsize * (2 * np.random.rand() - 1))
                    thermal = 1  # +0.05*np.random.rand()
                    # *np.exp(-(1.0*i/(nwalk*0.3)))

                if ((minind + ind < 1) or (minind + ind > (histogram_len))):
                    continue
                if (phist[minind + ind - 1] < (minval * thermal)):
                    minval = phist[minind + ind - 1]
                    minind = minind + ind
            else:
                break
        if (minind <= slide):
            foundvalley += 1
        if foundvalley >= 1:
            break

    # Never understood why we do this, but had it before 5/27/14, since EF C++
    # code had this. Now this appears to cause minind> histgram_length, we
    # comment it out and see how the code performs.
    # minind = avind/(steps-steps/2);

    # fprintf('minind=%d\n',minind);
    if (not foundvalley):  # because of noise, there is only peak
        minind = histogram_len / 2
    pval = pmin + (pmax - pmin) / histogram_len * float(minind - 0.5)
    tmp = array.copy()
    tmp[array > max(0, pval - 1e-8)] = 1
    tmp[array < pval] = 0
    # because tmp is a binary mask with only 1 and 0
    realR = np.sqrt(np.sum(tmp) / 3.1415926)

    jarray, iarray = np.mgrid[1:n + 1, 1:m + 1]
    realcx = np.sum(iarray * tmp) / np.sum(tmp)
    realcy = np.sum(jarray * tmp) / np.sum(tmp)

    # print realcx, realcy, realR
    return realcx, realcy, realR


def createPupilGrid(lutx, luty, onepixel, ca, cb, ra, rb, fieldX, fieldY=None):
    """Create the pupil grid"""

    if (fieldY is None):
        fldr = fieldX
        fieldX = fldr / 1.4142
        fieldY = fieldX

    cax, cay, cbx, cby = rotateMaskParam(ca, cb, fieldX, fieldY)

    lutr = np.sqrt((lutx - cax)**2 + (luty - cay)**2)
    tmp = lutr.copy()
    tmp[np.isnan(tmp)] = -999
    idxout = (tmp > ra + onepixel)
    lutx[idxout] = np.nan
    luty[idxout] = np.nan
    idxbound = (tmp <= ra + onepixel) & (tmp > ra) & (~np.isnan(lutx))
    lutx[idxbound] = (lutx[idxbound] - cax) / lutr[idxbound] * ra + cax
    luty[idxbound] = (luty[idxbound] - cay) / lutr[idxbound] * ra + cay

    lutr = np.sqrt((lutx - cbx)**2 + (luty - cby)**2)
    tmp = lutr.copy()
    tmp[np.isnan(tmp)] = 999
    idxout = (tmp < rb - onepixel)
    lutx[idxout] = np.nan
    luty[idxout] = np.nan
    idxbound = (tmp >= rb - onepixel) & (tmp < rb) & (~np.isnan(lutx))
    lutx[idxbound] = (lutx[idxbound] - cbx) / lutr[idxbound] * rb + cbx
    luty[idxbound] = (luty[idxbound] - cby) / lutr[idxbound] * rb + cby

    return lutx, luty


def aperture2image(Im, inst, algo, zcCol, lutx, luty, projSamples, model):

    R = inst.apertureDiameter / 2.0
    if (Im.type == 'intra'):
        myC = - inst.focalLength * \
            (inst.focalLength - inst.offset) / inst.offset / R**2
    elif (Im.type == 'extra'):
        myC = inst.focalLength * (inst.focalLength / inst.offset + 1) / R**2

    module = __import__('cwfsTools', 'poly%d_2D' % algo.offAxisPolyOrder)
    polyFunc = getattr(module, 'poly%d_2D' % algo.offAxisPolyOrder)

    module = __import__('cwfsTools', 'poly%dGrad' % algo.offAxisPolyOrder)
    polyGradFunc = getattr(module, 'poly%dGrad' % algo.offAxisPolyOrder)

    lutr = np.sqrt(lutx**2 + luty**2)
    # 1 pixel larger than projected pupil. No need to be EF-like, anything
    # outside of this will be masked off by the computational mask
    onepixel = 1 / (projSamples / 2 / inst.sensorFactor)
    # print "LUTR",lutr.shape,((lutr >1 + onepixel) |
    # (lutr<inst.obscuration-onepixel)).shape
    idxout = ((lutr > 1 + onepixel) | (lutr < inst.obscuration - onepixel))
    lutx[idxout] = np.nan
    luty[idxout] = np.nan
    # outer boundary (1 pixel wide boundary)
    idxbound = ((lutr <= 1 + onepixel) & (lutr > 1))
    lutx[idxbound] = lutx[idxbound] / lutr[idxbound]
    luty[idxbound] = luty[idxbound] / lutr[idxbound]
    idxinbd = ((lutr < inst.obscuration) & (
        lutr > inst.obscuration - onepixel))  # inner boundary
    lutx[idxinbd] = lutx[idxinbd] / lutr[idxinbd] * inst.obscuration
    luty[idxinbd] = luty[idxinbd] / lutr[idxinbd] * inst.obscuration

    if (model == 'offAxis'):
        lutx, luty = createPupilGrid(
            lutx, luty, onepixel,
            Im.maskCa, Im.maskCb, Im.maskRa, Im.maskRb,
            Im.fieldX, Im.fieldY)

    if (model == 'paraxial'):
        lutxp = lutx
        lutyp = luty
    elif (model == 'onAxis'):
        myA = np.sqrt((inst.focalLength**2 - R**2) /
                      (inst.focalLength**2 - lutr**2 * R**2))
        lutxp = algo.maskScalingFactor * myA * lutx
        lutyp = algo.maskScalingFactor * myA * luty
    elif (model == 'offAxis'):
        # nothing is hard-coded here: (1e-3) is because the
        # offAxis correction param files are based on offset=1.0mm
        if (Im.type == 'intra'):
            cx = (Im.offAxis_coeff[0, :] - Im.offAxis_coeff[2, :]) * \
                (1e-3 + inst.offset) / 2e-3 + Im.offAxis_coeff[2, :]
            cy = (Im.offAxis_coeff[1, :] - Im.offAxis_coeff[3, :]) * \
                (1e-3 + inst.offset) / 2e-3 + Im.offAxis_coeff[3, :]
        elif (Im.type == 'extra'):
            cx = (Im.offAxis_coeff[0, :] - Im.offAxis_coeff[2, :]) * \
                (1e-3 - inst.offset) / 2e-3 + Im.offAxis_coeff[2, :]
            cy = (Im.offAxis_coeff[1, :] - Im.offAxis_coeff[3, :]) * \
                (1e-3 - inst.offset) / 2e-3 + Im.offAxis_coeff[3, :]
            cx = -cx  # this will be inverted back by typesign later on.
            # we do the inversion here to make the (x,y)->(x',y') equations has
            # the same form as the paraxial case.
            cy = -cy

        costheta = (Im.fieldX + Im.fieldY) / Im.fldr / 1.4142
        if (costheta > 1):
            costheta = 1
        elif (costheta < -1):
            costheta = -1

        sintheta = np.sqrt(1 - costheta**2)
        if (Im.fieldY < Im.fieldX):
            sintheta = -sintheta

        # first rotate back to reference orientation
        lutx0 = lutx * costheta + luty * sintheta
        luty0 = -lutx * sintheta + luty * costheta
        # use mapping at reference orientation
        lutxp0 = polyFunc(cx, lutx0, y=luty0)
        lutyp0 = polyFunc(cy, lutx0, y=luty0)
        lutxp = lutxp0 * costheta - lutyp0 * sintheta  # rotate back
        lutyp = lutxp0 * sintheta + lutyp0 * costheta
        # Zemax data are in mm, therefore 1000
        reduced_coordi_factor = 1 / \
            (inst.sensorSamples / 2 * inst.pixelSize / inst.sensorFactor) \
            / 1000
        # reduced coordinates, so that this can be added with the dW/dz
        lutxp = lutxp * reduced_coordi_factor
        lutyp = lutyp * reduced_coordi_factor
    else:
        print('wrong model number in compensate\n')
        return

    if (zcCol.ndim == 1):
        if (algo.zobsR > 0):
            lutxp = lutxp + myC * \
                ZernikeAnnularGrad(zcCol, lutx, luty, algo.zobsR, 'dx')
            lutyp = lutyp + myC * \
                ZernikeAnnularGrad(zcCol, lutx, luty, algo.zobsR, 'dy')
        else:
            lutxp = lutxp + myC * ZernikeGrad(zcCol, lutx, luty, 'dx')
            lutyp = lutyp + myC * ZernikeGrad(zcCol, lutx, luty, 'dy')
    else:
        FX, FY = np.gradient(zcCol,
                             inst.sensorFactor / (inst.sensorSamples / 2))
        lutxp = lutxp + myC * FX
        lutyp = lutyp + myC * FY

    if (Im.type == 'extra'):
        lutxp = - lutxp
        lutyp = - lutyp

    # Below for calculation of the Jacobian

    if (zcCol.ndim == 1):
        if (model == 'paraxial'):
            if (algo.zobsR > 0):
                J = (1 +
                     myC * ZernikeAnnularJacobian(
                         zcCol, lutx, luty, algo.zobsR, '1st') +
                     myC**2 * ZernikeAnnularJacobian(
                         zcCol, lutx, luty, algo.zobsR, '2nd'))
            else:
                J = (1 + myC * ZernikeJacobian(zcCol, lutx, luty, '1st') +
                     myC**2 * ZernikeJacobian(zcCol, lutx, luty, '2nd'))

        elif (model == 'onAxis'):
            xpox = algo.maskScalingFactor * myA * (
                1 +
                lutx**2 * R**2. / (inst.focalLength**2 - R**2 * lutr**2)) + \
                myC * ZernikeAnnularGrad(zcCol, lutx, luty, algo.zobsR, 'dx2')
            ypoy = algo.maskScalingFactor * myA * (
                1 +
                luty**2 * R**2. / (inst.focalLength**2 - R**2 * lutr**2)) + \
                myC * ZernikeAnnularGrad(zcCol, lutx, luty, algo.zobsR, 'dy2')
            xpoy = algo.maskScalingFactor * myA * \
                lutx * luty * R**2 / (inst.focalLength**2 - R**2 * lutr**2) + \
                myC * ZernikeAnnularGrad(zcCol, lutx, luty, algo.zobsR, 'dxy')
            ypox = xpoy

            J = (xpox * ypoy - xpoy * ypox)
        elif (model == 'offAxis'):
            xp0ox = polyGradFunc(cx, lutx0, luty0, 'dx') * costheta - \
                polyGradFunc(cx, lutx0, luty0, 'dy') * sintheta
            yp0ox = polyGradFunc(cy, lutx0, luty0, 'dx') * costheta - \
                polyGradFunc(cy, lutx0, luty0, 'dy') * sintheta
            xp0oy = polyGradFunc(cx, lutx0, luty0, 'dx') * sintheta + \
                polyGradFunc(cx, lutx0, luty0, 'dy') * costheta
            yp0oy = polyGradFunc(cy, lutx0, luty0, 'dx') * sintheta + \
                polyGradFunc(cy, lutx0, luty0, 'dy') * costheta
            xpox = (xp0ox * costheta - yp0ox * sintheta) * \
                reduced_coordi_factor + \
                myC * ZernikeAnnularGrad(zcCol, lutx, luty, algo.zobsR, 'dx2')

            ypoy = (xp0oy * sintheta + yp0oy * costheta) * \
                reduced_coordi_factor + \
                myC * ZernikeAnnularGrad(zcCol, lutx, luty, algo.zobsR, 'dy2')

            temp = myC * \
                ZernikeAnnularGrad(zcCol, lutx, luty, algo.zobsR, 'dxy')
            # if temp==0,xpoy doesn't need to be symmetric about x=y
            xpoy = (xp0oy * costheta - yp0oy * sintheta) * \
                reduced_coordi_factor + temp
            # xpoy-flipud(rot90(ypox))==0 is true
            ypox = (xp0ox * sintheta + yp0ox * costheta) * \
                reduced_coordi_factor + temp
            J = (xpox * ypoy - xpoy * ypox)

    else:

        FXX, FXY = np.gradient(FX,
                               inst.sensorFactor / (inst.sensorSamples / 2))
        tmp, FYY = np.gradient(FY,
                               inst.sensorFactor / (inst.sensorSamples / 2))
        if (model == 'paraxial'):
            xpox = 1 + myC * FXX
            ypoy = 1 + myC * FYY
            xpoy = 1 + myC * FXY
            ypox = xpoy
        elif (model == 'onAxis'):
            xpox = algo.maskScalingFactor * myA * \
                (1 +
                 lutx**2 * R**2. / (inst.focalLength**2 - R**2 * lutr**2)) + \
                myC * FXX
            ypoy = algo.maskScalingFactor * myA * \
                (1 +
                 luty**2 * R**2 / (inst.focalLength**2 - R**2 * lutr**2)) + \
                myC * FYY
            xpoy = algo.maskScalingFactor * myA * lutx * luty * R**2. / \
                (inst.focalLength**2 - R**2 * lutr**2) + myC * FXY
            ypox = xpoy
        elif (model == 'offAxis'):
            xpox = polyGradFunc(cx, lutx, luty, 'dx') * \
                reduced_coordi_factor + myC * FXX
            ypoy = polyGradFunc(cy, lutx, luty, 'dy') * \
                reduced_coordi_factor + myC * FYY
            xpoy = polyGradFunc(cx, lutx, luty, 'dy') * \
                reduced_coordi_factor + myC * FXY
            ypox = polyGradFunc(cy, lutx, luty, 'dx') * \
                reduced_coordi_factor + myC * FXY

        J = (xpox * ypoy - xpoy * ypox)

    return lutxp, lutyp, J


def showProjection(lutxp, lutyp, sensorFactor, projSamples, raytrace):
    n1, n2 = lutxp.shape
    show_lutxyp = np.zeros((n1, n2))
    idx = (~np.isnan(lutxp)).nonzero()
#    idx = (~np.isnan(lutxp))
    for i, j in zip(idx[0], idx[1]):
        # x=0.5 is center of pixel#1
        xR = np.round((lutxp[i, j] + sensorFactor) *
                      (projSamples / sensorFactor) / 2 + 0.5)
        yR = np.round((lutyp[i, j] + sensorFactor) *
                      (projSamples / sensorFactor) / 2 + 0.5)

        if (xR > 0 and xR < n2 and yR > 0 and yR < n1):
            if raytrace:
                show_lutxyp[yR - 1, xR - 1] = show_lutxyp[yR - 1, xR - 1] + 1
            else:
                if show_lutxyp[yR - 1, xR - 1] < 1:
                    show_lutxyp[yR - 1, xR - 1] = 1
    return show_lutxyp
