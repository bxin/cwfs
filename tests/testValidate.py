import unittest
import lsst.utils.tests

import os
import pytest
import sys
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError as e:
    print("Initialising testValidate.py:", e)
    plt = None

from lsst.cwfs.instrument import Instrument
from lsst.cwfs.algorithm import Algorithm
from lsst.cwfs.image import Image, readFile

class MatlabValidationTestCase(lsst.utils.tests.TestCase):
    """Demo test case."""

    @classmethod
    def setUpClass(cls):
        """ setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        cls.rootdir = os.path.dirname(__file__)

        cls.myinst = 'lsst'
        cls.validationDir = os.path.join(str(cls.rootdir), 'validation')

        cls.tests = [
            ('testImages/F1.23_1mm_v61', 'z7_0.25_%s.txt', (0, 0),          ('fft',),       'paraxial'),
            ('testImages/LSST_C_SN26',   'z7_0.25_%s.txt', (0, 0),          ('fft', 'exp'), 'onAxis'),
            ('testImages/LSST_NE_SN25',  'z11_0.25_%s.txt', (1.185, 1.185), ('fft', 'exp'), 'offAxis'),
            ]
        # filenames with matlab results and tolerance on absolute discrepancy (in nm)
        #
        # N.b. these tolerances are set at 10nm because centering algorithm has changed.
        #      difference in the wavefront on the ~10nm is well below noise level.
        #
        cls.matlabZFile_Tol = [('F1.23_1mm_v61_z7_0.25_fft.txt', 10),#
                               ('LSST_C_SN26_z7_0.25_fft.txt',   10),
                               ('LSST_C_SN26_z7_0.25_exp.txt',   10),
                               ('LSST_NE_SN25_z11_0.25_fft.txt', 10),
                               ('LSST_NE_SN25_z11_0.25_exp.txt', 10),
        ]
        #
        # Check that we have the right number of matlab files.  Not really a unit test, just consistency
        #
        cls.nTest = 0
        for inDir, filenameFmt, fldxy, algorithms, model in cls.tests:
            cls.nTest += len(algorithms)
        assert cls.nTest == len(cls.matlabZFile_Tol)

        cls.Zernike0 = 4                # first Zernike to fit
        znmax = 22                      # last Zernike to fit
        cls.x = range(cls.Zernike0, znmax + 1)

    def testMatlab(self):
        global doPlot
        if doPlot:
            fig = plt.figure(figsize=(10, 10))

        j = 0                           # counter for matlab outputs, self.matlabZFile_Tol
        for imgDir, filenameFmt, fldxy, algorithms, model in self.tests:
            imgDir = os.path.join(str(self.rootdir), imgDir)
            intraFile = os.path.join(imgDir, filenameFmt % "intra")
            I1 = Image(readFile(intraFile), fldxy, Image.INTRA)

            extraFile = os.path.join(imgDir, filenameFmt % "extra")
            I2 = Image(readFile(extraFile), fldxy, Image.EXTRA)

            inst = Instrument(self.myinst, I1.sizeinPix)

            for algorithm in algorithms:
                matlabZFile, tol = self.matlabZFile_Tol[j]; j += 1

                algo = Algorithm(algorithm, inst, 1)
                algo.runIt(inst, I1, I2, model)

                zer = algo.zer4UpNm
                matZ = np.loadtxt(os.path.join(self.validationDir, matlabZFile))

                aerr = np.abs(matZ - zer)
                print("%-31s max(abs(err)) = %8.3g median(abs(err)) = %8.3g [Z_%d], tol=%.0f nm" %
                      (matlabZFile, np.max(aerr), np.median(aerr), self.Zernike0 + np.argmax(aerr), tol))

                if doPlot:
                    ax = plt.subplot(self.nTest, 1, j)
                    plt.plot(self.x, matZ, label='Matlab', marker='o', color='r', markersize=10)
                    plt.plot(self.x, zer, label='Python',  marker='.', color='b', markersize=10)
                    plt.axvline(self.Zernike0 + np.argmax(aerr), ls=':', color='black')
                    plt.legend(loc="best", shadow=True, title=matlabZFile, fancybox=True)
                    ax.get_legend().get_title().set_color("red")
                    plt.xlim(self.x[0] - 0.5, self.x[-1] + 0.5)

                assert np.max(aerr) < tol

        if doPlot:
            plt.show()

if False:                               # used for C++, and triggers a file descriptor leak from matplotlib
    class MemoryTester(lsst.utils.tests.MemoryTestCase):
        pass

def setup_module(module):
    lsst.utils.tests.init()

try:
    doPlot                              # should I plot things?
except NameError:
    doPlot = False

if __name__ == "__main__":
    doPlot = sys.stdout.isatty()        # enable matplotlib when run from a terminal

    lsst.utils.tests.init()
    unittest.main()
