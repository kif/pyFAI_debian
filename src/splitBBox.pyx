#!/usr/bin/env python
# -*- coding: utf8 -*-
#
#    Project: Azimuthal integration 
#             https://forge.epn-campus.eu/projects/azimuthal
#
#    File: "$Id$"
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#


import cython
cimport numpy
import numpy

cdef extern from "math.h":
    double floor(float)nogil


ctypedef numpy.int64_t DTYPE_int64_t
ctypedef numpy.float64_t DTYPE_float64_t
ctypedef numpy.float32_t DTYPE_float32_t


@cython.cdivision(True)
cdef float  getBinNr(float x0, float pos0_min, float dpos) nogil:
    """
    calculate the bin number for any point 
    param x0: current position
    param pos0_min: position minimum
    param dpos: bin width
    """
    return (x0 - pos0_min) / dpos


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def histoBBox1d(numpy.ndarray weights not None,
                numpy.ndarray pos0 not None,
                numpy.ndarray delta_pos0 not None,
                pos1=None,
                delta_pos1=None,
                long bins=100,
                pos0Range=None,
                pos1Range=None,
                float dummy=0.0
              ):
    """
    Calculates histogram of pos0 (tth) weighted by weights
    
    Splitting is done on the pixel's bounding box like fit2D
    
    @param weights: array with intensities
    @param pos0: 1D array with pos0: tth or q_vect
    @param delta_pos0: 1D array with delta pos0: max center-corner distance
    @param pos1: 1D array with pos1: chi
    @param delta_pos1: 1D array with max pos1: max center-corner distance, unused ! 
    @param bins: number of output bins
    @param pos0Range: minimum and maximum  of the 2th range
    @param pos1Range: minimum and maximum  of the chi range
    @param dummy: value for bins without pixels 
    @return 2theta, I, weighted histogram, unweighted histogram
    """
    cdef long  size = weights.size
    assert pos0.size == size
    assert delta_pos0.size == size
    assert  bins > 1

    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] cdata = weights.ravel().astype("float64")
    cdef numpy.ndarray[DTYPE_float32_t, ndim = 1] cpos0_inf = (pos0.ravel() - delta_pos0.ravel()).astype("float32")
    cdef numpy.ndarray[DTYPE_float32_t, ndim = 1] cpos0_sup = (pos0.ravel() + delta_pos0.ravel()).astype("float32")
    cdef numpy.ndarray[DTYPE_float32_t, ndim = 1] cpos1_inf
    cdef numpy.ndarray[DTYPE_float32_t, ndim = 1] cpos1_sup

    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] outData = numpy.zeros(bins, dtype="float64")
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] outCount = numpy.zeros(bins, dtype="float64")
    cdef numpy.ndarray[DTYPE_float32_t, ndim = 1] outMerge = numpy.zeros(bins, dtype="float32")
    cdef numpy.ndarray[DTYPE_float32_t, ndim = 1] outPos = numpy.zeros(bins, dtype="float32")
    cdef float  deltaR, deltaL, deltaA
    cdef float pos0_min, pos0_max, pos0_maxin, pos1_min, pos1_max, pos1_maxin, min0, max0, fbin0_min, fbin0_max
    cdef int checkpos1 = 0

    if pos0Range is not None and len(pos0Range) > 1:
        pos0_min = min(pos0Range)
        if pos0_min < 0.0:
            pos0_min = 0.0
        pos0_maxin = max(pos0Range)
    else:
        pos0_min = cpos0_inf.min()
        pos0_maxin = cpos0_sup.max()
    pos0_max = pos0_maxin * (1 + numpy.finfo(numpy.float32).eps)

    if pos1Range is not None and len(pos1Range) > 1:
        assert pos1.size == size
        assert delta_pos1.size == size
        checkpos1 = 1
        cpos1_inf = (pos1.ravel() - delta_pos1.ravel()).astype("float32")
        cpos1_sup = (pos1.ravel() + delta_pos1.ravel()).astype("float32")
        pos1_min = min(pos1Range)
        pos1_maxin = max(pos1Range)
        pos1_max = pos1_maxin * (1 + numpy.finfo(numpy.float32).eps)

    cdef float dpos = (pos0_max - pos0_min) / (< float > (bins))
    cdef long   bin = 0
    cdef long   i, idx
    cdef long   bin0_max, bin0_min
    cdef double epsilon = 1e-10

    with nogil:
        for i in range(bins):
                outPos[i] = pos0_min + (0.5 +< float > i) * dpos

        for idx in range(size):
            data = < double > cdata[idx]
            min0 = cpos0_inf[idx]
            max0 = cpos0_sup[idx]
            if checkpos1:
                if (cpos1_inf[idx] < pos1_min) or (cpos1_sup[idx] > pos1_max):
                    continue

            fbin0_min = getBinNr(min0, pos0_min, dpos)
            fbin0_max = getBinNr(max0, pos0_min, dpos)
            bin0_min = < long > floor(fbin0_min)
            bin0_max = < long > floor(fbin0_max)

            if bin0_min == bin0_max:
                #All pixel is within a single bin
                outCount[bin0_min] += < double > 1.0
                outData[bin0_min] += < double > data

            else: #we have pixel spliting.
                deltaA = 1.0 / (fbin0_max - fbin0_min)

                deltaL = < float > (bin0_min + 1) - fbin0_min
                deltaR = fbin0_max - (< float > bin0_max)

                outCount[bin0_min] += < double > deltaA * deltaL
                outData[bin0_min] += < double > data * deltaA * deltaL

                outCount[bin0_max] += < double > deltaA * deltaR
                outData[bin0_max] += < double > data * deltaA * deltaR

                if bin0_min + 1 < bin0_max:
                    for i in range(bin0_min + 1, bin0_max):
                        outCount[i] += < double > deltaA
                        outData[i] += data * < double > deltaA

        for i in range(bins):
                if outCount[i] > epsilon:
                    outMerge[i] = < float > (outData[i] / outCount[i])
                else:
                    outMerge[i] = dummy

    return  outPos, outMerge, outData, outCount




@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def histoBBox2d(numpy.ndarray weights not None,
                numpy.ndarray pos0 not None,
                numpy.ndarray delta_pos0 not None,
                numpy.ndarray pos1 not None,
                numpy.ndarray delta_pos1 not None,
                bins=(100, 36),
                pos0Range=None,
                pos1Range=None,
                float dummy=0.0):
    """
    Calculate 2D histogram of pos0(tth),pos1(chi) weighted by weights
    
    Splitting is done on the pixel's bounding box like fit2D
    

    @param weights: array with intensities
    @param pos0: 1D array with pos0: tth or q_vect
    @param delta_pos0: 1D array with delta pos0: max center-corner distance
    @param pos1: 1D array with pos1: chi
    @param delta_pos1: 1D array with max pos1: max center-corner distance, unused ! 
    @param bins: number of output bins (tth=100, chi=36 by default)
    @param pos0Range: minimum and maximum  of the 2th range
    @param pos1Range: minimum and maximum  of the chi range
    @param dummy: value for bins without pixels 
    @return 2theta, I, weighted histogram, unweighted histogram
    @return  I, edges0, edges1, weighted histogram(2D), unweighted histogram (2D)
    """

    cdef long  bins0, bins1, i, j, idx
    cdef long  size = weights.size
    assert pos0.size == size
    assert pos1.size == size
    assert delta_pos0.size == size
    assert delta_pos1.size == size
    try:
        bins0, bins1 = tuple(bins)
    except:
        bins0 = bins1 = < long > bins
    if bins0 <= 0:
        bins0 = 1
    if bins1 <= 0:
        bins1 = 1
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] cdata = weights.ravel().astype("float64")
    cdef numpy.ndarray[DTYPE_float32_t, ndim = 1] cpos0 = pos0.ravel().astype("float32")
    cdef numpy.ndarray[DTYPE_float32_t, ndim = 1] cdelta_pos0 = delta_pos0.ravel().astype("float32")
    cdef numpy.ndarray[DTYPE_float32_t, ndim = 1] cpos0_inf = cpos0 - cdelta_pos0
    cdef numpy.ndarray[DTYPE_float32_t, ndim = 1] cpos0_sup = cpos0 + cdelta_pos0
    cdef numpy.ndarray[DTYPE_float32_t, ndim = 1] cpos1 = pos1.ravel().astype("float32")
    cdef numpy.ndarray[DTYPE_float32_t, ndim = 1] cdelta_pos1 = delta_pos1.ravel().astype("float32")
    cdef numpy.ndarray[DTYPE_float32_t, ndim = 1] cpos1_inf = cpos1 - cdelta_pos1
    cdef numpy.ndarray[DTYPE_float32_t, ndim = 1] cpos1_sup = cpos1 + cdelta_pos1
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 2] outData = numpy.zeros((bins0, bins1), dtype="float64")
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 2] outCount = numpy.zeros((bins0, bins1), dtype="float64")
    cdef numpy.ndarray[DTYPE_float32_t, ndim = 2] outMerge = numpy.zeros((bins0, bins1), dtype="float32")
    cdef numpy.ndarray[DTYPE_float32_t, ndim = 1] edges0 = numpy.zeros(bins0, dtype="float32")
    cdef numpy.ndarray[DTYPE_float32_t, ndim = 1] edges1 = numpy.zeros(bins1, dtype="float32")

    cdef float min0, max0, min1, max1, deltaR, deltaL, deltaU, deltaD, deltaA, tmp
    cdef float pos0_min, pos0_max, pos1_min, pos1_max, pos0_maxin, pos1_maxin
    cdef float fbin0_min, fbin0_max, fbin1_min, fbin1_max
    cdef long  bin0_max, bin0_min, bin1_max, bin1_min
    cdef double epsilon = 1e-10
    cdef double data

    if pos0Range is not None and len(pos0Range) == 2:
        pos0_min = min(pos0Range)
        pos0_maxin = max(pos0Range)
    else:
        pos0_min = max(0.0, cpos0_inf.min())
        pos0_maxin = cpos0_sup.max()
    pos0_max = pos0_maxin * (1 + numpy.finfo(numpy.float32).eps)

    if pos1Range is not None and len(pos1Range) > 1:
        pos1_min = min(pos1Range)
        pos1_maxin = max(pos1Range)
    else:
        tmp = cdelta_pos1.min()
        pos1_min = cpos1.min() - tmp
        pos1_maxin = cpos1.max() + tmp
    pos1_max = pos1_maxin * (1 + numpy.finfo(numpy.float32).eps)

    cdef float dpos0 = (pos0_max - pos0_min) / (< float > (bins0))
    cdef float dpos1 = (pos1_max - pos1_min) / (< float > (bins1))

    with nogil:
        for i in range(bins0):
                edges0[i] = pos0_min + (0.5 +< double > i) * dpos0
        for i in range(bins1):
                edges1[i] = pos1_min + (0.5 +< double > i) * dpos1

        for idx in range(size):
            data = cdata[idx]
            min0 = cpos0_inf[idx]
            max0 = cpos0_sup[idx]
            min1 = cpos1_inf[idx]
            max1 = cpos1_sup[idx]

            if (max0 < pos0_min) or (max1 < pos1_min) or (min0 > pos0_maxin) or (min1 > pos1_maxin) :
                continue

            if min0 < pos0_min:
                min0 = pos0_min
            if min1 < pos1_min:
                min1 = pos1_min
            if max0 > pos0_maxin:
                max0 = pos0_maxin
            if max1 > pos1_maxin:
                max1 = pos1_maxin


            fbin0_min = getBinNr(min0, pos0_min, dpos0)
            fbin0_max = getBinNr(max0, pos0_min, dpos0)
            fbin1_min = getBinNr(min1, pos1_min, dpos1)
            fbin1_max = getBinNr(max1, pos1_min, dpos1)

            bin0_min = < long > floor(fbin0_min)
            bin0_max = < long > floor(fbin0_max)
            bin1_min = < long > floor(fbin1_min)
            bin1_max = < long > floor(fbin1_max)


            if bin0_min == bin0_max:
                if bin1_min == bin1_max:
                    #All pixel is within a single bin
                    outCount[bin0_min, bin1_min] += 1.0
                    outData[bin0_min, bin1_min] += data
                else:
                    #spread on more than 2 bins
                    deltaD = (< float > (bin1_min + 1)) - fbin1_min
                    deltaU = fbin1_max - (< double > bin1_max)
                    deltaA = 1.0 / (fbin1_max - fbin1_min)

                    outCount[bin0_min, bin1_min] += < double > deltaA * deltaD
                    outData[bin0_min, bin1_min] += data * deltaA * deltaD

                    outCount[bin0_min, bin1_max] += < double > deltaA * deltaU
                    outData[bin0_min, bin1_max] += data * deltaA * deltaU
                    for j in range(bin1_min + 1, bin1_max):
                        outCount[bin0_min, j] += < double > deltaA
                        outData[bin0_min, j] += data * deltaA

            else: #spread on more than 2 bins in dim 0
                if bin1_min == bin1_max:
                    #All pixel fall on 1 bins in dim 1
                    deltaA = 1.0 / (fbin0_max - fbin0_min)
                    deltaL = (< float > (bin0_min + 1)) - fbin0_min
                    outCount[bin0_min, bin1_min] += < double > deltaA * deltaL
                    outData[bin0_min, bin1_min] += < double > data * deltaA * deltaL
                    deltaR = fbin0_max - (< float > bin0_max)
                    outCount[bin0_max, bin1_min] += < double > deltaA * deltaR
                    outData[bin0_max, bin1_min] += < double > data * deltaA * deltaR
                    for i in range(bin0_min + 1, bin0_max):
                            outCount[i, bin1_min] += < double > deltaA
                            outData[i, bin1_min] += < double > data * deltaA
                else:
                    #spread on n pix in dim0 and m pixel in dim1:
                    deltaL = (< float > (bin0_min + 1)) - fbin0_min
                    deltaR = fbin0_max - (< float > bin0_max)
                    deltaD = (< float > (bin1_min + 1)) - fbin1_min
                    deltaU = fbin1_max - (< float > bin1_max)
                    deltaA = 1.0 / ((fbin0_max - fbin0_min) * (fbin1_max - fbin1_min))

                    outCount[bin0_min, bin1_min] += < double > deltaA * deltaL * deltaD
                    outData[bin0_min, bin1_min] += < double > data * deltaA * deltaL * deltaD

                    outCount[bin0_min, bin1_max] += < double > deltaA * deltaL * deltaU
                    outData[bin0_min, bin1_max] += < double > data * deltaA * deltaL * deltaU

                    outCount[bin0_max, bin1_min] += < double > deltaA * deltaR * deltaD
                    outData[bin0_max, bin1_min] += < double > data * deltaA * deltaR * deltaD

                    outCount[bin0_max, bin1_max] += < double > deltaA * deltaR * deltaU
                    outData[bin0_max, bin1_max] += < double > data * deltaA * deltaR * deltaU
                    for i in range(bin0_min + 1, bin0_max):
                            outCount[i, bin1_min] += < double > deltaA * deltaD
                            outData[i, bin1_min] += < double > data * deltaA * deltaD
                            for j in range(bin1_min + 1, bin1_max):
                                outCount[i, j] += < double > deltaA
                                outData[i, j] += < double > data * deltaA
                            outCount[i, bin1_max] += < double > deltaA * deltaU
                            outData[i, bin1_max] += < double > data * deltaA * deltaU
                    for j in range(bin1_min + 1, bin1_max):
                            outCount[bin0_min, j] += < double > deltaA * deltaL
                            outData[bin0_min, j] += < double > data * deltaA * deltaL

                            outCount[bin0_max, j] += < double > deltaA * deltaR
                            outData[bin0_max, j] += < double > data * deltaA * deltaR

        for i in range(bins0):
            for j in range(bins1):
                if outCount[i, j] > epsilon:
                    outMerge[i, j] = outData[i, j] / outCount[i, j]
                else:
                    outMerge[i, j] = dummy
    return outMerge.T, edges0, edges1, outData.T, outCount.T

