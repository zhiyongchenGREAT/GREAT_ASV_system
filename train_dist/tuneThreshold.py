#!/usr/bin/python
#-*- coding: utf-8 -*-

import os
import glob
import sys
import time
from sklearn import metrics
import numpy
import pdb

def compute_c_norm(fnr, fpr, p_target, c_miss=1, c_fa=1):
    """ computes normalized minimum detection cost function (DCF) given
        the costs for false accepts and false rejects as well as a priori
        probability for target speakers
    """

    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    c_det, c_det_ind = min(dcf), numpy.argmin(dcf)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))

    return c_det/c_def

def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
    
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    fnr = fnr*100
    fpr = fpr*100

    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])
    
    return (tunedThreshold, eer, fpr, fnr)

def tuneThresholdfromScore_std(scores, labels, target_fa=None, target_fr=None):
    
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    ## compute minc@0.01 and minc@0.001
    minc_s = compute_c_norm(fnr, fpr, p_target=0.01)
    minc_ss = compute_c_norm(fnr, fpr, p_target=0.001)
    
    fnr = fnr*100
    fpr = fpr*100

    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    if target_fa:
        for tfa in target_fa:
            idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])
    
    return (tunedThreshold, eer, fpr, fnr, minc_s, minc_ss)
