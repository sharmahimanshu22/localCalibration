import numpy as np
from scipy.stats import norm
from scipy.stats import dirichlet
import csv, os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from matplotlib.gridspec import GridSpec
from sklearn.metrics import auc, accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve
from GaussianMixDataGenerator.data.randomParameters import NormalMixPNParameters2 as NMixPar
from scipy.stats import dirichlet
from GaussianMixDataGenerator.data.utils import AUCFromDistributions
from GaussianMixDataGenerator.data.datagen import MVNormalMixDG as GMM
from ClingenCalibration import calibration
import bisect


def lininterpol(x, x1, x2, y1, y2):
    y = y1 + (x-x1)*( y2-y1 )/(x2-x1)
    return y


def getProbabilityBasedOnLinearInterpolation(s, scores, posterior):
    ix= bisect.bisect_left(scores, s)
    ans = None
    if ix == 0:
        ans =  lininterpol(s, scores[0], scores[1], posterior[0], posterior[1])
    elif ix == len(scores):
        ans = lininterpol(s, scores[-1], scores[-2], posterior[-1], posterior[-2])
    else:
        ans = lininterpol(s, scores[ix-1], scores[ix], posterior[ix-1], posterior[ix])

    if ans < 0.0:
        return 0.0
    if ans > 1.0:
        return 1.0
    return ans

def getProbabilitiesBasedOnLinearInterpolation(scores, thresholds, posteriors):
    return [getProbabilityBasedOnLinearInterpolation(e, thresholds, posteriors) for e in scores]


def calibrateModel(scores, label, pudata, alpha):
    thresh, prob = calibration.calibrate(scores, label, pudata, alpha, 100, 0.03)
    thresh.reverse()
    prob = prob.tolist()
    prob.reverse()
    return thresh, prob


def localCalibration(y_calibrate, y_calibrate_pred_nn_prob, y_test_pred_nn_prob, y_unlabelled_pred_nn_prob, alpha):
    thresh, calibrated_prob = calibrateModel(y_calibrate_pred_nn_prob, y_calibrate.flatten(), y_unlabelled_pred_nn_prob, alpha)
    local_calibrated_prob_test = getProbabilitiesBasedOnLinearInterpolation(y_test_pred_nn_prob,thresh,calibrated_prob)
    return local_calibrated_prob_test

def getPlattCalibratedProbs(y_calibrate, y_calibrate_pred_nn_prob, y_test_pred_nn_prob):
    logreg = LogisticRegression(class_weight = {0:0.5, 1: 0.5})
    logreg.fit(y_calibrate_pred_nn_prob.reshape(-1,1), y_calibrate.reshape(len(y_calibrate),));
    platt_calibrated_prob_test = [e[1] for e in logreg.predict_proba(y_test_pred_nn_prob.reshape(-1,1))]
    return platt_calibrated_prob_test

def getWeightedPlattCalibratedProbs(y_calibrate, y_calibrate_pred_nn_prob, y_test_pred_nn_prob, w0, w1):
    logreg = LogisticRegression(class_weight = {0:w0, 1: w1})
    logreg.fit(y_calibrate_pred_nn_prob.reshape(-1,1), y_calibrate.reshape(len(y_calibrate),));
    platt_calibrated_prob_test = [e[1] for e in logreg.predict_proba(y_test_pred_nn_prob.reshape(-1,1))]
    return platt_calibrated_prob_test

def getIsotonicCalibratedProbs(y_calibrate, y_calibrate_pred_nn_prob, y_test_pred_nn_prob):
    iso_reg = IsotonicRegression().fit(y_calibrate_pred_nn_prob, y_calibrate.flatten())
    isotonic_calibrated_prob_test = iso_reg.predict(y_test_pred_nn_prob)
    return isotonic_calibrated_prob_test

def getWeightedIsotonicCalibratedProbs(y_calibrate, y_calibrate_pred_nn_prob, y_test_pred_nn_prob, w0, w1):
    weights = [w0 if y_calibrate[i] == 0 else w1 for i in range(len(y_calibrate))]
    iso_reg = IsotonicRegression().fit(y_calibrate_pred_nn_prob, y_calibrate.flatten(), weights)
    isotonic_calibrated_prob_test = iso_reg.predict(y_test_pred_nn_prob)
    return isotonic_calibrated_prob_test
