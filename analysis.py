import numpy as np
from scipy.stats import norm
from scipy.stats import dirichlet
import csv, os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from matplotlib.gridspec import GridSpec
from sklearn.metrics import auc, accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve
from scipy.stats import dirichlet
from GaussianMixDataGenerator.data.utils import AUCFromDistributions
from GaussianMixDataGenerator.data.datagen import MVNormalMixDG as GMM
from ClingenCalibration import calibration
import bisect

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, utils, datasets
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sklearn.metrics as metrics
import statistics

from metrics import *
from plots import *


def dometrics(y_test, y_test_pred_class, y_test_pred_prob, M, pnratio_train, pnratio_calibrate, alpha , calibmethod, outdir):
    ecev = ece(y_test, y_test_pred_class, y_test_pred_prob, M, pnratio_train, pnratio_calibrate, alpha, calibmethod, outdir)    
    return ecev
    

def allPlotsAndMetricsHere(X_test, y_test, true_posterior, y_test_pred_prob, local_calibrated_prob_test, platt_calibrated_prob_test, weighted_platt_calibrated_prob_test, isotonic_calibrated_prob_test, weighted_isotonic_calibrated_prob_test, pnratio_train, pnratio_calibrate, alpha, n_calib, outdir, axs1=None, axs2=None, axs3=None):
    y_test_pred_class_local = [1 if e > 0.5 else 0 for e in local_calibrated_prob_test]
    y_test_pred_class_platt = [1 if e > 0.5 else 0 for e in platt_calibrated_prob_test]
    y_test_pred_class_weighted_platt = [1 if e > 0.5 else 0 for e in weighted_platt_calibrated_prob_test]
    y_test_pred_class_isotonic = [1 if e > 0.5 else 0 for e in isotonic_calibrated_prob_test]
    y_test_pred_class_wisotonic = [1 if e > 0.5 else 0 for e in weighted_isotonic_calibrated_prob_test]

    #plotROC(y_test, y_test_pred_prob, true_posterior)
    #plotFig1(X_test.flatten(), true_posterior, y_test_pred_prob, y_test, y_test_pred_prob, true_posterior)
    #plotFig2(X_test.flatten(), true_posterior, y_test_pred_prob, platt_calibrated_prob_test, isotonic_calibrated_prob_test)
    #plotFig3(X_test.flatten(), true_posterior, local_calibrated_prob_test)
    
    #pltAllProbs(X_test.flatten(), true_posterior, y_test_pred_prob, local_calibrated_prob_test, platt_calibrated_prob_test, weighted_platt_calibrated_prob_test, isotonic_calibrated_prob_test, pnratio_train, pnratio_calibrate, pnratio_test ,alpha)
    
    
    ecev_local = dometrics(y_test, y_test_pred_class_local, local_calibrated_prob_test, 20, pnratio_train,
                           pnratio_calibrate, alpha, "local", outdir)
    ecev_platt = dometrics(y_test, y_test_pred_class_platt, platt_calibrated_prob_test, 20, pnratio_train,
                           pnratio_calibrate, alpha, "platt", outdir)
    ecev_weighted_platt = dometrics(y_test, y_test_pred_class_weighted_platt, weighted_platt_calibrated_prob_test, 20,
                                    pnratio_train, pnratio_calibrate, alpha, "weightedplatt", outdir)
    ecev_isotonic = dometrics(y_test, y_test_pred_class_isotonic, isotonic_calibrated_prob_test, 20,
                              pnratio_train, pnratio_calibrate, alpha, "isotonic", outdir)

    


    '''
    with open(os.path.join(outdir,"mse.csv"),'a') as f:
        f.write(str(pnratio_train) + "\t" + str(pnratio_calibrate) + "\t" + str(alpha) + "\t" + str(ecev_local) + "\t" + str(ecev_platt) + "\t" + str(ecev_isotonic) + "\t" + str(ecev_weighted_platt) + "\n")

    with open(os.path.join(outdir,"ecev.csv"),'a') as f:
        f.write(str(pnratio_train) + "\t" + str(pnratio_calibrate) + "\t" + str(alpha) + "\t" + str(ecev_local) + "\t" + str(ecev_platt) + "\t" + str(ecev_isotonic) + "\t" + str(ecev_weighted_platt) + "\n")
    '''
    plotCalibrationCurve(y_test, local_calibrated_prob_test, platt_calibrated_prob_test, weighted_platt_calibrated_prob_test, isotonic_calibrated_prob_test, weighted_isotonic_calibrated_prob_test, pnratio_train, pnratio_calibrate, alpha, n_calib, ecev_local, outdir, axs1)

    pltPosteriorComparisonVsModel(true_posterior,  y_test_pred_prob, local_calibrated_prob_test, platt_calibrated_prob_test, weighted_platt_calibrated_prob_test, isotonic_calibrated_prob_test, weighted_isotonic_calibrated_prob_test, pnratio_train, pnratio_calibrate, alpha, n_calib, ecev_local, outdir,axs2)

    pltPosteriorComparison(true_posterior,  y_test_pred_prob, local_calibrated_prob_test, platt_calibrated_prob_test, weighted_platt_calibrated_prob_test, isotonic_calibrated_prob_test, pnratio_train, pnratio_calibrate, alpha, n_calib, ecev_local, outdir,axs3)


    print("MSE: ", mse(y_test_pred_prob, true_posterior))
    print("MSEStat: " , reliabilityerror(y_test, local_calibrated_prob_test, 49))
    return


