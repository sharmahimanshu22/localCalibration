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

import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, utils, datasets
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sklearn.metrics as metrics
import statistics





def ece(y_test, y_test_pred_class, y_test_pred_prob, M, pnratio_train, pnratio_calibrate, alpha, calibmethod, outdir):
    M = 49
    max_p = np.array([max(e,1-e) for e in y_test_pred_prob])
    correct_labels = np.array([1 if y_test[i] == y_test_pred_class[i] else 0 for i in range(len(y_test))])

    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    l1 = []
    l2 = []
    xs = []
    pos_accuracy = []
    neg_accuracy = []
    left_accuracies = []
    right_accuracies = []
    ece =  0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.array([i for i in range(len(max_p)) if max_p[i] <= bin_upper.item() and max_p[i] > bin_lower.item()])
        prob_in_bin = in_bin.size/len(max_p)
        if prob_in_bin > 0:
            accuracy_in_bin = correct_labels[in_bin].mean()

            pos_accuracy_in_bin = np.sum([1 if correct_labels[i] == 1 and y_test[i] == 1 else 0 for i in in_bin ])/np.sum([1 if y_test[i]==1 else 0 for i in in_bin])
            neg_accuracy_in_bin = np.sum([1 if correct_labels[i] == 1 and y_test[i] == 0 else 0 for i in in_bin ])/np.sum([1 if y_test[i]==0 else 0 for i in in_bin])

            right_accuracy = np.sum([1 if correct_labels[i] == 1 and y_test_pred_prob[i] >= 0.5 else 0 for i in in_bin])/np.sum([1 if y_test_pred_prob[i] >= 0.5 else 0 for i in in_bin])
            left_accuracy = np.sum([1 if correct_labels[i] == 1 and y_test_pred_prob[i] < 0.5 else 0 for i in in_bin])/np.sum([1 if y_test_pred_prob[i] < 0.5 else 0 for i in in_bin])
            
            pos_accuracy.append(pos_accuracy_in_bin)
            neg_accuracy.append(neg_accuracy_in_bin)
            right_accuracies.append(right_accuracy)
            left_accuracies.append(left_accuracy)
            
            avg_confidence_in_bin = max_p[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
            l1.append(accuracy_in_bin)
            l2.append(avg_confidence_in_bin)
            xs.append((bin_lower+bin_upper)/2.0)
            
    #plt.plot(xs,l1, label='accuracy')
    #plt.plot(xs,l2, label='confidence')
    #plt.plot(xs,pos_accuracy, label='pos_accuracy')
    #plt.plot(xs,neg_accuracy, label='neg_accuracy')
    #plt.plot(xs,right_accuracies, label='right_accuracy')
    #plt.plot(xs,left_accuracies, label='left_accuracy')

    #plt.legend(loc='best')
        
    #fname = os.path.join(outdir, calibmethod + "_ece_separate_" + str(pnratio_train) + "_pnratio_calibrate_" + str(pnratio_calibrate) + "_alpha_" + str(alpha))
    #plt.savefig(fname + ".png")
    #plt.close()

    return ece


def mse(y_pred_prob, true_posterior):
    return np.nanmean([(y_pred_prob[i] - true_posterior[i])**2 for i in range(len(true_posterior)) ])

def msehigh(y_pred_prob, true_posterior, model_output):    
    return np.nanmean([(y_pred_prob[i] - true_posterior[i])**2 for i in range(len(true_posterior)) if model_output[i] > 0.75])


def reliabilityerror(y_test, y_pred_prob, M):

    M = 49
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    error = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        
        in_bin = np.array([i for i in range(len(y_pred_prob)) if y_pred_prob[i] <= bin_upper.item()
                          and y_pred_prob[i] > bin_lower.item()])

        prob_in_bin = in_bin.size/len(y_pred_prob)
        
        if prob_in_bin > 0:
            fraction_positive_in_bin = np.array(y_test)[in_bin].mean()
            mean_prob_in_bin = np.array(y_pred_prob)[in_bin].mean()

            diff = fraction_positive_in_bin - mean_prob_in_bin
            weighteddiffsq = diff*diff * prob_in_bin
            error += weighteddiffsq
            
    return error

