import numpy as np
from scipy.stats import norm
import csv, os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from matplotlib.gridspec import GridSpec
from sklearn.metrics import auc, accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, utils, datasets
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sklearn.metrics as metrics
import statistics

from datautils import *
from analysis import *
from parser import *
from calibrationmethods import *
from modelutils import *    

        
            


def run(outdirtop, pnratio_train, pnratio_calibrate, pnratio_test, alpha, dim, model, n_train, n_calibrate, n_test, mu_pos = None, mu_neg= None, sig_pos = None, sig_neg = None, axs1=None, axs2=None, axs3=None):
    
    gmm = None
    if dim == 1:
        gmm = buildGaussianMixDataGenerator(mu_pos, mu_neg, sig_pos, sig_neg)
    else:
        gmm = buildGaussianMixDataGeneratorDim(dim, None)

    gmm.printInfo()

    outdir = os.path.join(outdirtop, model)

    outdir = os.path.join(outdir,"n_calibrate_" + str(n_calibrate))
    outdir = os.path.join(outdir,"n_train_"+ str(n_train))
    #outdir = os.path.join(outdir, "n_test_" + str(n_test))
    os.makedirs(outdir, exist_ok=True)

    # data preparation
    X_train, y_train = gmm.pn_data(n_train, pnratio_train)[0:2]
    X_calibrate, y_calibrate = gmm.pn_data(n_calibrate, pnratio_calibrate)[0:2]
    X_test, y_test = gmm.pn_data(n_test, pnratio_test)[0:2]


    
    X_test_pos = []
    X_test_neg = []
    for kk in range(len(X_test)):
        if y_test[kk] == 0:
            X_test_neg.append(X_test[kk][0])
        else:
            X_test_pos.append(X_test[kk][0])

    
    w0 = (1-alpha)/(2*(1-pnratio_calibrate))
    w1 = alpha/(2*pnratio_calibrate)
    print("w0: " + str(w0) + " w1: " + str(w1) + " " + str(pnratio_calibrate))
    
    xu, yu = gmm.pn_data(5000, alpha)[0:2]


    # Finding true posterior
    true_posterior = gmm.pn_posterior(X_test, alpha)


    # Training Model
    y_calibrate_pred_prob = None
    y_test_pred_prob = None
    y_unlabelled_pred_prob = None
    
    if model == 'NeuralNetwork':
        y_calibrate_pred_prob, y_test_pred_prob, y_unlabelled_pred_prob = toynn(X_train, y_train, X_calibrate, y_calibrate, X_test, y_test, alpha, pnratio_train, pnratio_calibrate, xu, yu, dim)

    if model == "RandomForest":
        y_calibrate_pred_prob, y_test_pred_prob, y_unlabelled_pred_prob = toyrf(X_train, y_train, X_calibrate, y_calibrate, X_test, y_test, alpha, pnratio_train, pnratio_calibrate, xu, yu)



    # Calibration Model
    local_calibrated_prob_test = localCalibration(y_calibrate, y_calibrate_pred_prob, y_test_pred_prob, y_unlabelled_pred_prob, alpha)
    platt_calibrated_prob_test = getPlattCalibratedProbs(y_calibrate, y_calibrate_pred_prob, y_test_pred_prob);
    weighted_platt_calibrated_prob_test = getWeightedPlattCalibratedProbs(y_calibrate, y_calibrate_pred_prob, y_test_pred_prob, w0, w1);
    isotonic_calibrated_prob_test = getIsotonicCalibratedProbs(y_calibrate, y_calibrate_pred_prob, y_test_pred_prob)
    weighted_isotonic_calibrated_prob_test = getWeightedIsotonicCalibratedProbs(y_calibrate, y_calibrate_pred_prob, y_test_pred_prob, w0, w1)

    # Plotting the results

    #plotFig1(X_test, true_posterior, y_test_pred_prob, y_test, ".")
    plotFig2(X_test, true_posterior, y_test_pred_prob, platt_calibrated_prob_test, isotonic_calibrated_prob_test, local_calibrated_prob_test, outdir)
    
    return

    allPlotsAndMetricsHere(X_test, y_test, true_posterior, y_test_pred_prob, local_calibrated_prob_test, platt_calibrated_prob_test, weighted_platt_calibrated_prob_test, isotonic_calibrated_prob_test, weighted_isotonic_calibrated_prob_test, pnratio_train, pnratio_calibrate, alpha,n_calibrate, outdir, axs1, axs2, axs3)
    
    recordMetrics(y_test, true_posterior, y_test_pred_prob, local_calibrated_prob_test, platt_calibrated_prob_test, weighted_platt_calibrated_prob_test, isotonic_calibrated_prob_test, outdirtop, alpha, n_calibrate, pnratio_calibrate)


def recordMetrics(y_test, true_posterior, y_test_pred_prob, local_calibrated_prob_test, platt_calibrated_prob_test,
                  weighted_platt_calibrated_prob_test, isotonic_calibrated_prob_test, outdir, alpha, n_calib, pnratio_calibrate):
    
    mse_model = round(mse(y_test_pred_prob, true_posterior),5)
    mse_local = round(mse(local_calibrated_prob_test, true_posterior),5) 
    mse_platt = round(mse(platt_calibrated_prob_test, true_posterior) , 5)
    mse_wplatt = round(mse(weighted_platt_calibrated_prob_test, true_posterior), 5)
    mse_isotonic = round(mse(isotonic_calibrated_prob_test, true_posterior), 5)

    mse_model_high = round(msehigh(y_test_pred_prob, true_posterior, y_test_pred_prob),5)
    mse_local_high = round(msehigh(local_calibrated_prob_test, true_posterior, y_test_pred_prob),5) 
    mse_platt_high = round(msehigh(platt_calibrated_prob_test, true_posterior, y_test_pred_prob) , 5)
    mse_wplatt_high = round(msehigh(weighted_platt_calibrated_prob_test, true_posterior, y_test_pred_prob), 5)
    mse_isotonic_high = round(msehigh(isotonic_calibrated_prob_test, true_posterior, y_test_pred_prob), 5)

    M = 49
    re_model = round(reliabilityerror(y_test, y_test_pred_prob, M),5)
    re_local = round(reliabilityerror(y_test, local_calibrated_prob_test, M),5)
    re_platt = round(reliabilityerror(y_test, platt_calibrated_prob_test, M), 5)
    re_wplatt = round(reliabilityerror(y_test, weighted_platt_calibrated_prob_test, M), 5)
    re_isotonic = round(reliabilityerror(y_test, isotonic_calibrated_prob_test, M), 5)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, true_posterior)
    aucpn = metrics.auc(fpr, tpr)


    st = str(aucpn) + '\t' + str(pnratio_calibrate) + "\t" + str(n_calib) + '\t'+ str(alpha) + "\t" + str(mse_model) + "\t" + str(mse_local) + "\t" + str(mse_platt) + "\t" + str(mse_wplatt) + "\t" + str(mse_isotonic) + "\t" + str(mse_model_high) + "\t" + str(mse_local_high) + "\t" + str(mse_platt_high) + "\t" + str(mse_wplatt_high) + "\t" + str(mse_isotonic_high) + "\n"

    if not os.path.isfile(os.path.join(outdir,"metric.csv")):
        with open(os.path.join(outdir,"metric.csv"), 'a+') as f:
            f.write('\t'.join(["auc","pnratio_calibrate","n_calib","alpha","mse_model","mse_local", "mse_platt", "mse_wplatt","mse_isotonic", "high_model", "high_local", "high_platt", "high_wplatt", "high_isotonic"]) + "\n")

            
    '''
        
    st = str(aucpn) + '\t' + str(pnratio_calibrate) + "\t" + str(n_calib) + '\t'+ str(alpha) + "\t" + str(mse_model) + "\t" + str(mse_local) + "\t" + str(mse_platt) + "\t" + str(mse_wplatt) + "\t" + str(mse_isotonic) + "\t" + str(re_model) + "\t" + str(re_local) + "\t" + str(re_platt) + "\t" + str(re_wplatt) + "\t" + str(re_isotonic) + "\n"

    if not os.path.isfile(os.path.join(outdir,"metric.csv")):
        with open(os.path.join(outdir,"metric.csv"), 'a+') as f:
            f.write('\t'.join(["auc","pnratio_calibrate","n_calib","alpha","mse_model","mse_local", "mse_platt", "mse_wplatt","mse_isotonic", "re_model", "re_local", "re_platt", "re_wplatt", "re_isotonic"]) + "\n")
    '''     

    with open(os.path.join(outdir,"metric.csv"), 'a+') as f:
        f.write(st)
    
def main():

    
    parser = getParser()
    args = parser.parse_args()
    outdir = args.outdir
    alpha = args.alpha
    pnratio_train = args.pnratio_train
    pnratio_calibrate = args.pnratio_calibrate
    pnratio_test = args.pnratio_test
    dim = args.diminput
    model = args.model
    change = args.change

    n_train = 20000
    n_calibrate = 10000
    n_test = 5000
    
    if args.n_train is not None:
        n_train = args.n_train
    if args.n_calibrate is not None:
        n_calibrate = args.n_calibrate
    if args.n_test is not None:
        n_test = args.n_test
    os.makedirs(outdir, exist_ok=True)
        


    fig1, axs1 = plt.subplots(2, 2, figsize=(30, 30), sharex=False, sharey=False)
    fig1.supylabel('Fraction of Positives', fontsize=30, linespacing=1 )
    fig1.supxlabel('Mean Predicted Probability', fontsize=30, )

    fig2, axs2 = plt.subplots(2, 2, figsize=(30, 30), sharex=False, sharey=False, layout='compressed')
    fig2.supxlabel('Model Output', fontsize=30, )
    fig2.supylabel('Computed Posterior', fontsize=30, va='bottom')

    fig3, axs3 = plt.subplots(2, 2, figsize=(30, 30), sharex=False, sharey=False)
    fig3.supxlabel('True Posterior', fontsize=30, va='top')
    fig3.supylabel('Computed Posterior', fontsize=30)

    mu_pos = 0.5
    mu_neg = -0.5
    sig_pos = 1
    sig_neg = 1



    if args.dummy == 1:
        fig1.savefig(os.path.join(outdir, "calibrationcurve.svg"))
        plt.close(fig1)
        fig2.savefig(os.path.join(outdir, "posteriorvsmodel.svg"))
        plt.close(fig2)
        fig3.savefig(os.path.join(outdir, "posteriorvstrue.svg"))
        plt.close(fig3)
        
        return


    if change == "ncalib":
        outdir = os.path.join(outdir,"changencalib" ,"pnratiocalib_" + str(pnratio_calibrate), "alpha_" + str(alpha), "mu_pos_" + str(mu_pos) + "_mu_neg_" + str(mu_neg) + "_sig_pos_" + str(sig_pos) + "_sig_neg_" + str(sig_neg))

        n_calib = [100,1000,10000,100000]

        for i in range(len(n_calib)):
            j,k = divmod(i,2)
            print (j,k)
            axs1[j][k].set_title("n_calib " + str(n_calib[i]))
            axs2[j][k].set_title("n_calib " + str(n_calib[i]))
            axs3[j][k].set_title("n_calib " + str(n_calib[i]))
        
            run(outdir, pnratio_train, pnratio_calibrate, alpha, alpha, dim, model,
                n_train, n_calib[i], n_test, mu_pos, mu_neg, sig_pos, sig_neg, axs1 = axs1[j][k], axs2 = axs2[j][k], axs3 = axs3[j][k])


            
    if change == "pnratiocalib":
        outdir = os.path.join(outdir,"changepnratiocalib" ,"ncalib_" + str(n_calibrate), "alpha_" + str(alpha), "mu_pos_" + str(mu_pos) + "_mu_neg_" + str(mu_neg) + "_sig_pos_" + str(sig_pos) + "_sig_neg_" + str(sig_neg))

        pnratio_calib = [0.01,0.1,0.25,0.5]
        for i in range(len(pnratio_calib)):
            j,k = divmod(i,2)
            print (j,k)
            axs1[j][k].set_title("pnratio_calib " + str(pnratio_calib[i]), fontsize=30)
            axs2[j][k].set_title("pnratio_calib " + str(pnratio_calib[i]), fontsize=30)
            axs3[j][k].set_title("pnratio_calib " + str(pnratio_calib[i]), fontsize=30)
        
            run(outdir, pnratio_train, pnratio_calib[i], alpha, alpha, dim, model, 
                n_train, n_calibrate, n_test, mu_pos, mu_neg, sig_pos, sig_neg,
                axs1 = axs1[j][k], axs2 = axs2[j][k], axs3 = axs3[j][k])

    if change == "alpha":
        outdir = os.path.join(outdir,"changealpha" ,"pnratiocalib_" + str(pnratio_calibrate), "ncalib_" + str(n_calibrate), "mu_pos_" + str(mu_pos) + "_mu_neg_" + str(mu_neg) + "_sig_pos_" + str(sig_pos) + "_sig_neg_" + str(sig_neg))
        alphas = [0.5,0.25,0.1,0.01]
        for i in range(len(alphas)):
            j,k = divmod(i,2)
            print (j,k)
            axs1[j][k].set_title("Prior " + str(alphas[i]), fontsize = 30)
            axs2[j][k].set_title("Prior " + str(alphas[i]), fontsize = 30)
            axs3[j][k].set_title("Prior " + str(alphas[i]), fontsize = 30)
        
            run(outdir, pnratio_train, pnratio_calibrate, alphas[i], alphas[i], dim, model, n_train, n_calibrate,
                n_test, mu_pos, mu_neg, sig_pos, sig_neg, axs1 = axs1[j][k], axs2 = axs2[j][k], axs3 = axs3[j][k])
            
    if change == "separability":
        mu_pos = [0.5, 0.0, 0.5, 1]
        mu_neg = [-0.5, -0.5, -0.5, -1]
        sig_pos = [1,1, 2, 3]
        sig_neg = [1,1,2,3]
        outdir = os.path.join(outdir,"changeseparability" ,"pnratiocalib_" + str(pnratio_calibrate), "ncalib_" + str(n_calibrate), "alpha_" + str(alpha))
        for i in range(len(mu_pos)):
            j,k = divmod(i,2)
            axs1[j][k].set_title("mu_pos " + str(mu_pos[i]) + " sig_pos " + str(sig_pos[i]) + " mu_neg " +
                                 str(mu_neg[i]) + " sig_neg " + str(sig_neg[i]) , fontsize=30)
            axs2[j][k].set_title("mu_pos " + str(mu_pos[i]) + " sig_pos " + str(sig_pos[i]) + " mu_neg " +
                                 str(mu_neg[i]) + " sig_neg " + str(sig_neg[i]) , fontsize=30)
            axs3[j][k].set_title("mu_pos " + str(mu_pos[i]) + " sig_pos " + str(sig_pos[i]) + " mu_neg " +
                                 str(mu_neg[i])+ " sig_neg " + str(sig_neg[i]) , fontsize=30)
        
            run(outdir, pnratio_train, pnratio_calibrate, alpha, alpha, dim, model, n_train, n_calibrate,
                n_test, mu_pos[i], mu_neg[i], sig_pos[i], sig_neg[i] ,axs1 = axs1[j][k], axs2 = axs2[j][k],
                axs3 = axs3[j][k])

            
    if change == None:

        run(outdir, pnratio_train, pnratio_calibrate, alpha, alpha, dim, model, n_train, n_calibrate,
            n_test)

        
    fig1.savefig(os.path.join(outdir, "calibrationcurve.svg"))
    plt.close(fig1)
    fig2.savefig(os.path.join(outdir, "posteriorvsmodel.svg"))
    plt.close(fig2)
    fig3.savefig(os.path.join(outdir, "posteriorvstrue.svg"))
    plt.close(fig3)





    


main()
