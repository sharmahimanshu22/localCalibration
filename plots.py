

from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import statistics
import os


def plotCalibrationCurve(y_test, local_calibrated_prob_test, platt_calibrated_prob_test, weighted_platt_calibrated_prob_test, isotonic_calibrated_prob_test, weighted_isotonic_calibrated_prob_test, pnratio_train, pnratio_calibrate, alpha, n_calib, ecev, outdir, axs=None):

    ax_calibration_curve = None
    fig = None
    
    if axs is None:
        fig = plt.figure(figsize=(30, 30))
        ax_calibration_curve = fig.add_subplot()
        ax_calibration_curve.set_xlabel("Fraction of Positives", fontsize = 30)
        ax_calibration_curve.set_ylabel("Mean Predicted Probability", fontsize = 30)
        
        #ax_calibration_curve.set_title("CalibratedPosteriorVsModel_pnratio_train_" + str(pnratio_train) + "_pnratio_calibrate_" + str(pnratio_calibrate) + "_alpha_" + str(alpha), fontsize="20")
    else:
        ax_calibration_curve = axs
        ax_calibration_curve.set_xlabel(None)
        ax_calibration_curve.set_ylabel(None)

        
    #gs = GridSpec(4, 2)
    colors = plt.get_cmap("Dark2")
    
    calibration_displays = {}
    
    display = CalibrationDisplay.from_predictions( y_test.flatten(), local_calibrated_prob_test, strategy='uniform', ref_line=True, n_bins=25, name="LocalPosterior", ax=ax_calibration_curve, color=colors(0),)
    calibration_displays["LocalPosterior"] = display
    
    display = CalibrationDisplay.from_predictions( y_test.flatten(), platt_calibrated_prob_test, strategy='uniform', ref_line=True, n_bins=25, name="Platt", ax=ax_calibration_curve, color=colors(1),)
    calibration_displays["Platt"] = display
    
    display = CalibrationDisplay.from_predictions( y_test.flatten(), weighted_platt_calibrated_prob_test, strategy='uniform', ref_line=True, n_bins=25, name="Weighted Platt", ax=ax_calibration_curve, color=colors(2),)
    calibration_displays["Weighted Platt"] = display

    display = CalibrationDisplay.from_predictions( y_test.flatten(), isotonic_calibrated_prob_test, strategy='uniform', ref_line=True, n_bins=25, name="Isotonic", ax=ax_calibration_curve, color=colors(3),)
    calibration_displays["Isotonic"] = display

    display = CalibrationDisplay.from_predictions( y_test.flatten(), weighted_isotonic_calibrated_prob_test, strategy='uniform', ref_line=True, n_bins=25, name="Weighted Isotonic", ax=ax_calibration_curve, color=colors(4),)
    calibration_displays["Weighted Isotonic"] = display

    #ax_calibration_curve.set_title("ncalib " + str(n_calib), fontsize=30)
    #ax_calibration_curve.set_title("pnratio_train: " + str(pnratio_train) + " pnratio_calibrate: " + str(pnratio_calibrate) + "\nalpha: " + str(alpha) + "_ece_" + str(ecev))

    ax_calibration_curve.set_xlabel(None)
    ax_calibration_curve.set_ylabel(None)
    ax_calibration_curve.legend(fontsize=20)
    
    if axs is None:
        fname = os.path.join(outdir,"CalibrationCurve_pnratio_train_" + str(pnratio_train) + "_pnratio_calibrate_" + str(pnratio_calibrate) + "_alpha_" + str(alpha) + ".png")
        fig.savefig(fname)
        plt.close(fig)

def pltPosteriorComparisonVsModel(true_posterior, model_output, local_calibrated_prob_test, platt_calibrated_prob_test, weighted_platt_calibrated_prob_test, isotonic_calibrated_prob_test, weighted_isotonic_calibrated_prob_test,pnratio_train, pnratio_calibrate, alpha, n_calib, ecev, outdir, axs=None):


    allprobs = sorted(zip(model_output, true_posterior, local_calibrated_prob_test, platt_calibrated_prob_test,
                          weighted_platt_calibrated_prob_test, isotonic_calibrated_prob_test,
                          weighted_isotonic_calibrated_prob_test))


    model_sorted = [e[0] for e in allprobs]
    true_sorted = [e[1] for e in allprobs]
    local_sorted = [e[2] for e in allprobs]
    platt_sorted = [e[3] for e in allprobs]
    weightedplatt_sorted = [e[4] for e in allprobs]
    isotonic_sorted = [e[5] for e in allprobs]
    wisotonic_sorted = [e[6] for e in allprobs]
    
    fig = None
    ax = None

    if axs is None:
        fig, ax = plt.subplots(1,1,figsize=(30,30), layout='compressed')
        #ax.set_title("CalibratedPosteriorVsModel_pnratio_train_" + str(pnratio_train) + "_pnratio_calibrate_" + str(pnratio_calibrate) + "_alpha_" + str(alpha), fontsize="30")
        ax.set_xlabel("Model Score", fontsize = 30)
        ax.set_ylabel("Calibrated Posterior", fontsize = 30)
        
    else:
        ax = axs

    #ax.scatter(model_output, isotonic_calibrated_prob_test, s=3, linewidths=0, color='r', label='Isotonic')

    
    ax.plot(model_sorted,local_sorted, linewidth=3, color='r', label='Local')
    ax.plot(model_sorted,platt_sorted, linewidth=3, color='g', label='Platt')
    ax.plot(model_sorted,weightedplatt_sorted,linewidth=3, color='brown', label='WeightedPlatt')
    ax.plot(model_sorted,isotonic_sorted, linewidth=3, color='orange', label='Isotonic')
    #ax.plot(model_sorted,wisotonic_sorted, linewidth=3, color='yellow', label='Weighted Isotonic')
    ax.scatter(model_sorted,true_sorted, s=8, color= 'black', label='True')
    ax.plot(model_sorted,model_sorted, linewidth=3, color= 'blue', label='Model')
    
    
    #ax.set_title("ncalib " + str(n_calib), fontsize="20")
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    #ax.set_xlabel("True Posterior", fontsize = 5)
    #ax.set_ylabel("Computed Posterior", fontsize = 5)
    #ax.set_title("pnratio_train_" + str(pnratio_train) + "_pnratio_calibrate_" + str(pnratio_calibrate) + "\nalpha_" + str(alpha) + "_ece_"+ str(ecev))
    ax.legend(loc='best', fontsize="30", markerscale=8)

    if axs is None:
        fname = os.path.join(outdir, "CalibratedPosteriorVsModel_pnratio_train_" + str(pnratio_train) + "_pnratio_calibrate_" + str(pnratio_calibrate) + "_alpha_" + str(alpha))
        fig.savefig(fname + ".png")
        plt.close(fig)

    
def pltPosteriorComparison(true_posterior, model_output, local_calibrated_prob_test, platt_calibrated_prob_test, weighted_platt_calibrated_prob_test, isotonic_calibrated_prob_test, pnratio_train, pnratio_calibrate, alpha, n_calib, ecev, outdir, axs=None):

    
    fig = None
    ax = None

    if axs is None:
        fig, ax = plt.subplots(1,1,figsize=(30,30))
        ax.set_title("CalibratedPosteriorVsTrue_pnratio_train_" + str(pnratio_train) + "_pnratio_calibrate_" +
                     str(pnratio_calibrate) + "_alpha_" + str(alpha), fontsize="30")
        ax.set_xlabel("True Posterior", fontsize = 20)
        ax.set_ylabel("Computed Posterior", fontsize = 20)
    
    else:
        ax = axs

    #ax.scatter(model_output, isotonic_calibrated_prob_test, s=3, linewidths=0, color='r', label='Isotonic')

    
    ax.scatter(true_posterior,local_calibrated_prob_test, s=3, linewidths=0, color='r', label='Local')
    ax.scatter(true_posterior,platt_calibrated_prob_test, s=3, linewidths=0, color='g', label='Platt')
    ax.scatter(true_posterior,weighted_platt_calibrated_prob_test, s=3,linewidths=0, color='brown', label='WeightedPlatt')
    ax.scatter(true_posterior,isotonic_calibrated_prob_test, s=3, linewidths=0, color='orange', label='Isotonic')
    ax.scatter(true_posterior,true_posterior, s=3, linewidths=0, color= 'black', label='True')
    ax.scatter(true_posterior,model_output, s=3, linewidths=0, color= 'blue', label='Model')
    
    
    #ax.set_title("ncalib " + str(n_calib), fontsize="20")
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    #ax.set_xlabel("True Posterior", fontsize = 5)
    #ax.set_ylabel("Computed Posterior", fontsize = 5)
    #ax.set_title("pnratio_train_" + str(pnratio_train) + "_pnratio_calibrate_" + str(pnratio_calibrate) + "\nalpha_" + str(alpha) + "_ece_"+ str(ecev))
    ax.legend(loc='best', fontsize="30", markerscale=10)

    if axs is None:
        fname = os.path.join(outdir, "CalibratedPosteriorVsTrue_pnratio_train_" + str(pnratio_train) + "_pnratio_calibrate_" + str(pnratio_calibrate) + "_alpha_" + str(alpha))
        fig.savefig(fname + ".png")
        plt.close(fig)



def pltAllProbs(xs, true, model, local, platt, weightedplatt, isotonic, pnratio_train, pnratio_calibrate, pnratio_test, alpha, outdir):
    plt.scatter(xs,true,s=1, label='true')
    plt.scatter(xs, model,s=1, label='model')
    plt.scatter(xs, local, s=1,label= 'local')
    #plt.scatter(xs, platt, s=1, label = 'platt')
    plt.scatter(xs, weightedplatt, s=1, label = 'weightedplatt')
    #plt.scatter(xs, isotonic, s=1, label = 'isotonic')
    plt.title("Score")
    
    plt.legend(loc='best')
        
    fname = os.path.join(outdir, "All Probs" + "_pnratio_train_" + str(pnratio_train) + "_pnratio_calibrate_" + str(pnratio_calibrate) + "_pnratio_test_" + str(pnratio_test) + "_alpha_" + str(alpha))
    plt.savefig(fname + ".svg")
    plt.close()

def plotData(xpos, xneg, datatype, outdir):
    plt.hist(xpos, bins=30, alpha=0.5)
    plt.hist(xneg, bins=30, alpha=0.5)
    fname = os.path.join(outdir,datatype)
    plt.savefig(fname + "_data.png")
    plt.close()

def plotTrueVSModel(true, model, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))

    ax1.scatter(true,model, s=2, label='true', color='blue')
    
    plt.savefig(os.path.join(outdir, "fig1.png"))
    plt.close(fig)

    
def plotFig1(xs, true, model, y_test_true, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))

    ax1.scatter(xs ,true, s=2, label='true', color='blue')
    ax1.scatter(xs , model, s=2, label='model', color='red')
    ax1.set_title("Model Output", fontsize = 40)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.set_ylabel("Raw Score", fontsize=20)
    ax1.set_xlabel("Feature", fontsize = 20)
    
    ax1.legend(loc='upper left', fontsize=20, markerscale=5)

    fpr, tpr, threshold = metrics.roc_curve(y_test_true, true)
    roc_auc = metrics.auc(fpr, tpr)
    ax2.plot(fpr, tpr ,linewidth=3.0, alpha=1, label = 'True AUC = %0.2f' % roc_auc, color='blue')
    fpr, tpr, threshold = metrics.roc_curve(y_test_true, model)
    roc_auc = metrics.auc(fpr, tpr)
    ax2.plot(fpr, tpr, linewidth = 1.0, alpha=1, label = 'Model AUC = %0.2f' % roc_auc, color='red')
    
    ax2.legend(loc = 'upper left', fontsize=20)
    ax2.plot([0, 1], [0, 1],'r--')
    ax2.axis(xmin=0.0,xmax=1.0)
    ax2.axis(xmin=0.0,xmax=1.0)
    ax2.tick_params(axis='x', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    
    ax2.set_ylabel('True Positive Rate', fontsize = 20)
    ax2.set_xlabel('False Positive Rate', fontsize = 20)
    ax2.set_title('ROC Curve', fontsize = 40)
    
    plt.savefig(os.path.join(outdir, "fig1.png"))
    plt.close(fig)

    
def plotFig2(xs, true, model, platt, isotonic, local, outdir):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(45, 15))

    ax1.scatter(model, local)
    ax1.scatter(model,isotonic)
    ax1.set_ylabel("LocalAndIsotonic", fontsize=20)
    ax1.set_xlabel("Model", fontsize = 20)
    
    ax2.scatter(true, model)
    ax2.set_ylabel("Model", fontsize=20)
    ax2.set_xlabel("True", fontsize = 20)

    ax3.scatter(true, local)
    ax3.set_ylabel("Local", fontsize=20)
    ax3.set_xlabel("True", fontsize = 20)

    '''
    ax1.scatter(xs ,true, s=2, label='true', color='blue')
    ax1.scatter(xs , model, s=2, label='model', color='red')
    ax1.scatter(xs ,platt, s=2, label='platt', color='green')
    ax1.scatter(xs , isotonic, s=2, label='isotonic', color='orange')

    ax1.set_title("Probability Score (Prior 0.5)", fontsize = 40)
    
    ax1.legend(loc='lower right', fontsize=20, markerscale=5)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.set_ylabel("Score", fontsize=20)
    ax1.set_xlabel("Feature", fontsize = 20)
    '''

    print("outdir", outdir)
    plt.savefig(os.path.join(outdir, "fig2.png"))
    plt.close(fig)

    
def plotFig3(xs, true, local, outdir):
    fig, (ax1) = plt.subplots(1, 1, figsize=(15, 15))

    ax1.scatter(xs ,true, s=2, label='true', color='green')
    ax1.scatter(xs , local, s=2, label='alpha-local', color='red')

    ax1.set_title("Score", fontsize = 40)
    
    ax1.legend(loc='lower right', fontsize=20, markerscale=5)

    plt.savefig(os.path.join(outdir, "fig3.png"))
    plt.close(fig)


def plotROC(y_test_true, y_test_pred, true_posterior, outdir):
    # calculate the fpr and tpr for all thresholds of the classification

    plt.title('Receiver Operating Characteristic')

    fpr, tpr, threshold = metrics.roc_curve(y_test_true, y_test_pred)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label = 'Model AUC = %0.2f' % roc_auc)
    fpr, tpr, threshold = metrics.roc_curve(y_test_true, true_posterior)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'r', linewidth=1, label = 'True AUC = %0.2f' % roc_auc)
    
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(os.path.join(outdir, "AUC.png"))
    plt.close()            
