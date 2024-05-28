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



def getNN10():
    print("nn10\n")
    NUM_CLASSES = 2
    input_layer = layers.Input((10,))
    x = layers.Flatten()(input_layer)
    x = layers.Dense(60,activation="relu")(x)
    x = layers.Dense(600, activation="relu")(x)
    x = layers.Dense(60, activation="relu")(x)
    output_layer = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(input_layer, output_layer)
    return model


def getNN100():
    NUM_CLASSES = 2
    input_layer = layers.Input((100,))
    x = layers.Flatten()(input_layer)
    
    x = layers.Dense(600,activation="relu")(x)
    x = layers.Dense(6000, activation="relu")(x)
    x = layers.Dense(600, activation="relu")(x)
    x = layers.Dense(60, activation="relu")(x)
    output_layer = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(input_layer, output_layer)
    return model

def getNN3():
    NUM_CLASSES = 2
    input_layer = layers.Input((3,))
    x = layers.Flatten()(input_layer)
    x = layers.Dense(120, activation="relu")(x)
    x = layers.Dense(360, activation="relu")(x)
    x = layers.Dense(120, activation="relu")(x)
    x = layers.Dense(60, activation="relu")(x)
    output_layer = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(input_layer, output_layer)
    return model

def getNN2():
    NUM_CLASSES = 2
    input_layer = layers.Input((2,))
    x = layers.Flatten()(input_layer)
    x = layers.Dense(120, activation="relu")(x)
    x = layers.Dense(60, activation="relu")(x)
    output_layer = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(input_layer, output_layer)
    return model

def getNN1():
    NUM_CLASSES = 2
    input_layer = layers.Input((1,))
    x = layers.Flatten()(input_layer)
    x = layers.Dense(60, activation="relu")(x)
    output_layer = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(input_layer, output_layer)
    return model


def getNNModelDim(diminput):
    if diminput == 1:
        return getNN1()
    if diminput == 2:
        return getNN2()
    if diminput == 3:
        return getNN3()
    if diminput == 10:
        return getNN10()
    if diminput == 100:
        return getNN100()

    return
    
def getNNModel():
    NUM_CLASSES = 2
    input_layer = layers.Input((1,))
    
    x = layers.Flatten()(input_layer)
    x = layers.Dense(60, activation="relu")(x)
    
    output_layer = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(input_layer, output_layer)

    return model


def trainModel(model, X_train, y_train):
    opt = optimizers.legacy.Adam(learning_rate=0.0005)
    model.compile(
        loss="binary_crossentropy", optimizer=opt, metrics=[tf.keras.metrics.AUC(), "accuracy"]
    );
    model.fit(X_train, y_train, batch_size=32, epochs=20, shuffle=True);



    
def toynn(X_train, y_train, X_calibrate, y_calibrate, X_test, y_test, alpha, pnratio_train, pnratio_calibrate, xu, yu, dim):
    
    model = getNNModelDim(dim)
    trainModel(model, X_train, y_train)

    y_calibrate_pred_nn_prob = model(X_calibrate, training=False).numpy().flatten()
    y_test_pred_nn_prob = model(X_test, training=False).numpy().flatten()
    y_unlabelled_pred_nn_prob = model(xu, training=False).numpy().flatten()

    return y_calibrate_pred_nn_prob, y_test_pred_nn_prob, y_unlabelled_pred_nn_prob
    #local_calibrated_prob_test = y_test_pred_nn_prob



def toyrf(X_train, y_train, X_calibrate, y_calibrate, X_test, y_test, alpha, pnratio_train, pnratio_calibrate, xu, yu):

    model = RandomForestClassifier(n_estimators=100,)
    model.fit(X_train, y_train);

    y_calibrate_pred_rf_prob = model.predict_proba(X_calibrate)[:, 1].flatten()
    y_test_pred_rf_prob = model.predict_proba(X_test)[:, 1].flatten()
    y_unlabelled_pred_rf_prob = model.predict_proba(xu)[:, 1].flatten()

    return y_calibrate_pred_rf_prob, y_test_pred_rf_prob, y_unlabelled_pred_rf_prob

