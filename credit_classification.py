""" CS5487: Course Project - German Credit Analysis """
import logging
import random

import pandas as pd

import numpy as np
import matplotlib as mlb

from numpy.dual import inv
mlb.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE


from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import ExtraTreesClassifier


SKIPROWS = 1

RANDOMFOREST = 'Randomforest'
LOGISTICREG = 'Logistic Reg'
SVM = 'SVM'
RANDOM_STATE = 12


def train_random_forest(X_train, X_test, y_train, resample, feature_sel, y_weight):
    if(feature_sel == True):
        fsm = ExtraTreesClassifier(random_state=RANDOM_STATE, n_estimators=40, max_features=5, max_depth=5, class_weight={0: 2.9, 1: 1})
        fsm = fsm.fit(X_train, y_train)
        model = SelectFromModel(fsm,  prefit=True)
        X_train = model.transform(X_train)
        X_test = model.transform(X_test)

    if(resample == True):
        sm = SMOTE(k_neighbors = 30, random_state=RANDOM_STATE)
        X_train, y_train = sm.fit_sample(X_train, y_train.ravel())

    rfc = RandomForestClassifier(n_estimators=40, oob_score=True, max_depth=5, class_weight=y_weight, random_state=RANDOM_STATE)
    rfc.fit(X_train, y_train)
    return rfc.predict(X_test)

def train_logistic_reg(X_train, X_test, y_train, resample, feature_sel, y_weight):

    if(feature_sel == True):
        fsm = LinearSVC(C=0.03, penalty="l1", dual=False, class_weight = y_weight)
        fsm = fsm.fit(X_train, y_train)
        model = SelectFromModel(fsm, prefit=True)
        X_train = model.transform(X_train)
        X_test = model.transform(X_test)

    if(resample == True):
        sm = SMOTE(k_neighbors = 30, random_state=RANDOM_STATE)
        X_train, y_train = sm.fit_sample(X_train, y_train.ravel())

    lr = LogisticRegressionCV(random_state=RANDOM_STATE, class_weight=y_weight, max_iter=1000, cv=10)
    lr.fit(X_train, y_train)
    return lr.predict(X_test)

def train_svm(X_train, X_test, y_train, resample, feature_sel, y_weight):

    if(feature_sel == True):
        fsm = LinearSVC(C=0.03, penalty="l1", dual=False)
        fsm = fsm.fit(X_train, y_train)
        model = SelectFromModel(fsm, prefit=True)
        X_train = model.transform(X_train)
        X_test = model.transform(X_test)

    if(resample == True):
        sm = SMOTE(k_neighbors = 30, random_state=RANDOM_STATE)
        X_train, y_train = sm.fit_sample(X_train, y_train.ravel())

    clf_svm = SVC(class_weight=y_weight, random_state=RANDOM_STATE, gamma='scale', kernel='poly')
    clf_svm.fit(X_train, y_train)
    return clf_svm.predict(X_test)
    


def preprocess(fpath, test_size=0.2):
    data_set = np.loadtxt(fpath, delimiter=',', skiprows=SKIPROWS)

    columns = data_set.shape[1]
    num_features = columns - 1

    X_data = data_set[:,0:num_features]
    y_data = data_set[:,columns-1:columns]

    X_train, X_test, y_train, y_test = \
        train_test_split(X_data, y_data, test_size=test_size, stratify=y_data, random_state=RANDOM_STATE)

    return (X_train, X_test, y_train.ravel(), y_test.ravel())


def evaluation(predicted, y_test, msg):
    print(msg)
    for (alg, pre) in predicted.items():
        accuracy = round(accuracy_score(y_test, pre),4)
        recall = round(recall_score(y_test, pre, pos_label=0),4)
        auc = round(roc_auc_score(y_test, pre),4)
        print("{}'s performance: accuracy,recall,roc_auc= {} {} {}".format(alg, accuracy, recall, auc))

def compare_feature_sel(X_train, X_test, y_train, y_test, y_weight):
    predicted = {}
    predicted[RANDOMFOREST] = train_random_forest(X_train, X_test, y_train, resample = False, feature_sel = False, y_weight=y_weight)
    predicted[LOGISTICREG] = train_logistic_reg(X_train, X_test, y_train,   resample = False, feature_sel = False, y_weight=y_weight)
    predicted[SVM] = train_svm(X_train, X_test, y_train,                    resample = False, feature_sel = False, y_weight=y_weight)
    evaluation(predicted, y_test, 'W/O feature selection')

    predicted[RANDOMFOREST] = train_random_forest(X_train, X_test, y_train, resample = False, feature_sel = True, y_weight=y_weight)
    predicted[LOGISTICREG]  = train_logistic_reg(X_train, X_test, y_train,  resample = False, feature_sel = True, y_weight=y_weight)
    predicted[SVM]          = train_svm(X_train, X_test, y_train,           resample = False, feature_sel = True, y_weight=y_weight)
    evaluation(predicted, y_test, 'W/ feature selection')

def compare_class_weight(X_train, X_test, y_train, y_test, y_weight1, y_weight2):
    predicted = {}
    predicted[RANDOMFOREST] = train_random_forest(X_train, X_test, y_train, resample = False, feature_sel = False, y_weight=y_weight1)
    predicted[LOGISTICREG] = train_logistic_reg(X_train, X_test, y_train,   resample = False, feature_sel = False, y_weight=y_weight1)
    predicted[SVM] = train_svm(X_train, X_test, y_train,                    resample = False, feature_sel = False, y_weight=y_weight1)
    evaluation(predicted, y_test, y_weight1)

    predicted[RANDOMFOREST] = train_random_forest(X_train, X_test, y_train, resample = False, feature_sel = False, y_weight=y_weight2)
    predicted[LOGISTICREG] = train_logistic_reg(X_train, X_test, y_train,   resample = False, feature_sel = False, y_weight=y_weight2)
    predicted[SVM] = train_svm(X_train, X_test, y_train,                    resample = False, feature_sel = False, y_weight=y_weight2)
    evaluation(predicted, y_test, y_weight2)

def compare_resampling(X_train, X_test, y_train, y_test, y_weight):
    predicted = {}
    # predicted[RANDOMFOREST] = train_random_forest(X_train, X_test, y_train, resample = False, feature_sel = False, y_weight=y_weight)
    # predicted[LOGISTICREG] = train_logistic_reg(X_train, X_test, y_train,   resample = False, feature_sel = False, y_weight=y_weight)
    # predicted[SVM] = train_svm(X_train, X_test, y_train,                    resample = False, feature_sel = False, y_weight=y_weight)
    # evaluation(predicted, y_test, 'W/O Resampling')

    predicted[RANDOMFOREST] = train_random_forest(X_train, X_test, y_train, resample = True, feature_sel = True, y_weight=y_weight)
    predicted[LOGISTICREG]  = train_logistic_reg(X_train, X_test, y_train,  resample = True, feature_sel = True, y_weight=y_weight)
    predicted[SVM]          = train_svm(X_train, X_test, y_train,           resample = True, feature_sel = True, y_weight=y_weight)
    evaluation(predicted, y_test, 'W/ Resampling')

def main():
    X_train, X_test, y_train, y_test = preprocess('./data/credit_g_normalized.csv')

    compare_resampling(X_train, X_test, y_train, y_test, y_weight={0: 2.4, 1: 1})
    # compare_feature_sel(X_train, X_test, y_train, y_test, y_weight={0: 2.9, 1: 1})
    # compare_class_weight(X_train, X_test, y_train, y_test, y_weight1={0: 1, 1: 1}, y_weight2={0: 2.9, 1: 1})



if __name__ == '__main__':
    main()
