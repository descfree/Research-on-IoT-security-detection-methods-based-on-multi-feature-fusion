import numpy as np
import tensorflow
from keras.models import load_model
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             precision_score, auc)
import matplotlib.pyplot as plt
from kdd_processing import kdd_encoding

from sklearn.metrics import roc_curve, auc
import csv
from sklearn import metrics
import os
import random
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

params = {'train_data': 160000, 'features_nb': 4,
          'batch_size': 1024, 'encoder': 'standarscaler',
          'dataset': 'kdd'}

model_name = 'D:/Research on IoT security detection methods based on multi-feature fusion/Research on IoT security detection methods based on multi-feature fusion/models/' + \
             'RNN_7_594021_rmsprop_sigmoid_1_128_1024_0.2_RNN_labelencoder_1668931521.9302652'



def load_data():

    x_train, x_test, y_train, y_test = kdd_encoding(params)



    x_train = np.array(x_train).reshape([-1, x_train.shape[1], 1])
    x_test = np.array(x_test).reshape([-1, x_test.shape[1], 1])
    print(x_train, x_test, y_train, y_test)
    return x_train, x_test, y_train, y_test



def print_results(params, model, x_train, x_test, y_train, y_test):
    print('Val loss and acc:')
    print(model.evaluate(x_test, y_test, params['batch_size']))

    y_pred = model.predict(x_test, params['batch_size'])
    # print(y_pred)

    print('\nConfusion Matrix:')
    conf_matrix = confusion_matrix(y_test.argmax(axis=1),
                                   y_pred.argmax(axis=1))


    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    TP = np.diag(conf_matrix)
    TN = conf_matrix.sum() - (FP + FN + FP)

    print('\nTPR:')

    print(TP / (TP + FN))
    TPR = TP / (TP + FN)
    print('\nFPR:')

    print(FP / (FP + TN))
    FPR = FP / (FP + TN)

    cost_matrix =[[0, 1, 2, 2, 2],
                   [1, 0, 2, 2, 2],
                   [2, 1, 0, 2, 2],
                   [4, 2, 2, 0, 2],
                   [4, 2, 2, 2, 0]]

    tmp_matrix = np.zeros((2, 2))

    for i in range(2):
        for j in range(2):
            tmp_matrix[i][j] = conf_matrix[i][j] * cost_matrix[i][j]


    print('\nCost:')
    print(tmp_matrix.sum()/conf_matrix.sum())

    print('\nAUC:')
    print(roc_auc_score(y_true=y_test, y_score=y_pred, average=None))
    AUC=roc_auc_score(y_true=y_test, y_score=y_pred, average=None)
    fpr, tpr, thersholds = roc_curve(y_test.argmax(axis=1), y_pred.argmax(axis=1), pos_label=1)

    roc_auc = metrics.auc(fpr, tpr)

    print('\nPrecision:')

    print(precision_score(y_true=y_test.argmax(axis=1),
                          y_pred=y_pred.argmax(axis=1), average=None))
    pre = precision_score(y_true=y_test.argmax(axis=1),
                          y_pred=y_pred.argmax(axis=1), average=None)

    acc= metrics.accuracy_score(y_pred.argmax(axis=1),y_test.argmax(axis=1) )
    print('\nacc:')
    print(acc)
    return TPR, FPR, AUC, pre

def res(params, model, x_train, x_test, y_train, y_test):
    y_pred = model.predict(x_test, params['batch_size'])


    conf_matrix = confusion_matrix(y_test.argmax(axis=1),
                                   y_pred.argmax(axis=1))


    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    TP = np.diag(conf_matrix)
    TN = conf_matrix.sum() - (FP + FN + FP)



    TPR=TP / (TP + FN)




    FPR=FP / (FP + TN)
    AUC = roc_auc_score(y_true=y_test, y_score=y_pred, average=None)

    pre=precision_score(y_true=y_test.argmax(axis=1),
                          y_pred=y_pred.argmax(axis=1), average=None)

    return TPR,FPR,AUC,pre

if __name__ == "__main__":


    model = load_model(model_name)
    model.summary()

    x_train, x_test, y_train, y_test = load_data()
    TPR, FPR, AUC, pre=print_results(params, model, x_train, x_test, y_train, y_test)






