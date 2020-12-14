#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_test_SVM_HMM_AdaBoosting.py
@Time    :   2020/12/14 12:03
@Author  :   Yinghao Ma
@Version :   0.0.1
@Contact :   yinghaom@andrew.cmu.edu
@License :   Copyright @ 2020 LazyComposer
@Desc    :
'''

import os
import tqdm
import numpy as np
import pandas as pd
from eval import show_results


def get_data(path_name, feature, extra_feature):
    data = pd.read_csv(path_name, index_col=0)  # keep_default_na=False)
    data = data.loc[:, ~data.columns.str.match('Unnamed')]
    train = data[data["split"] == "train"]
    if extra_feature == "stc":
        begin_mark = 2
    #     elif feature =="MFCC" or feature =="MFCCnorm": #不能or "xxx"
    #         begin_mark = "aver_mfcc_band1"
    #     elif feature == "log":
    #         begin_mark = "ave_logf_1"
    else:
        begin_mark = 15
    x_whole = data.iloc[:, begin_mark:]
    x_train = train.iloc[:, begin_mark:]
    y_train = train.iloc[:, 1]  # python begin with 0 and MATLAB with 1
    y_whole = data.iloc[:, 1]

    #     x_train = pd.DataFrame.to_numpy(x_train)
    #     y_train = pd.DataFrame.to_numpy(y_train)
    #     x_whole = pd.DataFrame.to_numpy(x_whole)
    #     y_whole = pd.DataFrame.to_numpy(y_whole)
    return x_train, y_train, x_whole, y_whole, data


def save_pred(clf, data, x_data, name, tag):
    y_hat = clf.predict(x_data)
    tem = data.iloc[:, 0:2]
    tem = pd.concat([tem, pd.DataFrame(columns=['prediction'])])
    tem["prediction"] = y_hat
    tem.to_csv(path + f"../../{tag}_class{name[0]}.csv")

#HMM will input the observation and outupt the hiddenstate 0 or 1 for classification
from seqlearn import hmm

method = "HMM"
features = os.listdir("E:/课程/11755ML4SP/project/feature/")
for feature in features:
    print(feature)
    path = f"E:/课程/11755ML4SP/project/feature/{feature}/"
    names = os.listdir(path)
    for i in tqdm.tqdm(range(len(names))):#
        x_train, y_train, x_whole, y_whole, data = get_data(path+names[i],feature, "stc")
        clf = hmm.MultinomialHMM(decode='viterbi', alpha=0.01)
        clf.fit(x_train, y_train, 1)
        save_pred(clf, data, x_whole, names[i], f"{feature}_{method}_stc")

        x_train, y_train, x_whole, y_whole, data = get_data(path+names[i],feature, "non-stc")
        clf = hmm.MultinomialHMM(decode='viterbi', alpha=0.01)
        clf.fit(x_train, y_train, 1)
        save_pred(clf,data, x_whole, names[i], f"{feature}_{method}_non")

    show_results("E:\\课程\\11755ML4SP\\project", feat_type=f"{feature}_{method}")

#SVM-rbf you can change other kernel function
from sklearn import svm

method = "SVMrbf"
features = os.listdir("E:/课程/11755ML4SP/project/feature/")
for feature in ['logfbank_long', 'mfcc_long']:#features:
    print(feature)
    path = f"E:/课程/11755ML4SP/project/feature/{feature}/"
    names = os.listdir(path)
    for i in range(len(names)):#
        x_train, y_train, x_whole, y_whole, data = get_data(path+names[i],feature, "stc")
        clf = svm.SVC(C=0.5,kernel='rbf')
    #     print(x_train, y_train)
        clf.fit(x_train, y_train)
        save_pred(clf, data, x_whole, names[i], f"{feature}_{method}_stc")

        x_train, y_train, x_whole, y_whole, data = get_data(path+names[i],feature, "non-stc")
        clf = svm.SVC(C=0.5,kernel='rbf')
        clf.fit(x_train, y_train)
        save_pred(clf,data, x_whole, names[i], f"{feature}_{method}_non")

    show_results("E:\\课程\\11755ML4SP\\project", feat_type=f"{feature}_{method}")

#AddBoosting
from sklearn.ensemble import AdaBoostClassifier

method = "AddBoosting"
feature = "log" # "MFCCnorm"
features = os.listdir("E:/课程/11755ML4SP/project/feature/")
for feature in features:
    print(feature)
    path = f"E:/课程/11755ML4SP/project/feature/{feature}/"
    names = os.listdir(path)
    for i in range(len(names)):#
        x_train, y_train, x_whole, y_whole, data = get_data(path+names[i],feature, "stc")
        clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        clf.fit(x_train, y_train)
        save_pred(clf, data, x_whole, names[i], f"{feature}_{method}_stc")

        x_train, y_train, x_whole, y_whole, data = get_data(path+names[i],feature, "non-stc")
        clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        clf.fit(x_train, y_train)
        save_pred(clf,data, x_whole, names[i], f"{feature}_{method}_non")

    show_results("E:\\课程\\11755ML4SP\\project", feat_type=f"{feature}_{method}")