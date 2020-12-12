import os
import numpy as np
import pandas as pd
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve


def eval(pred_path):
    pred_df = pd.read_csv(pred_path)
    # test samples only
    pred_df = pred_df[pred_df["split"]=="test"]
    # eg. 6_music_presence
    label_key = pred_df.columns[2]
    # print(pred_df[label_key])
    labels = pred_df[label_key]
    #print("np.sum(labels): ", np.sum(labels))

    print(pred_path, sum(labels)/len(labels))

    ratio = np.sum(labels) / len(labels)
    preds = pred_df["prediction"]

    
    precision, recall, _ = precision_recall_curve(labels, preds)
    TPs, FPs, FNs = confusion_matrix_coarse(labels, preds)
    auprc = sklearn.metrics.average_precision_score(labels, preds)
    
    return label_key, TPs, FPs, FNs, auprc, ratio, precision, recall


def get_svm_results(pred_dir="../../svm_prediction", feat_type="_"):
    non_stc = {}
    with_stc = {}
    for filename in sorted(os.listdir(pred_dir)):
        print(filename)
        if ".csv" not in filename:
            continue
        path = "{}/{}".format(pred_dir, filename)
        label_key, TPs, FPs, FNs, auprc, ratio, precision, recall = eval(path)
        dic = {
            "TPs" : TPs,
            "FPs" : FPs,
            "FNs" : FNs,
            "auprc" : auprc,
            "ratio" : ratio,
            "precision" : precision,
            "recall" : recall,
        }
        # only audio feature
        if "non_" in filename and feat_type in filename:
            non_stc[label_key] = dic
        # audio + stc feature
        elif "stc_" in filename and feat_type in filename:
            with_stc[label_key] = dic
    return non_stc, with_stc


def show_results(pred_dir="../../svm_prediction", num_class=8, feat_type="_"):
    non_stc, with_stc = get_svm_results(pred_dir, feat_type)
    classes = sorted(list(non_stc.keys()))
    
    stc_precision = []
    stc_recall = []
    print(with_stc)
    for c in classes:
        print(c)
        print("Only  Audio ::",  non_stc[c])
        print("STC + Audio ::",  with_stc[c])
        print('-------------------------------------------------------------------------------')
        stc_precision.append(with_stc[c]['precision'])
        stc_recall.append(with_stc[c]['recall'])

    # plot bar graph
    non_auprc = [d["auprc"] for d in non_stc.values()]
    stc_auprc = [d["auprc"] for d in with_stc.values()]
    
    non_ratio = [d["ratio"] for d in non_stc.values()]
    stc_ratio = [d["ratio"] for d in with_stc.values()]

    non_avg = np.array(non_auprc) @ np.array(non_ratio).T
    print("weighted average for non-stc:", non_avg)
    
    stc_avg = np.array(stc_auprc) @ np.array(stc_ratio).T
    print("weighted average for stc:    ", stc_avg)

    non_list = [[i+1, non_auprc[i]] for i in range(len(classes))]
    stc_list = [[i+1, stc_auprc[i]] for i in range(len(classes))]
    
    # add weighted average
    non_list.append([9, non_avg])
    stc_list.append([9, stc_avg])
    classes.append("Weighted Average")

    
    for i in range(8):
        plt.plot(stc_recall[i], stc_precision[i], lw=2, label='{} ({})'.format(classes[i], str(stc_auprc[i])[:5]))
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.show()

    x1,y1 = zip(*non_list)
    x2,y2 = zip(*stc_list)
    plt.bar(np.array(x1)-0.15, y1, width = 0.3)
    plt.bar(np.array(x2)+0.15, y2, width = 0.3)
    plt.xticks(list(range(1, len(classes)+ 1)), classes, size='smaller', rotation=20)
    plt.title("AURPC for Coarse Classes ({})".format(feat_type, non_avg, stc_avg))
    plt.xlabel("Coarse Classes")
    plt.ylabel("AUPRC")
    plt.legend(["Only Audio Features", "STC + Audio Features"])
    plt.show()

# cite: https://github.com/sonyc-project/dcase2020task5-uststc-baseline.git
def confusion_matrix_coarse(y_true, y_pred):
    """
    Counts overall numbers of true positives (TP), false positives (FP),
    and false negatives (FN) in the predictions of a system, for a single
    Boolean attribute, in a dataset of N different samples.


    Parameters
    ----------
    y_true: array of bool, shape = [n_samples,]
        One-hot encoding of true presence for a given coarse tag.
        y_true[n] is equal to 1 if the tag is present in the sample.

    y_pred: array of bool, shape = [n_samples,]
        One-hot encoding of predicted presence for a given coarse tag.
        y_pred[n] is equal to 1 if the tag is present in the sample.


    Returns
    -------
    TP: int
        Number of true positives.

    FP: int
        Number of false positives.

    FN: int
        Number of false negatives.
    """
    cm = confusion_matrix(y_true, y_pred)
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    return TP, FP, FN


if __name__ == '__main__':
    #print(get_svm_results())
    #show_results("../../svm_prediction", feat_type="")
    # show_results("../../prediction2.0", feat_type="MFCCnorm_SVMrbf")
    # show_results("../../prediction2.0", feat_type="MFCCnorm_SVMpoly")
    #show_results("../../prediction2.0", feat_type="log")
    
    #show_results("../../predictions", feat_type="logfbank_long_logistic")
    #show_results("../../predictions", feat_type="logfbank_long_NMF_logisticReg")
    #show_results("../../predictions", feat_type="logfbank_long_PCA_logisticReg")
    #show_results("../../predictions", feat_type="logfbank_short_NMF_logisticReg")
    #show_results("../../predictions", feat_type="logfbank_short_PCA_logisticReg")
    
    #show_results("../../predictions", feat_type="mfcc_long_logisticReg")
    #show_results("../../predictions", feat_type="mfcc_long_NMF_logisticReg")
    #show_results("../../predictions", feat_type="mfcc_long_PCA_logisticReg")
    #show_results("../../predictions", feat_type="mfcc_short_NMF_logisticReg")
    #show_results("../../predictions", feat_type="mfcc_short_PCA_logisticReg")

    #show_results("../../predictions", feat_type="logfbank_long_decisionTree")
    #show_results("../../predictions", feat_type="logfbank_long_NMF_decisionTree")
    show_results("../../predictions", feat_type="logfbank_long_PCA_decisionTree")
    # show_results("../../predictions", feat_type="logfbank_short_NMF_decisionTree")
    #show_results("../../predictions", feat_type="logfbank_short_PCA_decisionTree")
    
    #show_results("../../predictions", feat_type="mfcc_long_decisionTree")
    #show_results("../../predictions", feat_type="mfcc_long_NMF_logisticReg")
    #show_results("../../predictions", feat_type="mfcc_long_PCA_logisticReg")
    #show_results("../../predictions", feat_type="mfcc_short_NMF_logisticReg")
    #show_results("../../predictions", feat_type="mfcc_short_PCA_logisticReg")

