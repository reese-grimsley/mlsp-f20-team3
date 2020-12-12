#cell 0

import os
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.metrics
from sklearn.metrics import confusion_matrix

#classifiers
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import mixture
from sklearn.neighbors import KNeighborsClassifier



#cell 1
## Training/Predicition Functions

#cell 2
def get_data(path_name, feature="stc", audio_features='mfcc'):
    print('Read data from %s' % path_name)
    data = pd.read_csv(path_name, index_col = 0)

    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    train = data[data["split"]=="train"]

    data.iloc[:,2:], train.iloc[:,2:] = normalize_data(data.iloc[:,2:], train.iloc[:,2:])

    if feature =="stc":
        begin_mark = 2
    else:
        begin_mark = 15


    x_whole = data.iloc[:, begin_mark:]
    x_train = train.iloc[:, begin_mark:]
    y_train = train.iloc[:,1] #python begin with 0 and MATLAB with 1
    y_whole = data.iloc[:,1]
    
#     x_train = pd.DataFrame.to_numpy(x_train)
#     y_train = pd.DataFrame.to_numpy(y_train)
#     x_whole = pd.DataFrame.to_numpy(x_whole)
#     y_whole = pd.DataFrame.to_numpy(y_whole)
    return x_train, y_train, x_whole, y_whole, data

def normalize_data(data, train):
    #normalize all data according to the training data. Should be mean 0, variance 1 on the features
    arr = train.to_numpy()

    mean = np.mean(arr, axis=0)
    arr = arr - mean

    std = np.std(arr, axis=0)
    arr = arr/std

    normalized_train_data = pd.DataFrame(data=arr, index=train.index, columns=train.columns)

    arr = data.to_numpy()
    arr = arr - mean
    arr = arr/std
    normalized_data = pd.DataFrame(data=arr, index=data.index, columns=data.columns)

    return normalized_data, normalized_train_data
    
def save_pred(clf, data, x_data, path, name, method, feature="stc"):
    y_hat = clf.predict(x_data)
    tem = data.iloc[:,0:2]
    tem = pd.concat([tem, pd.DataFrame(columns=['prediction'])])
    tem["prediction"] = y_hat
    tem.to_csv(os.path.join(path, f'{feature}_{method}_pred_{name}'))

def train_model(data, labels, model_type, params={}):

    if model_type=='svm':
        return train_svm(data, labels, params)
    elif model_type=='rf':
        return train_rf(data, labels, params)
    elif model_type=='gmm':
        return train_gmm(data, labels, params)
    elif model_type=='knn':
        return train_knn(data, labels, params)
    else:    
        raise Exception('invalid model type %s ' % model_type)
    
def train_svm(data, labels, params):
    # SVM Classifier training, with parameters specified in params dictionary

 
    clf = svm.SVC(C=svm_params['C'], kernel=params['kernel'])
    clf.fit(x_train, y_train)

    return clf

def train_rf(data, labels, params):
    # Random Forest training, with parameters specified in params dictionary
    if not 'n_estimators' in params:
        params['n_estimators'] = 100
    if not 'criterion' in params:
        params['criterion'] = 'gini'
    if not 'min_samples_leaf' in params:
        params['min_samples_leaf'] = 1
    if not 'min_samples_split' in params:
        params['min_samples_split'] = 2
    if not 'class_weight' in params:
        # params['class_weight'] = [1, 1]
        params['class_weight'] = 'balanced'

    clf = RandomForestClassifier(n_estimators=params['n_estimators'], criterion=params['criterion'], min_samples_split=params['min_samples_split'], min_samples_leaf=params['min_samples_leaf'], class_weight=params['class_weight'])
    clf.fit(data, labels)

    return clf

def train_gmm(data, labels, params):

    if not 'covariance_type' in params:
        params['covariance_type'] = 'full'
    if not 'tol' in params:
        params['tol'] = 1e-3
    if not 'n_init' in params:
        params['n_init'] = 1


    clf = mixture.GaussianMixture(n_components=2, covariance_type=params['covariance_type'], tol=params['tol'])

    clf.fit(data, labels)

    return clf

    

def train_knn(data, labels, params):

    if not 'n_neighbors' in params:
        params['n_neighbors'] = 5
    if not 'weights' in params:
        params['weights'] = 'uniform'
    if not 'algorithm' in params:
        params['algorithm'] = 'auto'
    if not 'leaf_size' in params:
        params['leaf_size'] = 30

    clf = KNeighborsClassifier(n_neighbors=params['n_neighbors'], weights=params['weights'], algorithm=params['algorithm'], leaf_size=params['leaf_size'])

    clf.fit(data, labels)

    return clf

#cell 3
## Eval Functions

#cell 4


def eval(pred_path):
    pred_df = pd.read_csv(pred_path)
    # test samples only
    pred_df = pred_df[pred_df["split"]=="test"]
    # eg. 6_music_presence

    label_key = pred_df.columns[2]


    labels = pred_df[label_key]

    ratio = np.sum(labels) / len(labels)
    preds = pred_df["prediction"]
    
    TPs, FPs, FNs = confusion_matrix_coarse(labels, preds)
    auprc = sklearn.metrics.average_precision_score(labels, preds)
    return label_key, TPs, FPs, FNs, auprc, ratio


def get_svm_results(pred_dir="../../svm_prediction"):
    non_stc = {}
    with_stc = {}
    
    for filename in sorted(os.listdir(pred_dir)):
        if  'DS_Store' in filename:
            continue

        path = "{}/{}".format(pred_dir, filename)
        label_key, TPs, FPs, FNs, auprc, ratio = eval(path)
        dic = {
            "TPs" : TPs,
            "FPs" : FPs,
            "FNs" : FNs,
            "auprc" : auprc,
            "ratio" : ratio,
        }
        # only audio feature
        if "non_" in filename:
            non_stc[label_key] = dic
        # audio + stc feature
        elif "stc_" in filename:
            with_stc[label_key] = dic
    return non_stc, with_stc

def weighted_average(dicts):

    N = 0
    total_auprc = 0
    for d in dicts:
        Nd = d['TPs'] + d('FNs')
        total_auprc += d['auprc'] * Nd
        N += Nd

    total_auprc = total_auprc / N
    print(total_auprc)
    return total_auprc


def show_results(pred_dir="../../svm_prediction", num_class=8):
    non_stc, with_stc = get_svm_results(pred_dir)
    classes = sorted(list(non_stc.keys()))
    class_names = []

    for c in classes:
        print(c)
        print("Only  Audio ::",  non_stc[c])
        print("STC + Audio ::",  with_stc[c])
        print('-------------------------------------------------------------------------------')
        cc = c[2:-9]
        class_names.append(cc)

    # weighted_avg_non(non_stc)
    # weighted_avg_stc(with_stc)
    
    # plot bar graph
    non_auprc = [d["auprc"] for d in non_stc.values()]
    stc_auprc = [d["auprc"] for d in with_stc.values()]


    # non_list = [[i+1, non_auprc[i]] for i in range(len(classes))]
    # stc_list = [[i+1, stc_auprc[i]] for i in range(len(classes))]
    

    # x1,y1 = zip(*non_list)
    # x2,y2 = zip(*stc_list)
    # plt.bar(np.array(x1)-0.15, y1, width = 0.3)
    # plt.bar(np.array(x2)+0.15, y2, width = 0.3)
    # # plt.xticks(list(range(1, len(classes)+ 1)), classes, size='smaller', rotation=20)
    # plt.xticks(list(range(1, len(classes)+ 1)), class_names, size='smaller', rotation=20)
    # plt.title("AUPRC for Coarse Classes")
    # plt.xlabel("Coarse Classes")
    # plt.ylabel("AUPRC")
    # plt.legend(["Only Audio Features", "STC + Audio Features"])
    # plt.show()
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
    class_names.append("Weighted Average")
    x1,y1 = zip(*non_list)
    x2,y2 = zip(*stc_list)
    plt.bar(np.array(x1)-0.15, y1, width = 0.3)
    plt.bar(np.array(x2)+0.15, y2, width = 0.3)
    plt.xticks(list(range(1, len(class_names)+1)), class_names, size='smaller', rotation=20)
    plt.title("AURPC for Coarse Classes")
    plt.xlabel("Coarse Classes")
    plt.ylabel("AUPRC")
    plt.legend(["Only Audio Features", "STC + Audio Features"])
    plt.show()


    return stc_avg, non_avg

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
    # print(cm)
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    TN = cm[0, 0]

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP+TN+FN+FP)

    # print("Precision: %f\nRecall: %f\nAccuracy: %f" % (precision, recall, accuracy))
    # print("Average PR (guess): %f" % (precision * recall))
    # print('\n')
    return TP, FP, FN



#cell 5
# Training

#cell 6
use_mfcc_over_flogbank = False
use_long_window = True
use_PCA = False #takes precedence of NMF
use_NMF = True
model_type = 'rf' #'svm', 'rf', 'gmm', 'knn'

input_path ='./dataset/readied_data/'
output_path = 'dataset/predictions/'


if use_mfcc_over_flogbank:
    input_path = os.path.join(input_path, 'mfcc')
    output_path = os.path.join(output_path, 'mfcc')
    audio_features = 'mfcc'
else:
    input_path = os.path.join(input_path, 'logfbank')
    output_path = os.path.join(output_path, 'logfbank')
    audio_features = 'log_fbank'

if use_long_window:
    input_path += '_long'
else:
    input_path += '_short'

if use_PCA:
    input_path += '_PCA/'
elif use_NMF:
    input_path += '_NMF/'
else:
    input_path += '/'


params = {} #classifier dependent

if model_type == 'svm':
    model_name = 'RBF'
    svm_params = params.copy()
    svm_params['C'] = 1.0
    svm_params['kernel'] = 'rbf'
    # output_path = os.path.join(output_path, 'SVM')
    # params = svm_params


elif model_type == 'rf':
    model_name = 'random-forest'
    rf_params = params.copy()
    rf_params['min_samples_leaf'] = 5
    rf_params['min_samples_split'] = 11
    # rf_params['n_estimators'] = 200
    rf_params['criterion'] = 'entropy' #'gini' (default) or 'entropy'
    output_path = os.path.join(output_path, 'RF')
    params = rf_params


elif model_type == 'gmm':
    model_name = 'gmm'
    gmm_params = params.copy()
    gmm_params['covariance_type'] = 'spherical' #'full', 'tied', 'diag', 'spherical'
    gmm_params['tol'] = 1e-6
    gmm_params['n_init'] = 1
    # output_path = os.path.join(output_path, 'GMM')
    # params = gmm_params


elif model_type == 'knn':
    model_name = 'nearest-neighbor'
    knn_params = params.copy()
    knn_params['n_neighbors'] = 15
    knn_params['algorithm'] = 'brute' #'auto', 'ball_tree', 'kd_tree', 'brute'
    knn_params['weights'] = 'distance' #'uniform', 'distance'
    # knn_params['leaf_size'] = 5
    # output_path = os.path.join(output_path, 'KNN')
    # params = knn_params

else:
    raise Exception("What model type?")


print('start')

# rbf_svm, poly_svm = [], []
names = os.listdir(input_path)
print(names)

#clear output directory
for f in os.listdir(output_path):
    os.remove(os.path.join(output_path, f))

for i in tqdm.tqdm(range(0,len(names))):
    print()
# for i in range(len(names)):
    x_train, y_train, x_whole, y_whole, data = get_data(os.path.join(input_path, names[i]), audio_features=audio_features)
    # print(x_train.sum())
    model = train_model(x_train, y_train, model_type, params)
    save_pred(model, data, x_whole, output_path, names[i], model_name)

    
#     clf = svm.SVC(C=1.0,kernel='poly')
#     clf.fit(x_train, y_train)
#     save_pred(clf,x_whole, names[i], "POLY")
    
    x_train, y_train, x_whole, y_whole, data = get_data(os.path.join(input_path, names[i]), "non-stc", audio_features=audio_features)
    model = train_model(x_train, y_train, model_type, params)
    save_pred(model, data, x_whole, output_path, names[i], model_name, "non")

#cell 7
# Evaluation


#cell 8
def grid_search(model_type):
    cols = ['mfcc_over_flogbank', 'long_window', 'data-driven', 'AUPRC_stc', 'AUPRC_non-stc']

    results = pd.DataFrame(columns = cols)

    use_mfcc_over_flogbank = False
    use_long_window = True
    # model_type = 'rf' #'svm', 'rf', 'gmm', 'knn'

    tf = [True, False]
    data_types = [None, 'PCA', 'NMF']

    for feature_type in tf:
        for window in tf:
            for dt in data_types:
                AUPRC_stc, AUPRC_non = train_and_test_model(model_type=model_type, use_mfcc_over_flogbank=feature_type, use_long_window=window, data_type=dt)
                new_result = pd.DataFrame(data=[[feature_type, window, dt, AUPRC_stc, AUPRC_non]], columns = cols)
                print(new_result)
                results = results.append(new_result)


    print('Finished grid-search')
    return results
                


def train_and_test_model(model_type='rf', use_mfcc_over_flogbank=False, use_long_window=False, data_type=None):

    input_path ='./dataset/readied_data/'
    output_path = './dataset/predictions/'

    if use_mfcc_over_flogbank:
        input_path = os.path.join(input_path, 'mfcc')
        output_path = os.path.join(output_path, 'mfcc')
        audio_features = 'mfcc'
    else:
        input_path = os.path.join(input_path, 'logfbank')
        output_path = os.path.join(output_path, 'logfbank')
        audio_features = 'log_fbank'

    if use_long_window:
        input_path += '_long'
    else:
        input_path += '_short'

    if data_type == 'PCA':
        input_path += '_PCA/'
    elif data_type == 'NMF':
        input_path += '_NMF/'
    else:
        input_path += '/'


    params = {} #classifier dependent

    if model_type == 'svm':
        model_name = 'RBF'
        svm_params = params.copy()
        svm_params['C'] = 1.0
        svm_params['kernel'] = 'rbf'
        output_path = os.path.join(output_path, 'SVM')
        params = svm_params

    elif model_type == 'rf':
        model_name = 'random-forest'
        rf_params = params.copy()
        rf_params['min_samples_leaf'] = 5
        rf_params['min_samples_split'] = 11
        # rf_params['n_estimators'] = 200
        rf_params['criterion'] = 'entropy' #'gini' (default) or 'entropy'
        output_path = os.path.join(output_path, 'RF')
        params = rf_params

    elif model_type == 'gmm':
        model_name = 'gmm'
        gmm_params = params.copy()
        gmm_params['covariance_type'] = 'spherical' #'full', 'tied', 'diag', 'spherical'
        gmm_params['tol'] = 1e-6
        # gmm_params['n_init'] = 1
        output_path = os.path.join(output_path, 'GMM')
        params = gmm_params

    elif model_type == 'knn':
        model_name = 'nearest-neighbor'
        knn_params = params.copy()
        knn_params['n_neighbors'] = 15
        knn_params['algorithm'] = 'auto' #'auto', 'ball_tree', 'kd_tree', 'brute'
        knn_params['weights'] = 'distance' #'uniform', 'distance'
        # knn_params['leaf_size'] = 5
        output_path = os.path.join(output_path, 'KNN')
        params = knn_params

    else:
        raise Exception("What model type?")

    print('start')
    names = os.listdir(input_path)

    #clear output directory
    for f in os.listdir(output_path):
        os.remove(os.path.join(output_path, f))

    for i in tqdm.tqdm(range(0,len(names))):
        print()
    # for i in range(len(names)):
        x_train, y_train, x_whole, y_whole, data = get_data(os.path.join(input_path, names[i]), audio_features=audio_features)
        # print(x_train.sum())
        print(x_whole.shape)
        model = train_model(x_train, y_train, model_type, params)
        save_pred(model, data, x_whole, output_path, names[i], model_name)

        
    #     clf = svm.SVC(C=1.0,kernel='poly')
    #     clf.fit(x_train, y_train)
    #     save_pred(clf,x_whole, names[i], "POLY")
        
        x_train, y_train, x_whole, y_whole, data = get_data(os.path.join(input_path, names[i]), "non-stc", audio_features=audio_features)
        model = train_model(x_train, y_train, model_type, params)
        save_pred(model, data, x_whole, output_path, names[i], model_name, "non")


    avg_stc, avg_non_stc = show_results(output_path)
    print(input_path)
    print(params)
    return avg_stc, avg_non_stc

#cell 9
model_type='rf'
res = grid_search(model_type)

#cell 10
print(res)
res.to_csv('./results_'+model_type+'.csv')

#cell 11


