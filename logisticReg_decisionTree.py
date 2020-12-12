import os
import pandas as pd
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


def get_data(path, feature, extra_feature):
    data = pd.read_csv(path, index_col = 0)
    train = data[data["split"]=="train"]
    
    if extra_feature =="stc":
        begin_mark = "DOB"
    else:
        begin_mark = feature + '1'
    x_whole = data.loc[:, begin_mark:]
    x_train = train.loc[:, begin_mark:]
    y_train = train.iloc[:,1] 
    y_whole = data.iloc[:,1]
    return x_train, y_train, x_whole, y_whole, data


def save_pred(clf, data, x_data, name, tag):
    y_hat = clf.predict(x_data)
    tem = data.iloc[:,0:2]
    tem = pd.concat([tem, pd.DataFrame(columns=['prediction'])])
    tem["prediction"] = y_hat
    tem.to_csv("predictions/{}_class{}.csv".format(tag, name[0]))


if __name__ == "__main__":
    # method = "logisticReg"
    method = "decisionTree"
    # method = "dnaiveBayes"
    # feature = "logfbank_long"
    # feature = "logfbank_long_NMF"
    feature = "logfbank_long_PCA"
    # feature = "logfbank_short_NMF"
    # feature = "logfbank_short_PCA"

    # feature = "mfcc_long"
    # feature = "mfcc_long_NMF"
    # feature = "mfcc_long_PCA"
    # feature = "mfcc_short_NMF"
    # feature = "mfcc_short_PCA"

    path = "features/{}/".format(feature)
    print(path)
    names = os.listdir(path)
    for i in range(len(names)):#
        x_train, y_train, x_whole, y_whole, data = get_data(path+names[i],feature, "stc")
        # clf = LogisticRegression(max_iter=1000)
        clf = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 1000, max_depth = 6, min_samples_leaf = 2)
        #clf = make_pipeline(TfidfVectorizer(), MultinomialNB())
        clf.fit(x_train, y_train)
        save_pred(clf, data, x_whole, names[i], "{}_{}_stc".format(feature, method))

        x_train, y_train, x_whole, y_whole, data = get_data(path+names[i],feature, "non-stc")
        # clf = LogisticRegression(max_iter=1000)
        clf = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 1000, 
                max_depth = 6, min_samples_leaf = 2)
        Eclf = make_pipeline(TfidfVectorizer(), MultinomialNB())
        clf.fit(x_train, y_train)
        save_pred(clf,data, x_whole, names[i], "{}_{}_non".format(feature, method))
