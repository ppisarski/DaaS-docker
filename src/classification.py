import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.preprocessing import *
from src.features import *


def main():
    train_data, test_data = get_data()
    train_data = impute_all_features(train_data)
    test_data = impute_all_features(test_data)
    train_data.drop(labels='PassengerId', axis=1, inplace=True)
    test_data.drop(labels='PassengerId', axis=1, inplace=True)

    X = train_data.iloc[:, 1:12].values
    y = train_data.iloc[:, 0].values

    label_encoder_sex_tr = LabelEncoder()
    label_encoder_title_tr = LabelEncoder()
    label_encoder_embarked_tr = LabelEncoder()
    X[:, 1] = label_encoder_sex_tr.fit_transform(X[:, 1])
    X[:, 5] = label_encoder_title_tr.fit_transform(X[:, 5])
    X[:, 4] = label_encoder_embarked_tr.fit_transform(X[:, 4])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler_x = MinMaxScaler((-1, 1))
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)

    accuracies = []

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    lr_score = classifier.score(X_test, y_test)
    accuracies.append(lr_score)

    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    svm_score = svm.score(X_test, y_test)
    accuracies.append(svm_score)

    k_svm = SVC(kernel='rbf')
    k_svm.fit(X_train, y_train)
    k_svm_score = k_svm.score(X_test, y_test)
    accuracies.append(k_svm_score)

    knn = KNeighborsClassifier(p=2, n_neighbors=10)
    knn.fit(X_train, y_train)
    knn_score = knn.score(X_test, y_test)
    accuracies.append(knn_score)

    rf = RandomForestClassifier(n_estimators=20, criterion='entropy')
    rf.fit(X_train, y_train)
    rf_score = rf.score(X_test, y_test)
    accuracies.append(rf_score)

    # xgb = XGBClassifier()
    # xgb.fit(X_train, y_train)
    # xgb_score = xgb.score(X_test, y_test)
    # accuracies.append(xgb_score)

    with PdfPages('../results/classifiers.pdf') as pdf:
        # myLabels = ['Logistic Regression', 'SVM', 'Kernel SVM', 'KNN', 'Random Forest', 'Xgboost']
        labels = ['Logistic Regression', 'SVM', 'Kernel SVM', 'KNN', 'Random Forest']
        fig1_accu = sns.barplot(x=accuracies, y=labels)
        plt.tight_layout()
        pdf.savefig()
        plt.close()


if __name__ == '__main__':
    main()
