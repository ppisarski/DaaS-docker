import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from src.preprocessing import *
from src.features import *
from src.classification import *


def main():
    train_data, test_data = get_data()
    train_data = impute_all_features(train_data)
    test_data = impute_all_features(test_data)
    test_data['Title'] = test_data['Title'].replace('Dona.', 'Mrs.')
    passenger_id = test_data['PassengerId']
    train_data.drop(labels='PassengerId', axis=1, inplace=True)
    test_data.drop(labels='PassengerId', axis=1, inplace=True)

    X_train = train_data.iloc[:, 1:12].values
    y_train = train_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 0:11].values

    label_encoder_sex = LabelEncoder()
    label_encoder_title = LabelEncoder()
    label_encoder_embarked = LabelEncoder()
    X_train[:, 1] = label_encoder_sex.fit_transform(X_train[:, 1])
    X_train[:, 5] = label_encoder_title.fit_transform(X_train[:, 5])
    X_train[:, 4] = label_encoder_embarked.fit_transform(X_train[:, 4])
    X_test[:, 1] = label_encoder_sex.transform(X_test[:, 1])
    X_test[:, 5] = label_encoder_title.transform(X_test[:, 5])
    X_test[:, 4] = label_encoder_embarked.transform(X_test[:, 4])

    scaler_x = MinMaxScaler((-1, 1))
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)

    param_grid = {"criterion": ["gini", "entropy"],
                  "min_samples_leaf": [1, 5, 10],
                  "min_samples_split": [2, 4, 10, 12],
                  "n_estimators": [50, 100, 400, 700]}

    rf = RandomForestClassifier(max_features='auto')
    gs = GridSearchCV(estimator=rf, param_grid=param_grid,
                      scoring='accuracy', cv=3, n_jobs=-1)
    gs = gs.fit(X_train, y_train)
    print(gs.best_score_)
    print(gs.best_params_)

    rf = RandomForestClassifier(max_features='auto', **gs.best_params_)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    titanic_pred = pd.DataFrame({'PassengerId': passenger_id, 'Survived': y_pred})
    titanic_pred.to_csv('../results/pred_titanic.csv', index=False)


if __name__ == '__main__':
    main()
