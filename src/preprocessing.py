import numpy as np
import pandas as pd

from src.features import *


def get_data():
    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv')
    return train_data, test_data


def main():
    train_data, test_data = get_data()
    train_data = impute_all_features(train_data)
    test_data = impute_all_features(test_data)

    print(train_data.head())
    print(test_data.head())


if __name__ == '__main__':
    main()
