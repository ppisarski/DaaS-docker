import numpy as np
import pandas as pd


def impute_title(data):
    data['Title'] = data['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
    data['Name_Len'] = data['Name'].apply(lambda x: len(x))
    data.drop(labels='Name', axis=1, inplace=True)
    return data


def impute_age(data):
    age_mean = data.Age.mean()
    age_std = data.Age.std()
    age_null = data.Age.isnull().sum()
    rand_age = np.random.randint(age_mean - age_std, age_mean + age_std, size=age_null)
    data['Age'][np.isnan(data['Age'])] = rand_age
    data['Age'] = data['Age'].astype(int) + 1
    return data


def impute_family(data):
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['isAlone'] = data['FamilySize'].map(lambda x: 1 if x == 1 else 0)
    data.drop(labels=['SibSp', 'Parch'], axis=1, inplace=True)
    return data


def impute_ticket(data):
    data['Ticket_Len'] = data['Ticket'].apply(lambda x: len(x))
    data.drop(labels='Ticket', axis=1, inplace=True)
    return data


def impute_fare(data):
    data['Fare'][np.isnan(data['Fare'])] = data.Fare.mean()
    return data


def impute_cabin(data):
    # Making a new feature hasCabin which is 1 if cabin is available else 0
    data['hasCabin'] = data.Cabin.notnull().astype(int)
    data.drop(labels='Cabin', axis=1, inplace=True)
    return data


def impute_embarked(data):
    data['Embarked'] = data['Embarked'].fillna('S')
    return data


def impute_all_features(data):
    data = impute_title(data)
    data = impute_age(data)
    data = impute_family(data)
    data = impute_ticket(data)
    data = impute_fare(data)
    data = impute_cabin(data)
    data = impute_embarked(data)
    return data
