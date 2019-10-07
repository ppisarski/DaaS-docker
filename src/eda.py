import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.preprocessing import *
from src.features import *


def pclass(data):
    print('Pclass survival')
    print(data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())

    fx, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title("Pclass vs Frequency")
    axes[1].set_title("Pclass vise Survival rate")
    fig1_pclass = sns.countplot(data=data, x='Pclass', ax=axes[0])
    fig2_pclass = sns.barplot(data=data, x='Pclass', y='Survived', ax=axes[1])


def name(data):
    data = impute_title(data)
    print('Name survival')
    print(data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
    print(data[['Name_Len', 'Survived']].groupby(['Name_Len'], as_index=False).mean())

    fx, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].set_title("Title vs Frequency")
    axes[1].set_title("Title vise Survival rate")
    fig1_title = sns.countplot(data=data, x='Title', ax=axes[0])
    fig2_title = sns.barplot(data=data, x='Title', y='Survived', ax=axes[1])


def gender(data):
    print('Gender survival')
    print(data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())

    fx, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title("Gender vs Frequency")
    axes[1].set_title("Gender vise Survival rate")
    fig1_gen = sns.countplot(data=data, x='Sex', ax=axes[0])
    fig2_gen = sns.barplot(data=data, x='Sex', y='Survived', ax=axes[1])


def age(data):
    data = impute_age(data)
    print('Age survival')
    data_age = data.Age.dropna(axis=0)

    fx, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title("Age vs frequency")
    axes[1].set_title("Age vise Survival rate")
    fig1_age = sns.distplot(a=data_age, bins=15, ax=axes[0], hist_kws={'rwidth': 0.7})

    pass_survived_age = data[data.Survived == 1].Age
    pass_dead_age = data[data.Survived == 0].Age

    axes[1].hist([data.Age, pass_survived_age, pass_dead_age], bins=5, range=(0, 100),
                 label=['Total', 'Survived', 'Dead'])
    axes[1].legend()


def family(data):
    data = impute_family(data)
    print('Family survival')
    fx, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    axes[0].set_title('Family Size counts')
    axes[1].set_title('Survival Rate vs Family Size')
    fig1_family = sns.countplot(x=data.FamilySize, ax=axes[0], palette='cool')
    fig2_family = sns.barplot(x=data.FamilySize, y=data.Survived, ax=axes[1], palette='cool')

    print('Alone survival')
    fig1_alone = sns.countplot(data=data, x='isAlone', ax=axes[2])
    fig2_alone = sns.barplot(data=data, x='isAlone', y='Survived', ax=axes[3])


def ticket(data):
    data = impute_ticket(data)
    print('Ticket survival')
    print(data[['Ticket_Len', 'Survived']].groupby(data['Ticket_Len'], as_index=False).mean())
    fx, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].set_title("Ticket Length vs Frequency")
    axes[1].set_title("Length vise Survival rate")
    fig1_tlen = sns.countplot(data=data, x='Ticket_Len', ax=axes[0])
    fig2_tlen = sns.barplot(data=data, x='Ticket_Len', y='Survived', ax=axes[1])


def fare(data):
    data = impute_fare(data)
    print('Fare survival')
    fx, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig1_fare = sns.distplot(a=data.Fare, bins=15, ax=axes[0], hist_kws={'rwidth': 0.7})
    fig1_fare.set_title('Fare vise Frequency')

    # Creating a new list of survived and dead

    pass_survived_fare = data[data.Survived == 1].Fare
    pass_dead_fare = data[data.Survived == 0].Fare

    axes[1].hist(x=[data.Fare, pass_survived_fare, pass_dead_fare], bins=5,
                 label=['Total', 'Survived', 'Dead'], log=True)
    axes[1].legend()
    axes[1].set_title('Fare vise Survival')


def cabin(data):
    data['hasCabin'] = data.Cabin.notnull().astype(int)
    fx, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig1_hascabin = sns.countplot(data=data, x='hasCabin', ax=axes[0])
    fig2_hascabin = sns.barplot(data=data, x='hasCabin', y='Survived', ax=axes[1])


def embarked(data):
    data = impute_embarked(data)
    print(data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
    print(data[['Embarked', 'Fare']].groupby(['Embarked'], as_index=False).mean())
    fx, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title('Embarked Counts')
    axes[1].set_title('Survival Rate vs Embarked')
    fig1_embarked = sns.countplot(x=data.Embarked, ax=axes[0])
    fig2_embarked = sns.barplot(x=data.Embarked, y=data.Survived, ax=axes[1])


def main():
    train_data, test_data = get_data()
    print("\n\nMissing values in train data:\n", train_data.isnull().sum())
    print("\n\nMissing values in test data:\n", test_data.isnull().sum())

    with PdfPages('../figs/eda.pdf') as pdf:
        pclass(train_data)
        pdf.savefig()
        plt.close()

        name(train_data)
        pdf.savefig()
        plt.close()

        gender(train_data)
        pdf.savefig()
        plt.close()

        age(train_data)
        pdf.savefig()
        plt.close()

        family(train_data)
        pdf.savefig()
        plt.close()

        ticket(train_data)
        pdf.savefig()
        plt.close()

        fare(train_data)
        pdf.savefig()
        plt.close()

        cabin(train_data)
        pdf.savefig()
        plt.close()

        embarked(train_data)
        pdf.savefig()
        plt.close()


if __name__ == '__main__':
    main()
