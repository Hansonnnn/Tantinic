import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import SVC


def get_data_from_csv(file_name, sep=',', encoding='utf-8', **kwargs):
    if kwargs.get('usecols') is not None:
        usecols = kwargs.get('usecols')
        data = pd.read_csv(file_name, sep=sep, encoding=encoding, usecols=usecols)
        return data
    data = pd.read_csv(file_name, sep=sep, encoding=encoding)
    return data


def explore_data():
    train_df = get_data_from_csv('/Users/hanzhao/Downloads/train.csv')
    test_df = get_data_from_csv('/Users/hanzhao/Downloads/test.csv')
    combine = [train_df, test_df]
    # print(train_df.columns.values)
    # print(train_df.info())
    # print(train_df.describe())
    # g = sns.FacetGrid(train_df, col='Survived')
    # g.map(plt.hist, 'Age', bins=20)
    # plt.show()
    # grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
    # grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
    # grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    # grid.add_legend();
    # plt.show()
    # g = sns.FacetGrid(train_df, col='Survived',row='Sex',size=2.2)
    # g.map(plt.hist, 'Age', bins=20)
    # plt.show()

def train():
    input = get_input()
    # lr = LogisticRegression()
    svc = SVC()
    # lr.fit(input[0], input[1])
    svc_predict = svc.fit(input[0], input[1])
    # lr_predict = lr.predict(input[2])
    acc_log = round(svc.score(input[0], input[1]) * 100, 2)
    print(acc_log)


def get_input():
    after_analysize = analysize_data()
    x_train = after_analysize[1].drop('Survived', axis=1)
    y_train = after_analysize[1]['Survived']
    x_test = after_analysize[2].copy()
    return x_train, y_train, x_test


def analysize_data():
    train_df = get_data_from_csv('/Users/hanzhao/PycharmProjects/Titanic/dataset/train.csv')
    test_df = get_data_from_csv('/Users/hanzhao/PycharmProjects/Titanic/dataset/test.csv')
    after_select = select_feature(train_df, test_df)
    title_extract = extract_Title_feature(after_select[0], after_select[1], after_select[2])
    sex_convert = convert_sex_features(title_extract[0], title_extract[1], title_extract[2])
    sex_complete = complete_sex_features(sex_convert[0], sex_convert[1], sex_convert[2])
    embarked_complete = complete_embarked_feature(sex_complete[0], sex_complete[1], sex_complete[2])
    embarked_convert = convert_embarked_feature(embarked_complete[0], embarked_complete[1], embarked_complete[2])
    convert_fare = convert_fare_feature(embarked_convert[1], embarked_complete[2])
    end = convert_age_features(convert_fare[0], convert_fare[1], convert_fare[2])
    after_create = create_features(end[0], end[1], end[2])

    return after_create


"""@description  step 1:"""


def select_feature(train_df, test_df):
    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]

    return combine, train_df, test_df


"""@description step 2"""


def extract_Title_feature(combine, train_df, test_df):
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                                     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name', 'PassengerId'], axis=1)
    combine = [train_df, test_df]
    return combine, train_df, test_df


"""@description step 3:"""


def convert_sex_features(combine, train_df, test_df):
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
    return combine, train_df, test_df


def convert_age_features(combine, train_df, test_df):
    for dataset in combine:
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4
    return combine, train_df, test_df


def convert_embarked_feature(combine, train_df, test_df):
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    return combine, train_df, test_df


def convert_fare_feature(train_df, test_df):
    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
    combine = [train_df, test_df]
    for dataset in combine:
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    return combine, train_df, test_df


"""@description step 4:"""


def complete_sex_features(combine, train_df, test_df):
    guess_ages = np.zeros((2, 3))
    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) & \
                                   (dataset['Pclass'] == j + 1)]['Age'].dropna()

                age_guess = guess_df.median()

                guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

                dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                            'Age'] = guess_ages[i, j]
        dataset['Age'] = dataset['Age'].astype(int)
    return combine, train_df, test_df


def complete_embarked_feature(combine, train_df, test_df):
    freqport = train_df.Embarked.dropna().mode()[0]
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freqport)

    return combine, train_df, test_df


"""@description step 5:"""


def create_features(combine, train_df, test_df):
    for dataset in combine:
        dataset['FamilySize'] = dataset['Parch'] + dataset['SibSp'] + 1
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
        dataset['Age*Class'] = dataset.Age * dataset.Pclass
    train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    return combine, train_df, test_df


if __name__ == '__main__':
    train()
