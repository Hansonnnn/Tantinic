from speedml import Speedml


def input_data():
    sml = Speedml('/Users/hanzhao/PycharmProjects/Titanic/dataset/train.csv', '/Users/hanzhao/PycharmProjects/Titanic/dataset/test.csv', target='Survived',
                  uid='PassengerId')

    print(sml.shape())
    print(sml.train.describe())
    sml.plot.distribute()



input_data()
