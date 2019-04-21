import numpy as np
from sklearn.model_selection import KFold


class SklearnHelper(object):
    def __init__(self, clf, seed=0, x_train=None, x_test=None, y_train=None, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)
        self.ntrain = x_train.shape[0]
        self.ntest = x_test.shape[0]
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.SEED = seed  # for reproducibility
        self.NFOLDS = 5  # set folds for out-of-fold prediction
        self.kf = KFold(n_splits=self.NFOLDS, random_state=self.SEED, shuffle=False)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)

    def get_oof(self, clf):
        oof_train = np.zeros((self.ntrain,))
        oof_test = np.zeros((self.ntest,))
        oof_test_skf = np.empty((self.NFOLDS, self.ntest))

        for i, (train_index, test_index) in enumerate(self.kf.split(self.x_train)):
            x_tr = self.x_train[train_index]
            y_tr = self.y_train[train_index]
            x_te = self.x_train[test_index]

            clf.train(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(self.x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
