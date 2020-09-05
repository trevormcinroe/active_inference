"""

"""
import numpy as np
from scipy.stats import norm
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB


class NaiveBayes:
    """"""

    def __init__(self, X, y):
        self.X = X
        self.y = y

        # self.num_classes = len(np.unique(self.y))
        self.classes = np.unique(self.y)
        self.N = len(self.y)

        self.class_cond_priors = dict()
        self.priors = dict()

        self._get_priors()
        self._get_class_conditional_priors()

    def _get_priors(self):
        """"""
        for c in self.classes:
            self.priors[c] = np.sum(self.y == c) / self.N

        # self.priors = np.array([np.sum(self.y == c) / self.N for c in self.classes])

    def _get_class_conditional_priors(self):
        """"""
        for c in self.classes:
            self.class_cond_priors[c] = list()
            subset = self.X[self.y == c]
            # Need to loop through each column vector in X
            for i in range(self.X.shape[1]):
                self.class_cond_priors[c].append(norm(np.mean(subset[:, i]), np.std(subset[:, i])))

    def predict_proba(self, X):
        """"""
        results = dict()

        for c in self.classes:
            p = self.priors[c]
            for j in range(len(X)):
                p *= self.class_cond_priors[c][j].pdf(X[j])

            results[c] = p * 100

        return results

    def predict(self, X):
        results = self.predict_proba(X=X)
        return np.argmax([v for k, v in results.items()])


iris = datasets.load_iris()
X = iris['data']
y = iris['target']

tm = NaiveBayes(X=X, y=y)

gnb = GaussianNB()
gnb.fit(X=X, y=y)

tm_preds = list()
gnb_preds = list()
for row in X:
    tm_preds.append(tm.predict(X=row))
    gnb_preds.append(gnb.predict(X=[row])[0])

print(tm_preds)
print(gnb_preds)

print(tm_preds == gnb_preds)
