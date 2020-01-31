import numpy as np
from sklearn.metrics import classification_report

class BaseClassifier:
    def __init__(self):
        pass

    def predict(self, x):
        return 1

class AdaBoost:
    def __init__(self, m=50):
        self.num_clf = m
        self.clfs = []
        self.alphas = np.zeros(m)
    
    def error(self, X, Y, w, clf):
        Y_pred = []
        wrong = []
        for x, y in zip(X, Y):
            y_pred = clf(x)
            Y_pred.append(y_pred)
            if y_pred != y:
                wrong.append(1)
        error = np.dot(wrong, w)
        return error, Y_pred
    
    def alpha(self, error):
        return 0.5 * np.log((1 - error) / error)
    
    def Z(self, w, alpha, Y, Y_pred):
        return np.sum(w * np.exp(-alpha * Y * Y_pred))
    
    def update_w(self, w, alpha, Y, Y_pred, Z):
        return (w * np.exp(-alpha * Y * Y_pred)) / Z

    def train(self, X, Y):
        w = np.full(len(X), 1 / len(X))
        for i in range(self.num_clf):
            clf = BaseClassifier()
            error, Y_pred = self.error(X, Y, w, clf)
            alpha = self.alpha(error)

            self.clfs.append(clf)
            self.alphas[i] = alpha

            Z = self.Z(w, alpha, Y, Y_pred)
            w = self.update_w(w, alpha, Y, Y_pred, Z)
    
    def predict(self, x):
        clf_pred = np.array([clf(x) for clf in self.clfs])
        result = np.dot(self.alphas, clf_pred)
        return 1 if result > 0 else -1
    
    def score(self, X, Y):
        Y_pred = [self.predict(x) for x in X]
        print(classification_report(Y, Y_pred))



if __name__ == '__main__':
    n = np.full(10, 1)
    print(type(n))
    print(1/3)
    m = np.ones(3)
    m[2] = 0
    print(m)



