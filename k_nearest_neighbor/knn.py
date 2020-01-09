import numpy as np
from math import sqrt
from collections import Counter
from sklearn.metrics import classification_report

np.random.seed(42)

class KNN:
    def __init__(self, k=5):
        self.k = k
        self.data = None
        self.labels = None
    
    def load_data(self, X, Y):
        self.data = X
        self.labels = Y
    
    def distance(self, p1, p2):
        return sqrt(np.sum(np.power(p1 - p2, 2)))

    def vote(self, Y):
        Y_dict = Counter(Y)
        return Y_dict.most_common()[0][0]

    def predict(self, x):
        dists = [self.distance(x, point) for point in self.data]
        # get the index list which is sorted by values on according positions
        ids = np.argsort(dists)
        top_k_ids = ids[: self.k]
        top_k_labels = self.labels[top_k_ids]
        return self.vote(top_k_labels)
    
    def score(self, X, Y):
        Y_pred = [self.predict(x) for x in X]
        print(classification_report(Y, Y_pred))

def generate_digit_data():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    digits = load_digits()
    X = digits.data
    Y = digits.target
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y)
    print('Train data shape', X_train.shape)
    print('Validation data shape', X_val.shape)
    return X_train, X_val, Y_train, Y_val

if __name__ == '__main__':
    X_train, X_val, Y_train, Y_val = generate_digit_data()
    knn = KNN()
    knn.load_data(X_train, Y_train)
    knn.score(X_val, Y_val)