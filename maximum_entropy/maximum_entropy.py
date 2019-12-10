import numpy as np
from math import exp
from math import log
from scipy.misc import logsumexp
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class MaximumEntropy:

    def __init__(self, learning_rate=0.001, epoch=500, batch_size=150):
        self.lr = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.W = None
        self.X_dict = None
        self.Y_dict = None

    def initialize_weights(self, X, y):
        feat_set = {feat for x in X for feat in x}
        self.X_dict = {feat: i for i, feat in enumerate(feat_set)}
        self.Y_dict = {y_: i for i, y_ in enumerate(set(y))}
        self.W = np.zeros((len(self.Y_dict), len(self.X_dict)))
        X = np.array([self.featurize(x) for x in X])
        return X, np.array(y)


    def featurize(self, x):
        feat_vec = np.zeros(len(self.X_dict))
        for feat in x:
            feat_vec[self.X_dict[feat]] = 1
        return feat_vec


    def shuffle_data(self, x, y):
        shuffled_index = np.random.permutation(len(x))
        x = x[shuffled_index]
        y = y[shuffled_index]
        return x, y

    def compute_loss(self, X, Y):
        """loss = negative log likelihood"""
        loss = -sum([log(self.posterior(x, y)) for x, y in zip(X, Y)])
        return loss

    def show_loss(self, X_train, X_val, Y_train, Y_val, epoch_num):
        train_loss = self.compute_loss(X_train, Y_train)
        val_loss = self.compute_loss(X_val, Y_val)

        print("Training loss at epoch {}: {}".format(epoch_num, train_loss))
        print("Validation loss at eopch {}: {}\n".format(epoch_num, val_loss))

    def predict(self, x):
        feat_vec = self.featurize(x)
        ret = [(np.dot(self.W[idx], feat_vec), y_) for y_, idx in self.Y_dict.items()]
        return max(ret)[1]

    def train(self, X_train, X_val, Y_train, Y_val):

        # construct a batch generator
        def batch_generator(X, Y):
            num_samples = len(X)
            for i in range(0, num_samples):
                yield X[i: min(i + self.batch_size, num_samples)], \
                      Y[i: min(i + self.batch_size, num_samples)]

        X_train, Y_train = self.initialize_weights(X_train, Y_train)
        X_val, Y_val = self.initialize_weights(X_val, Y_val)

        for i in range(1, self.epoch + 1):
            # Step 1: compute train and validation loss
            if i % 100 == 0 or i == 1:
                self.show_loss(X_train, X_val, Y_train, Y_val, i)
            X_train, Y_train = self.shuffle_data(X_train, Y_train)
            for X_batch, Y_batch in batch_generator(X_train, Y_train):
                self.update_weights(X_batch, Y_batch)

    def posterior(self, x, y):
        """Compute p(y|x) by softmax."""
        prob_y = exp(np.dot(self.W[self.Y_dict[y]], x))
        Z = [exp(np.dot(self.W[self.Y_dict[y]], x)) for y in self.Y_dict]
        return prob_y / sum(Z)

    def update_weights(self, X, Y):
        empirical_expectations = np.zeros(self.W.shape)
        model_expectations = np.zeros(self.W.shape)
        for i in range(len(X)):
            empirical_expectations[self.Y_dict[Y[i]]] += X[i]
            for y, idx in self.Y_dict.items():
                model_expectations[idx] += X[i] * self.posterior(X[i], y)
        self.W += self.lr * (empirical_expectations - model_expectations)

    def score(self, X, Y):
        pred_y = [self.predict(x) for x in X]
        print(classification_report(Y, pred_y))


def generate_data():
    dataset = [['no', 'sunny', 'hot', 'high', 'FALSE'],
               ['no', 'sunny', 'hot', 'high', 'TRUE'],
               ['yes', 'overcast', 'hot', 'high', 'FALSE'],
               ['yes', 'rainy', 'mild', 'high', 'FALSE'],
               ['yes', 'rainy', 'cool', 'normal', 'FALSE'],
               ['no', 'rainy', 'cool', 'normal', 'TRUE'],
               ['yes', 'overcast', 'cool', 'normal', 'TRUE'],
               ['no', 'sunny', 'mild', 'high', 'FALSE'],
               ['yes', 'sunny', 'cool', 'normal', 'FALSE'],
               ['yes', 'rainy', 'mild', 'normal', 'FALSE'],
               ['yes', 'sunny', 'mild', 'normal', 'TRUE'],
               ['yes', 'overcast', 'mild', 'high', 'TRUE'],
               ['yes', 'overcast', 'hot', 'normal', 'FALSE'],
               ['no', 'rainy', 'mild', 'high', 'TRUE']]
    X = []
    Y = []
    for data in dataset:
        X.append(data[1:])
        Y.append(data[0])
    return X, Y


if __name__ == '__main__':
    X, Y = generate_data()
    maxent = MaximumEntropy()
    maxent.train(X, X, Y, Y)
    maxent.score(X, Y)



