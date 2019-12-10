import numpy as np
from math import exp
from scipy.misc import logsumexp


class MaximumEntropy:

    def __init__(self, learning_rate=0.001, epoch=300, batch_size=150):
        self.lr = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.W = None
        self.Y_dict = None

    def initialize_weights(self, x, y):
        self.Y_dict = {y_: i for i, y_ in enumerate(set(y))}
        self.W = np.zeros((len(self.Y_dict), x.shape[1]))

    def shuffle_data(self, x, y):
        shuffled_index = np.random.permutation(len(x))
        x = x[shuffled_index]
        y = y[shuffled_index]
        return x, y

    def compute_loss(self, x, y):
        pass

    def predict(self, x):


    def train(self, x_train, x_val, y_train, y_val):

        # construct a batch generator
        def batch_generator(x, y):
            num_samples = len(x)
            for i in range(0, num_samples):
                yield x[i: min(i + self.batch_size, num_samples)], \
                      y[i: min(i + self.batch_size, num_samples)]

        self.initialize_weights(x_train, y_train)

        for i in range(self.epoch):
            x_train, y_train = self.shuffle_data(x_train, y_train)
            for x_batch, y_batch in batch_generator(x_train, y_train):
                self.update_weights(x_batch, y_batch)

    def posterior(self, y_, features):
        "Softmax."
        prob_y_ = np.dot(self.W[self.Y_dict[y_]], features)
        Z = [np.dot(self.W[self.Y_dict[y_]], features) for y_ in self.Y_dict]
        return exp(prob_y_ - logsumexp(Z))

    def update_weights(self, x, y):
        empirical_expectations = np.zeros(self.W.shape)
        model_expectations = np.zeros(self.W.shape)
        for i in range(len(x)):
            empirical_expectations[self.Y_dict[y[i]]] += x.features
            for y_, idx in self.Y_dict:
                model_expectations[idx] += x.features * self.posterior(y_, x.features)
        self.W += self.lr * (empirical_expectations - model_expectations)