from Document import Document
from math import exp
from scipy.misc import logsumexp
from random import shuffle
import numpy as np


class MaxEnt():

    def initialize(self, train, dev):
        '''Collect all features and labels, convert them to numbers by dicts.'''
        print("Initializing...")
        l, v = set(), set()  # l: label set; v: vocabulary set
        for inst in train:
            l.add(inst.label)
            v = v.union(inst.features)
        self.L = {label: i for i, label in
                  enumerate(l)}  # a dict to record label and its id
        self.V = {word: i for i, word in
                  enumerate(v)}  # a dict to record vocab and its id
        self.P = np.zeros((len(l), len(v)))  # the parameter matrix

        for inst in dev + train:  # featurization
            inst.feature_vector = self.get_vec(inst.features)

    def get_vec(self, features):
        '''Make features a vector.'''
        vec = np.zeros(len(self.V))
        for v in features:
            try:
                vec[self.V[v]] = 1
            except:
                pass  # ignore out-of-vocabulary words
        return vec

    def train(self, train, dev=None):
        """Construct a statistical model from labeled instances.
           Learning rate: 0.0001
           Batch size: 30
        """
        self.initialize(train, dev)
        self.train_sgd(train, dev, 0.0001, 30)

    def train_sgd(self, train, dev, learning_rate, batch_size):
        """Train MaxEnt model with Mini-batch Stochastic Gradient."""
        start, count = 0, 1  # start: for batch slicing; count: to record iteration time
        self.max, self.max_count = 0, 0  # max: the maximum accuracy; max_count: how many next tests don't win max
        print(("[Iteration {}]").format(count))
        while True:
            self.addGradient(train[start: start + batch_size],
                             learning_rate)
            start += batch_size
            if start >= len(train):
                if not self.converge(dev):
                    count += 1
                    start = 0
                    shuffle(train)
                    print(("[Iteration {}]").format(count))
                else:
                    break

    def acc(self, dev):
        correct = sum([self.classify(inst) == inst.label for inst in dev])
        return correct * 100 / len(dev)

    def converge(self, dev):
        acc = self.acc(dev)
        print(("Accuracy: {}%").format(acc))
        if acc > self.max:
            self.max = acc
            self.max_count = 0
        else:
            self.max_count += 1
        if self.max_count == 10:  # if all next 10 tests don't win the max, converge
            print("Training converged.")
            return True
        return False

    def posterior(self, label, feature_vector):
        '''Get the posterior value by the given label and instance, which will be used in the addGradient function.'''
        numerator = np.dot(self.P[self.L[label]], feature_vector)
        denominator = [np.dot(self.P[self.L[l]], feature_vector) for l in
                       self.L]
        return exp(numerator - logsumexp(denominator))

    def addGradient(self, instances, lr):
        '''Compute the observation matrix and expected matrix, perform a subtraction to get the value that the Parameter matrix should decrese.'''
        ob = np.zeros((len(self.L), len(self.V)))
        ex = np.zeros(ob.shape)
        for inst in instances:
            ob[self.L[
                inst.label]] += inst.feature_vector  # this is the observation value
            for l in self.L:  # this is the model value
                ex[self.L[l]] += inst.feature_vector * self.posterior(l,
                                                                      inst.feature_vector)
        self.P += lr * (ob - ex)

    def classify(self,
                 instance):  # compare with numerator without exp is already enough
        instance.feature_vector = self.get_vec(instance.features)
        return max(
            [(np.dot(self.P[self.L[l]], instance.feature_vector), l) for l
             in self.L])[1]