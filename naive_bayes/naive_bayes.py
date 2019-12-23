import numpy as np
import math
from collections import defaultdict
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

np.random.seed(42)

class NaiveBayesClassifier:
    def __init__(self, laplace=1, mode='Multinomial'):
        self.laplace = laplace
        self.mode = mode

        self.X_dict = {}
        self.Y_dict = {}
        self.Prior = None
        self.Likelihood = None

        self.Mean = None
        self.Stdev = None
    
    def load_data(self, X, Y):
        if self.mode == 'Multinomail' or self.mode == 'Bernoulli':
            feat_set = {feat for x in X for feat in x}
            self.X_dict = {feat: i for i, feat in enumerate(feat_set)}
            self.Y_dict = {y: i for i, y in enumerate(set(Y))}
            self.Prior = np.zeros(len(self.Y_dict))
            self.Likelihood = np.zeros((len(self.X_dict), len(self.Y_dict)))
        
        elif self.mode == 'Gaussian':
            self.Y_dict = {y: i for i, y in enumerate(set(Y))}
            self.Mean = np.zeros((len(X[0]), len(set(Y))))
            self.Stdev = np.zeros((len(X[0]), len(set(Y))))
    
    def remake_data(self, X, Y):
        data = defaultdict(list)
        for x, y in zip(X, Y):
            data[y].append(x)
        return data

    def gaussian_prob(self, x):
        probs = []
        for y, i in self.Y_dict.items():
            exponent = np.exp(-(np.power(x - self.Mean[:, i], 2) /
                              (2 * np.power(self.Stdev[:, i], 2))))
            prob = (1 / (math.sqrt(2 * math.pi) * self.Stdev[:, i])) * exponent
            probs.append((sum(np.log(prob)), y))
        return probs
    
    def train(self, X, Y):
        if self.mode == 'Multinomial':
            self.train_Multinomial(X, Y)
        elif self.mode == 'Bernoulli':
            self.train_Bernoulli(X, Y)
        elif self.mode == 'Gaussian':
            self.train_Gaussian(X, Y)
    
    def train_Gaussian(self, X, Y):
        # Step 1: init weights
        self.load_data(X, Y)

        # Step 2: compute mean and stdev for each feature dimension of each label
        data = self.remake_data(X, Y)

        for y, i in self.Y_dict.items():
            for x_dim, x_feats in enumerate(zip(*data[y])):
                self.Mean[x_dim, i] = sum(x_feats) / len(x_feats)
                self.Stdev[x_dim, i] = math.sqrt(sum(np.power(x_feats - self.Mean[x_dim, i], 2)) / len(x_feats))

    def train_Multinomial(self, X, Y):
        # Step 1: init weights
        self.load_data(X, Y)

        # Step 2: count occurrences
        for x, y in zip(X, Y):
            self.Prior[self.Y_dict[y]] += len(x)
            for feat in x:
                self.Likelihood[self.X_dict[feat]][self.Y_dict[y]] += 1
        
        # Step 3: compute probabilities
        self.Likelihood = (self.Likelihood + self.laplace) / (self.Prior + self.laplace * len(self.X_dict))
        
        self.Prior = self.Prior / sum(self.Prior)
    
    def train_Bernoulli(self, X, Y):
        # Step 1: init weights
        self.load_data(X, Y)

        # Step 2: count occurrences
        for x, y in zip(X, Y):
            self.Prior[self.Y_dict[y]] += 1
            for feat in set(x):
                self.Likelihood[self.X_dict[feat]][self.Y_dict[y]] += 1
        
        # Step 3: compute probabilities
        self.Likelihood = (self.Likelihood + self.laplace) / (self.Prior + self.laplace * len(self.Y_dict))
        
        self.Prior = self.Prior / sum(self.Prior)

    def feat_vec(self, x):
        vec = np.zeros(len(self.X_dict))
        feat_set = x if self.mode == 'Multinomial' else set(x)
        for feat in feat_set:
            if feat in self.X_dict:
                vec[self.X_dict[feat]] += 1
        return vec

    def predict(self, x, i):
        if self.mode == 'Multinomial' or self.mode == 'Bernoulli':
            print('predicting new x {}...'.format(i))
            vec = self.feat_vec(x)
            # return max([(np.dot(vec, np.log(self.Likelihood[:, i])) + np.log(self.Prior[i]), y) for y, i in self.Y_dict.items()])[1]
            return max([(np.dot(vec, self.process_likelihood(vec, self.Likelihood[:, i])) + np.log(self.Prior[i]), y) for y, i in self.Y_dict.items()])[1]
        elif self.mode == 'Gaussian':
            return max(self.gaussian_prob(x))[1]

    def process_likelihood(self, feat_vec, likelihood):
        if self.mode == 'Bernoulli':
            return np.log([p if feat_vec[i] > 0 else 1 - p for i, p in enumerate(likelihood)])
        return np.log(likelihood)

    def score(self, X, Y):
        Y_pred = [self.predict(x, i) for i, x in enumerate(X)]
        print(classification_report(Y, Y_pred))

def generate_news_data():
    def split_data(dataset):
        dataset = [data.lower().split() for data in dataset]
        return dataset

    from sklearn.datasets import fetch_20newsgroups
    news = fetch_20newsgroups()
    X = news.data
    Y = news.target
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y)
    print('Train data shape', X_train.shape)
    print('Validation data shape', X_val.shape)
    return split_data(X_train), split_data(X_val), Y_train, Y_val

def generate_iris_data():
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    Y = iris.target
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y)
    print('Train data shape', X_train.shape)
    print('Validation data shape', X_val.shape)
    return X_train, X_val, Y_train, Y_val

if __name__ == '__main__':
    X_train, X_val, Y_train, Y_val = generate_news_data()
    # X_train, X_val, Y_train, Y_val = generate_iris_data()
    nb = NaiveBayesClassifier(mode='Bernoulli')
    # # nb = NaiveBayesClassifier(mode='Multinomial')
    # nb = NaiveBayesClassifier(mode='Gaussian')
    nb.train(X_train, Y_train)
    nb.score(X_val, Y_val)