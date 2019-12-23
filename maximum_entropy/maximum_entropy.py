import numpy as np
from math import exp
from math import log
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

np.random.seed(42)

class MaximumEntropy:

    def __init__(self, learning_rate=0.001, epoch=500, batch_size=150):
        self.lr = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.W = None
        self.X_dict = None
        self.Y_dict = None

    def load_data(self, X, Y):
        feat_set = {feat for x in X for feat in x}
        self.X_dict = {feat: i for i, feat in enumerate(feat_set)}
        self.Y_dict = {y: i for i, y in enumerate(set(Y))}
        self.W = np.zeros((len(self.Y_dict), len(self.X_dict)))

    def remake_data(self, X, Y):
        X = np.array([self.featurize(x) for x in X])
        Y = np.array(Y)
        return X, Y

    def featurize(self, x):
        feat_vec = np.zeros(len(self.X_dict))
        for feat in x:
            if feat in self.X_dict:
                feat_vec[self.X_dict[feat]] = 1
        return feat_vec

    def shuffle_data(self, X, Y):
        shuffled_index = np.random.permutation(len(X))
        X = X[shuffled_index]
        Y = Y[shuffled_index]
        return X, Y

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
        ret = [(np.dot(self.W[i], feat_vec), y) for y, i in self.Y_dict.items()]
        return max(ret)[1]

    def train(self, X_train, X_val, Y_train, Y_val):

        # construct a batch generator
        def batch_generator(X, Y):
            num_samples = len(X)
            for i in range(0, num_samples, self.batch_size):
                yield X[i: min(i + self.batch_size, num_samples)], \
                      Y[i: min(i + self.batch_size, num_samples)]

        self.load_data(X_train, Y_train)
        X_train, Y_train = self.remake_data(X_train, Y_train)
        X_val, Y_val = self.remake_data(X_val, Y_val)

        for i in range(1, self.epoch + 1):
            # Step 1: compute train and validation loss
            if i % 100 == 0 or i == 1:
                self.show_loss(X_train, X_val, Y_train, Y_val, i)

            # Step 2: shuffle data
            X_train, Y_train = self.shuffle_data(X_train, Y_train)

            # Step 3: compute gradient and update weights
            for X_batch, Y_batch in batch_generator(X_train, Y_train):
                self.update_weights(X_batch, Y_batch)

    def posterior(self, x, y):
        """Compute p(y|x) by softmax."""
        y_prob = exp(np.dot(self.W[self.Y_dict[y]], x))
        z = sum([exp(np.dot(self.W[self.Y_dict[y]], x)) for y in self.Y_dict])
        return y_prob / z

    def update_weights(self, X, Y):
        # observed expectations from data
        ob_exp = np.zeros(self.W.shape)
        # model expectations from model parameters
        model_exp = np.zeros(self.W.shape)

        for i in range(len(X)):
            ob_exp[self.Y_dict[Y[i]]] += X[i]
            for y, y_id in self.Y_dict.items():
                model_exp[y_id] += X[i] * self.posterior(X[i], y)

        self.W -= self.lr * (model_exp - ob_exp)

    def score(self, X, Y):
        Y_pred = [self.predict(x) for x in X]
        print(classification_report(Y, Y_pred))


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

if __name__ == '__main__':
    X, Y = generate_data()
    maxent = MaximumEntropy(batch_size=5)
    maxent.train(X, X, Y, Y)
    maxent.score(X, Y)
    print(maxent.predict(['sunny', 'hot', 'high', 'FALSE']))
    # X_train, X_val, Y_train, Y_val = generate_news_data()
    # maxent = MaximumEntropy(batch_size=1024)
    # maxent.train(X_train, X_val, Y_train, Y_val)
    # maxent.score(X_val, Y_val)