import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

np.random.seed(42)


class LogisticRegression:

    def __init__(self, learning_rate=0.001, epoch=300, batch_size=150):
        self.lr = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.w = None
        self.b = None

    def initialize_weights(self, num_features):
        self.w = np.random.randn(num_features)
        self.b = np.random.randint(5)

    def predict(self, X):
        pred_probs = self.sigmoid(X)
        pred_Y = np.array([0 if prob < 0.5 else 1 for prob in pred_probs])
        return pred_Y

    def sigmoid(self, X):
        Z = np.sum(self.w * X, axis=1) + self.b
        return 1 / (1 + np.exp(-Z))

    def compute_loss(self, X, Y):
        # minimize loss = minimize negative log likelihood = maximize likelihood
        pred_probs = self.sigmoid(X)
        return -np.mean(Y * np.log(pred_probs) + (1 - Y) * np.log(1 - pred_probs))

    def show_loss(self, X_train, X_val, Y_train, Y_val, epoch_num):
        train_loss = self.compute_loss(X_train, Y_train)
        val_loss = self.compute_loss(X_val, Y_val)

        print("Training loss at epoch {}: {}".format(epoch_num, train_loss))
        print("Validation loss at eopch {}: {}\n".format(epoch_num, val_loss))

    def shuffle_data(self, X, Y):
        shuffled_index = np.random.permutation(len(X))
        X = X[shuffled_index]
        Y = Y[shuffled_index]
        return X, Y

    def train(self, X_train, X_val, Y_train, Y_val, mode='SGD'):
        self.initialize_weights(X_train.shape[1])

        if mode == 'BGD':
            self.train_BGD(X_train, X_val, Y_train, Y_val)
        elif mode == 'SGD':
            self.train_SGD(X_train, X_val, Y_train, Y_val)
        elif mode == 'MBGD':
            self.train_MBGD(X_train, X_val, Y_train, Y_val)

    def train_BGD(self, X_train, X_val, Y_train, Y_val):
        for i in range(1, self.epoch + 1):

            # Step 1: compute train and validation loss
            if i % 100 == 0 or i == 1:
                self.show_loss(X_train, X_val, Y_train, Y_val, i)

            # Step 2: compute gradient and update weights
            self.update_weights(X_train, Y_train)

    def train_MBGD(self, X_train, X_val, Y_train, Y_val):
        # construct a batch generator
        def batch_generator(X, Y):
            num_samples = len(X)
            for i in range(0, num_samples, self.batch_size):
                yield X[i: min(i + self.batch_size, num_samples)], \
                      Y[i: min(i + self.batch_size, num_samples)]

        for i in range(1, self.epoch + 1):

            # Step 1: compute train and validation loss
            if i % 100 == 0 or i == 1:
                self.show_loss(X_train, X_val, Y_train, Y_val, i)

            # Step 2: shuffle data
            X_train, Y_train = self.shuffle_data(X_train, Y_train)

            # Step 3: compute gradients and update weights
            for X_batch, Y_batch in batch_generator(X_train, Y_train):
                self.update_weights(X_batch, Y_batch)

    def train_SGD(self, X_train, X_val, Y_train, Y_val):
        for i in range(1, self.epoch + 1):

            # Step 1: compute train and validation loss
            if i % 100 == 0 or i == 1:
                self.show_loss(X_train, X_val, Y_train, Y_val, i)

            # Step 2: shuffle data
            X_train, Y_train = self.shuffle_data(X_train, Y_train)

            # Step 3: compute gradients and update weights
            for x, y in zip(X_train, Y_train):
                self.update_weights([x], [y])

    def update_weights(self, X, Y):
        pred_probs = self.sigmoid(X)
        error_diffs = (pred_probs - Y).reshape(-1, 1)
        d_w = (1 / len(X)) * np.sum(error_diffs * X, axis=0)
        d_b = (1 / len(X)) * np.sum(error_diffs)
        self.w -= self.lr * d_w
        self.b -= self.lr * d_b

    def score(self, X, Y):
        pred_Y = self.predict(X)
        print(classification_report(Y, pred_Y))


def generate_data():
    iris = load_iris()
    X_train, X_val, Y_train, Y_val = train_test_split(iris.data[:100], iris.target[:100])
    return X_train, X_val, Y_train, Y_val


if __name__ == "__main__":
    X_train, X_val, Y_train, Y_val = generate_data()
    lr = LogisticRegression(batch_size=10)
    # lr.train(X_train, X_val, Y_train, Y_val, mode='BGD')
    lr.train(X_train, X_val, Y_train, Y_val, mode='MBGD')
    # lr.train(X_train, X_val, Y_train, Y_val, mode='SGD')
    print(lr.w)
    print(lr.b)
    lr.score(X_val, Y_val)
