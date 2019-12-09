import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

np.random.seed(42)


class LogisticRegression:

    def __init__(self, learning_rate=0.001, epoch=500, batch_size=150):
        self.lr = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.w = None
        self.b = None

    def initialize_weights(self, num_features):
        self.w = np.random.randn(num_features)
        self.b = np.random.randint(5)

    def predict(self, x):
        pred_prob = self.sigmoid(x)
        pred_y = np.array([0 if prob < 0.5 else 1 for prob in pred_prob])
        return pred_y

    def sigmoid(self, x):
        z = np.sum(self.w * x, axis=1) + self.b
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, x, y):
        # minimize loss = minimize negative log likelihood = maximize likelihood
        pred_prob = self.sigmoid(x)
        return -np.mean(y * np.log(pred_prob) + (1 - y) * np.log(1 - pred_prob))

    def show_loss(self, train_x, val_x, train_y, val_y, epoch_num):
        train_loss = self.compute_loss(train_x, train_y)
        val_loss = self.compute_loss(val_x, val_y)

        print("Training loss at epoch {}: {}".format(epoch_num, train_loss))
        print("Validation loss at eopch {}: {}\n".format(epoch_num, val_loss))

    def shuffle_data(self, x, y):
        shuffled_index = np.random.permutation(len(x))
        x = x[shuffled_index]
        y = y[shuffled_index]
        return x, y

    def train(self, train_x, val_x, train_y, val_y, mode='SGD'):
        self.initialize_weights(train_x.shape[1])
        if mode == 'BGD':
            self.train_BGD(train_x, val_x, train_y, val_y)
        elif mode == 'SGD':
            self.train_SGD(train_x, val_x, train_y, val_y)
        elif mode == 'MBGD':
            self.train_MBGD(train_x, val_x, train_y, val_y)

    def train_BGD(self, train_x, val_x, train_y, val_y):
        for i in range(1, self.epoch + 1):

            # Step 1: compute train and validation loss
            if i % 100 == 0 or i == 1:
                self.show_loss(train_x, val_x, train_y, val_y, i)

            # Step 2: compute gradient and update weights
            self.update_weights(train_x, train_y)

    def train_MBGD(self, train_x, val_x, train_y, val_y):
        # construct a batch generator
        def batch_generator(x, y):
            num_samples = len(x)
            for i in range(0, num_samples):
                yield x[i: min(i + self.batch_size, num_samples)], \
                      y[i: min(i + self.batch_size, num_samples)]

        for i in range(1, self.epoch + 1):

            # Step 1: compute train and validation loss
            if i % 100 == 0 or i == 1:
                self.show_loss(train_x, val_x, train_y, val_y, i)

            # Step 2: shuffle data
            train_x, train_y = self.shuffle_data(train_x, train_y)

            # Step 3: compute gradients and update weights
            for batch_x, batch_y in batch_generator(train_x, train_y):
                self.update_weights(batch_x, batch_y)

    def train_SGD(self, train_x, val_x, train_y, val_y):
        for i in range(1, self.epoch + 1):

            # Step 1: compute train and validation loss
            if i % 100 == 0 or i == 1:
                self.show_loss(train_x, val_x, train_y, val_y, i)

            # Step 2: shuffle data
            train_x, train_y = self.shuffle_data(train_x, train_y)

            # Step 3: compute gradients and update weights
            for x, y in zip(train_x, train_y):
                self.update_weights([x], [y])

    def update_weights(self, x, y):
        pred_prob = self.sigmoid(x)
        error_diffs = (pred_prob - y).reshape(-1, 1)
        d_w = (1 / len(x)) * np.sum(error_diffs * x, axis=0)
        d_b = (1 / len(x)) * np.sum(error_diffs)
        self.w -= self.lr * d_w
        self.b -= self.lr * d_b

    def score(self, x, y):
        pred_y = self.predict(x)
        print(classification_report(y, pred_y))


def generate_data():
    iris = load_iris()
    train_x, val_x, train_y, val_y = train_test_split(iris.data[:100], iris.target[:100])
    return train_x, val_x, train_y, val_y


if __name__ == "__main__":
    train_x, val_x, train_y, val_y = generate_data()
    lr = LogisticRegression(batch_size=10)
    # lr.train(train_x, val_x, train_y, val_y, mode='BGD')
    lr.train(train_x, val_x, train_y, val_y, mode='MBGD')
    # lr.train(train_x, val_x, train_y, val_y, mode='SGD')
    print(lr.w)
    print(lr.b)
    lr.score(val_x, val_y)
