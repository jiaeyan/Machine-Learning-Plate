import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

"""
BGD needs much more iterations than SGD/MBGD.
"""


class LinearRegression:

    def __init__(self, learning_rate=0.001, epoch=300, batch_size=150):
        self.lr = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.w = None
        self.b = None

    def initialize_weights(self, num_features):
        self.w = np.random.randn(num_features)
        self.b = np.random.randint(5)

    def train(self, train_x, val_x, train_y, val_y, mode='SGD'):
        self.initialize_weights(train_x.shape[1])

        if mode == 'BGD':
            self.train_BGD(train_x, val_x, train_y, val_y)
        elif mode == 'SGD':
            self.train_SGD(train_x, val_x, train_y, val_y)
        elif mode == 'MBGD':
            self.train_MBGD(train_x, val_x, train_y, val_y)
        elif mode == 'NE':
            self.train_NE(train_x, val_x, train_y, val_y)

    def predict(self, x):
        # broadcast multiplication of weight vector to sample features,
        # sum each row to get predictions.
        # pred_y here is a row vector.
        return np.sum(self.w * x, axis=1) + self.b

    def compute_loss(self, x, y):
        pred_y = self.predict(x)
        return (1 / (2 * len(x))) * np.sum((pred_y - y) ** 2)

    def show_loss(self, train_x, val_x, train_y, val_y, epoch_num):
        train_loss = self.compute_loss(train_x, train_y)
        val_loss = self.compute_loss(val_x, val_y)

        print("Training loss at epoch {}: {}".format(epoch_num, train_loss))
        print("Validation loss at eopch {}: {}\n".format(epoch_num, val_loss))

    def update_weights(self, x, y):
        # broadcast multiplication of error diff of each sample to all its features
        # then accumulate all errors of each feature weight, update
        pred_y = self.predict(x)
        error_diffs = (pred_y - y).reshape(-1, 1)
        d_w = (1 / len(x)) * np.sum(error_diffs * x, axis=0)
        d_b = (1 / len(x)) * np.sum(error_diffs)
        self.w -= self.lr * d_w
        self.b -= self.lr * d_b

    def shuffle_data(self, x, y):
        shuffled_index = np.random.permutation(len(x))
        x = x[shuffled_index]
        y = y[shuffled_index]
        return x, y

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

    def train_NE(self, train_x, val_x, train_y, val_y):
        self.show_loss(train_x, val_x, train_y, val_y, 0)

        # add bias terms to all samples, default as 1
        train_x_b = np.c_[np.ones((len(train_x), 1)), train_x]

        # conduct normal equations
        a = np.dot(train_x_b.T, train_x_b)
        b = np.linalg.inv(a)
        c = np.dot(b, train_x_b.T)
        theta = np.dot(c, train_y)

        # the 1st item is the bias
        self.b, self.w = theta[0], theta[1:]

        self.show_loss(train_x, val_x, train_y, val_y, 1)


def generate_parameters(num_features):
    W = np.random.randn(num_features)
    B = np.random.randint(5)
    return W, B


def generate_data(num_samples, num_features, W, B):
    # each feature satisfies normal distribution, with different means
    # and standard deviations
    X = np.array([np.random.normal(
        loc=np.random.randint(10),
        scale=np.random.random() * 5,
        size=num_samples)
        for _ in range(num_features)]).T
    Y = np.sum(W * X, axis=1) + B

    train_x, val_x, train_y, val_y = train_test_split(X, Y)

    return train_x, val_x, train_y, val_y

def generate_house_data():
    houses = fetch_california_housing()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(houses.data)
    # data = minmax_scale(houses.data)
    train_x, val_x, train_y, val_y = train_test_split(scaled_data, houses.target)
    return train_x, val_x, train_y, val_y

if __name__ == '__main__':
    W, B = generate_parameters(5)
    # train_x, val_x, train_y, val_y = generate_house_data()
    train_x, val_x, train_y, val_y = generate_data(1000, 5, W, B)
    lr = LinearRegression()
    lr.train(train_x, val_x, train_y, val_y, mode='MBGD')
    # lr.train(train_x, val_x, train_y, val_y, mode='SGD')
    # lr.train(train_x, val_x, train_y, val_y, mode='BGD')
    # lr.train(train_x, val_x, train_y, val_y, mode='NE')
    print('Pred weights:', lr.w)
    print('True weights:', W)
    print('Pred bias:', lr.b)
    print('True bias:', B)