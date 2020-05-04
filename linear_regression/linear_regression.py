import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

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

    def train(self, X_train, X_val, Y_train, Y_val, mode='SGD'):
        self.initialize_weights(X_train.shape[1])

        if mode == 'BGD':
            self.train_BGD(X_train, X_val, Y_train, Y_val)
        elif mode == 'SGD':
            self.train_SGD(X_train, X_val, Y_train, Y_val)
        elif mode == 'MBGD':
            self.train_MBGD(X_train, X_val, Y_train, Y_val)
        elif mode == 'NE':
            self.train_NE(X_train, X_val, Y_train, Y_val)

    def predict(self, X):
        # broadcast multiplication of weight vector to sample features,
        # sum each row to get predictions.
        # pred_y here is a row vector.
        return np.sum(self.w * X, axis=1) + self.b

    def compute_loss(self, X, Y):
        Y_pred = self.predict(X)
        return (1 / (2 * len(X))) * np.sum((Y_pred - Y) ** 2)

    def show_loss(self, X_train, X_val, Y_train, Y_val, epoch_num):
        train_loss = self.compute_loss(X_train, Y_train)
        val_loss = self.compute_loss(X_val, Y_val)

        print("Training loss at epoch {}: {}".format(epoch_num, train_loss))
        print("Validation loss at eopch {}: {}\n".format(epoch_num, val_loss))

    def update_weights(self, X, Y):
        # broadcast multiplication of error diff of each sample to all its features
        # then accumulate all errors of each feature weight, update
        Y_pred = self.predict(X)
        error_diffs = (Y_pred - Y).reshape(-1, 1)
        d_w = (1 / len(X)) * np.sum(error_diffs * X, axis=0)
        d_b = (1 / len(X)) * np.sum(error_diffs)
        self.w -= self.lr * d_w
        self.b -= self.lr * d_b

    def shuffle_data(self, X, Y):
        shuffled_index = np.random.permutation(len(X))
        X = X[shuffled_index]
        Y = Y[shuffled_index]
        return X, Y

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

    def train_NE(self, X_train, X_val, Y_train, Y_val):

        self.show_loss(X_train, X_val, Y_train, Y_val, 0)

        # add bias terms to all samples, default as 1
        X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]

        # conduct normal equations
        a = np.dot(X_train_b.T, X_train_b)
        b = np.linalg.inv(a)
        c = np.dot(b, X_train_b.T)
        theta = np.dot(c, Y_train)

        # the 1st item is the bias
        self.b, self.w = theta[0], theta[1:]

        self.show_loss(X_train, X_val, Y_train, Y_val, 1)
    
    def score(self, X, Y):
        Y_pred = self.predict(X)
        self.regression_report(Y, Y_pred)
    
    def regression_report(self, Y_true, Y_pred):
        explained_variance=metrics.explained_variance_score(Y_true, Y_pred)
        mean_absolute_error=metrics.mean_absolute_error(Y_true, Y_pred) 
        mse=metrics.mean_squared_error(Y_true, Y_pred) 
        mean_squared_log_error=metrics.mean_squared_log_error(Y_true, Y_pred)
        median_absolute_error=metrics.median_absolute_error(Y_true, Y_pred)
        r2=metrics.r2_score(Y_true, Y_pred)

        print('Explained_variance: ', round(explained_variance,4))    
        print('Mean_squared_log_error: ', round(mean_squared_log_error,4))
        print('Median_absolute_error: ', round(median_absolute_error))
        print('R2: ', round(r2,4))
        print('MAE: ', round(mean_absolute_error,4))
        print('MSE: ', round(mse,4))
        print('RMSE: ', round(np.sqrt(mse),4))


def generate_parameters(num_features):
    w = np.random.randn(num_features)
    b = np.random.randint(5)
    return w, b


def generate_data(num_samples, num_features, w, b):
    # each feature satisfies normal distribution, with different means
    # and standard deviations
    X = np.array([np.random.normal(
        loc=np.random.randint(10),
        scale=np.random.random() * 5,
        size=num_samples)
        for _ in range(num_features)]).T
    Y = np.sum(w * X, axis=1) + b

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y)

    print('Train data shape', X_train.shape)
    print('Validation data shape', X_val.shape)

    return X_train, X_val, Y_train, Y_val


def generate_house_data():
    from sklearn.datasets.california_housing import fetch_california_housing
    from sklearn.preprocessing import StandardScaler
    houses = fetch_california_housing()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(houses.data)
    X_train, X_val, Y_train, Y_val = train_test_split(scaled_data, houses.target)
    print('Train data shape', X_train.shape)
    print('Validation data shape', X_val.shape)
    return X_train, X_val, Y_train, Y_val

if __name__ == '__main__':
    # W, B = generate_parameters(5)
    X_train, X_val, Y_train, Y_val = generate_house_data()
    # X_train, X_val, Y_train, Y_val = generate_data(1000, 5, W, B)
    lr = LinearRegression()
    lr.train(X_train, X_val, Y_train, Y_val, mode='MBGD')
    # lr.train(X_train, X_val, Y_train, Y_val, mode='SGD')
    # lr.train(X_train, X_val, Y_train, Y_val, mode='BGD')
    # lr.train(X_train, X_val, Y_train, Y_val, mode='NE')
    # print('Pred weights:', lr.w)
    # print('True weights:', W)
    # print('Pred bias:', lr.b)
    # print('True bias:', B)
    lr.score(X_val, Y_val)