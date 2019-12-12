import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report

np.random.seed(42)


class Perceptron:

    def __init__(self, learning_rate=0.001):
        self.lr = learning_rate
        self.w = None
        self.b = None

    def initialize_weights(self, num_features):
        self.w = np.random.randn(num_features)
        self.b = np.random.randint(5)

    def predict(self, X):
        Z = self.sign(X)
        return [1 if z > 0 else -1 for z in Z]

    def sign(self, X):
        return np.sum(self.w * X, axis=1) + self.b

    def compute_loss(self, X, Y, epoch_num):
        loss = -np.sum(self.sign(X) * Y)
        print("Training loss at epoch {}: {}".format(epoch_num, loss))

    def train(self, X, Y):
        self.initialize_weights(X.shape[1])
        epoch = 0
        converge = False

        while not converge:
            wrong = False
            X_wrong = []
            Y_wrong = []
            for x, y in zip(X, Y):
                if y * self.sign([x]) <= 0:
                    wrong = True
                    X_wrong.append(x)
                    Y_wrong.append(y)

            if wrong:
                self.compute_loss(X_wrong, Y_wrong, epoch)
                for x, y in zip(X_wrong, Y_wrong):
                    self.w -= -self.lr * y * x
                    self.b -= -self.lr * y
            else:
                converge = True

            epoch += 1

    def score(self, X, Y):
        pred_Y = self.predict(X)
        print(classification_report(Y, pred_Y))


def generate_data():
    iris = load_iris()
    X = iris.data[:100]
    Y = [1 if y == 1 else -1 for y in iris.target[:100]]
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y)
    return X_train, X_val, Y_train, Y_val


if __name__ == "__main__":
    X_train, X_val, Y_train, Y_val = generate_data()
    p = Perceptron()
    p.train(X_train, Y_train)
    print(p.w)
    print(p.b)
    p.score(X_val, Y_val)