import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

np.random.seed(42)
random.seed(42)

class HiddenMarkovModel:
    def __init__(self):
        self.Trans = None
        self.Emit = None
        self.Prior = None

        self.S_dict = None
        self.O_dict = None
        self.N = None
        self.M = None
    
    def load_data(self, X, Y):
        ob_set = {ob for x in X for ob in x} | {'<unk>'}
        state_set = {state for y in Y for state in y}
        self.S_dict = {state: i for i, state in enumerate(state_set)}
        self._S_dict = {i: state for state, i in self.S_dict.items()}
        self.O_dict = {ob: i for i, ob in enumerate(ob_set)}
        self.N = len(self.S_dict)
        self.M = len(self.O_dict)

        # add-1 laplace for all metrices
        self.Trans = np.zeros((self.N, self.N)) + 1
        self.Emit = np.zeros((self.N, self.M)) + 1
        self.Prior = np.zeros(self.N) + 1

    def train(self, X, Y):
        self.load_data(X, Y)

        for x, y in zip(X, Y):
            o_last, s_first, s_last = self.O_dict[x[-1]], self.S_dict[y[0]], self.S_dict[y[-1]]
            self.Prior[s_first] += 1
            self.Trans[s_last, s_last] += 1
            self.Emit[s_last, o_last] += 1

            for i in range(len(x) - 1):
                o1, s1, s2 = self.O_dict[x[i]], self.S_dict[y[i]], self.S_dict[y[i + 1]]
                self.Trans[s1, s2] += 1
                self.Emit[s1, o1] += 1

        S_count = np.sum(self.Trans, axis=1).reshape(-1, 1)
        self.Trans = self.Trans / S_count
        self.Emit = self.Emit / (S_count - self.N + self.M)
        self.Prior = self.Prior / sum(self.Prior)

    def forward(self, x):
        F = np.zeros((self.N, len(x)))
        F[:, 0] = self.Prior * self.Emit[:, x[0]]    # initialize all states from Prior
        for t in range(1, len(x)):  # for each t, for each state, sum(all prev-state * transition * ob)
            for s in range(self.N):
                paths = F[:, t-1] * self.Trans[:, s] * self.Emit[s, x[t]]
                F[s, t] = np.sum(paths)
        return F

    def backward(self, x):
        B = np.zeros((self.N, len(x)))
        B[:, -1] = np.ones(self.N)
        for t in range(len(x) - 2, -1, -1):
            for s in range(self.N):
                paths = B[:, t+1] * self.Trans[s] * self.Emit[:, x[t+1]]
                B[s, t] = np.sum(paths)
        return B
    
    def viterbi(self, x):
        V = np.zeros((self.N, len(x)))
        B = np.zeros((self.N, len(x)))
        V[:, 0] = self.Prior * self.Emit[:, x[0]]
        for t in range(1, len(x)):               # for each t, for each state, choose the biggest from all prev-state * transition * ob, remember the best prev
            for s in range(self.N):
                paths = V[:, t-1] * self.Trans[:, s] * self.Emit[s, x[t]]
                V[s, t] = np.max(paths)
                B[s, t] = np.argmax(paths)
        return V, B
    
    def backtrace(self, V, B):
        best = np.argmax(V[:, -1])
        path = []
        for t in range(B.shape[1] - 1, -1, -1):
            path.append(self._S_dict[best])
            best = int(B[best, t])
        return path[::-1]

    def featurize(self, x):
        '''Handle unk, and convert features to ids.'''
        return [self.O_dict[ob] if ob in self.O_dict else self.O_dict['<unk>'] for ob in x]

    def likelihood(self, x):
        f = self.featurize(x)
        F = self.forward(f)
        return np.sum(F[:, -1])  # sum(self.backward(f)[:, 0] * self.Prior * self.Emit[:, f[0]])

    def decode(self, x):
        f = self.featurize(x)
        V, B = self.viterbi(f)
        return self.backtrace(V, B)
    
    def score(self, X, Y):
        Y_true = [s for y in Y for s in y]
        Y_pred = [s for x in X for s in self.decode(x)]
        print(classification_report(Y_true, Y_pred))

    def learn(self):
        pass

def generate_data():
    from nltk.corpus import brown
    X = []
    Y = []
    for sent in brown.tagged_sents():
        x = []
        y = []
        for w, pos in sent:
            x.append(w)
            y.append(pos)
        X.append(x)
        Y.append(y)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y)
    return X_train, X_val, Y_train, Y_val


if __name__ == '__main__':
    X_train, X_val, Y_train, Y_val = generate_data()
    hmm = HiddenMarkovModel()
    hmm.train(X_train, Y_train)
    # hmm.score(X_val, Y_val)

    print(np.sum(hmm.Trans, axis=1))
    print(np.sum(hmm.Emit, axis=1))
    print(np.sum(hmm.Prior))

    def test_(X, Y):
        correct_num = 0.0
        token_num = 0.0
        for x, y in zip(X[10:11], Y[10:11]):
            print(x)
            result = hmm.decode(x)
            print(y)
            print(result)
            ch_x = hmm.featurize(x)
            B = hmm.backward(ch_x)
            # print(sum(B[:, 0] * hmm.Prior))
            print(sum(B[:, 0] * hmm.Prior * hmm.Emit[:, ch_x[0]]))
            print(hmm.likelihood(x))
            print()
            token_num += len(x)
            for i in range(len(x)):
                if y[i] == result[i]:
                    correct_num += 1
        print ('\nAccuracy: ' + str(correct_num / token_num))
    test_(X_val, Y_val)