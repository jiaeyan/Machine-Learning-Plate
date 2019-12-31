import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

np.random.seed(42)
random.seed(42)

class HiddenMarkovModel:
    def __init__(self):
        self.Prior = None
        self.Trans = None
        self.Emit = None

        self.S_dict = None
        self._S_dict = None
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

        # add-1 laplace to all metrices
        self.Prior = np.zeros(self.N) + 1
        self.Trans = np.zeros((self.N, self.N)) + 1
        self.Emit = np.zeros((self.N, self.M)) + 1

    def train(self, X, Y):
        self.load_data(X, Y)

        for x, y in zip(X, Y):
            o_end, s_start, s_end = self.O_dict[x[-1]], self.S_dict[y[0]], self.S_dict[y[-1]]
            self.Prior[s_start] += 1
            self.Trans[s_end, s_end] += 1
            self.Emit[s_end, o_end] += 1

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
        F[:, 0] = self.Prior * self.Emit[:, x[0]]
        for t in range(1, len(x)):
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
        B = np.zeros((self.N, len(x)), dtype=int)
        V[:, 0] = self.Prior * self.Emit[:, x[0]]
        for t in range(1, len(x)):
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
            best = B[best, t]
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
    
    def learn(self, X, Y, iterations=100):
        self.load_data(X, Y)
        self.init_parameters()
        X = [self.featurize(x) for x in X]

        for _ in range(iterations):
            Gammas, Xis = self.E_step(X)
            self.M_step(X, Gammas, Xis)
    
    def init_parameters(self):
        self.Prior = np.zeros(self.N) + 1/self.N
        self.Trans = np.zeros((self.N, self.N)) + 1/self.N

        # To make some init parameters unequal. 
        # BW algorithm works poor with all equal init parameters.
        Sum = self.Trans[0][0] + self.Trans[0][-1]
        self.Trans[0][0], self.Trans[0][-1] = Sum / 3, 2 * Sum / 3

        self.Emit = np.zeros((self.N, self.M)) + 1/self.M
    
    def E_step(self, X):
        Gammas = []
        Xis = []
        for x in X:
            F = self.forward(x)
            B = self.backward(x)
            Gamma = self.gamma(F, B)
            Gammas.append(Gamma)
            Xi = self.xi(x, F, B)
            Xis.append(Xi)
        return Gammas, Xis

    def gamma(self, F, B):
        Gamma = F * B
        Gamma = Gamma / np.sum(Gamma, 0)
        return Gamma
    
    def xi(self, x, F, B):
        Xi = np.zeros((self.N, self.N, len(x) - 1))
        for t in range(len(x) - 1):
            for i in range(self.N):
                for j in range(self.N):
                    Xi[i, j, t] = F[i, t] * self.Trans[i, j] * self.Emit[j, x[t+1]] * B[j, t+1]
            Xi[:, :, t] /= np.sum(np.sum(Xi[:, :, t], 1), 0)	
        return Xi

    def M_step(self, X, Gammas, Xis):
        self.learn_prior(X, Gammas)
        self.learn_trans(X, Gammas, Xis)
        self.learn_emit(X, Gammas)
    
    def learn_prior(self, X, Gammas):
        for i in range(self.N):
            gammas = np.sum(Gammas[xid][i, 0] for xid in range(len(X)))
            self.Prior[i] = gammas / len(X)
    
    def learn_trans(self, X, Gammas, Xis):
        for i in range(self.N):
            denominator = np.sum(np.sum(Gammas[xid][i, :len(x) - 1]) for xid, x in enumerate(X))
            for j in range(self.N):
                numerator = np.sum(np.sum(Xis[xid][i, j, :len(x) - 1]) for xid, x in enumerate(X))
                self.Trans[i, j] = numerator / denominator
    
    def learn_emit(self, X, Gammas):
        for j in range(self.N):
            denominator = np.sum(np.sum(Gammas[xid][j]) for xid in range(len(X)))
            for k in range(self.M):
                numerator = 0.0
                for xid, x in enumerate(X):
                    for t in range(len(x)):
                        if x[t] == k:
                            numerator += Gammas[xid][j, t]
                self.Emit[j, k] = numerator / denominator

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

def test_learn():
    hmm = HiddenMarkovModel()
    X = [
        ['he', 'want', 'to', 'eat', 'food'],
        ['John', 'eat', 'food'],
        ['he', 'want', 'food'],
        ['John', 'want', 'food']
    ]
    Y = [
        ['PRON', 'VB', 'TO', 'VB', 'NN', 'NN', 'VB', 'PRON', 'VB', 'NN', 'TO', 'VB'],
        ['NNP', 'VB', 'NN'],
        ['PRON', 'VB', 'NN'],
        ['NNP', 'VB', 'NN']
    ]
    hmm.learn(X, Y, iterations=50)
    print(hmm.decode(['John', 'want', 'to', 'eat']))

def test_train():
    X_train, X_val, Y_train, Y_val = generate_data()
    hmm = HiddenMarkovModel()
    hmm.train(X_train, Y_train)

    print(np.sum(hmm.Trans, axis=1))
    print(np.sum(hmm.Emit, axis=1))
    print(np.sum(hmm.Prior))

    x = X_val[0]
    y = Y_val[0]
    print('Instance:', x)
    print('True labels:', y)
    print('Predicted labels:', hmm.decode(x))
    ch_x = hmm.featurize(x)
    B = hmm.backward(ch_x)
    print('Forward prob:', hmm.likelihood(x))
    print('Backward prob:', sum(B[:, 0] * hmm.Prior * hmm.Emit[:, ch_x[0]]))
    print()

    hmm.score(X_val[:10], Y_val[:10])
    print(hmm.decode(['John', 'want', 'to', 'eat']))

if __name__ == '__main__':
    # test_learn()
    test_train()