import numpy as np
import random

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
        self.O_dict = {ob: i for i, ob in enumerate(ob_set)}
        self.N = len(self.S_dict)
        self.M = len(self.O_dict)
        # self.S_dict['<END>'] = self.N

        # add-1 laplace for all metrices
        self.Trans = np.zeros((self.N, self.N)) + 1
        self.Emit = np.zeros((self.N, self.M)) + 1
        self.Prior = np.zeros(self.N) + 1

    def train(self, X, Y):
        self.load_data(X, Y)

        for x, y in zip(X, Y):
            o_last, s_first, s_last = self.O_dict[x[-1]], self.S_dict[y[0]], self.S_dict[y[-1]]
            self.Prior[s_first] += 1
            # self.Trans[s_last, self.S_dict['<END>']] += 1
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
        for t in range(1, len(x)):   # for each t, for each state, sum(all prev-state * transition * ob)
            F[:, t] = [np.dot(F[:, t-1], self.Trans[s]) * self.Emit[x[t], s] for s in range(self.N)]
        return F

    def backward(self, x):
        B = np.zeros((self.N, len(x)))
        B[:, -1] = self.Trans[-1]               # np.ones(self.T.shape[1])
        for t in range(len(x) - 2, -1, -1):
            B[:, t] = [sum(B[:, t+1] * self.Trans[:, s] * self.Emit[x[t+1]]) for s in range(self.N)]
        return B
    
    def checkOb(self, x):
        '''Handle unk, >> also convert feature to id.'''
        return [self.O_dict[ob] if ob in self.O_dict else self.O_dict['<unk>'] for ob in x]

    def likelihood(self, x):
        pass

    def decode(self, x):
        pass

    def learn(self):
        pass

if __name__ == '__main__':
    # m = np.array([[1,2,3,4], [4,5,6,4], [7,8,9,4]])
    # col = np.array([1, 2, 3]).reshape(-1, 1)
    # print(m)
    # # print(col)
    # # m = m /col
    # # print(m)
    # ret = np.sum(m, axis=1).reshape(-1, 1)
    # print(ret - 1)

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
    hmm = HiddenMarkovModel()
    hmm.train(X, Y)
    print(hmm.Trans[10].sum()) # test if all transitions from one state add to 1 in Transition matrix.
    print(hmm.Emit[10].sum()) # test if all emissions from one state add to 1 in Emission maxtrix.
    print(hmm.Prior.sum())
    # S_count = np.sum(hmm.Trans, axis=1).reshape(-1, 1)
    # print(sum(hmm.Trans[10]))
    # print(S_count[10][0])
    # print(sum(hmm.Trans[-10]))
    # print(S_count[-10][0])