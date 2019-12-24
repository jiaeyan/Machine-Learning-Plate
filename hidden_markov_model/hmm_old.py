from numpy import array, dot, zeros, log, sum
from numpy.random import uniform


class Document():
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def modify_f(self):
        self.features = map(str.lower, self.features)

class HMM():

    def train(self, data):
        '''Create parameter matrices and normalize.
           T: Transiton maxtrix
           E: Emission matrix
           P: Prior matrix
        '''
        T, E, P = self.count(data)
        S = array([T[:, col].sum() for col in range(self.N)]) # a vector recording the number of each state 
        self.T = T / S                                        # denominator = s1_unicount + s_type (include end_state)
        self.E = E / (S - self.N - 1 + self.M)
        self.P = P / P.sum()
    
    def count(self, data):
        T, E, P = self.formulate(data)
        for seq in data:
            ob, lb = seq.features, seq.labels
            P[self.S[lb[0]]] += 1                      # start transition
            E[self.O[ob[-1]], self.S[lb[-1]]] += 1     # for the last s, record emission count
            T[self.S["<END>"], self.S[lb[-1]]] += 1    # and end-transition count
            for i in range(len(ob) - 1):
                o1, s1, s2 = ob[i], lb[i], lb[i+1]
                T[self.S[s2], self.S[s1]] += 1
                E[self.O[o1], self.S[s1]] += 1
        return T, E, P
    
    def formulate(self, data, supervised = True, S_set = set(), O_set = {'<unk>'}):
        for seq in data:                                   # O_set: observation set; S_set: state set;
            O_set.update(seq.features)
            if supervised:
                S_set.update(seq.labels)
        self.M, self.N = len(O_set), len(S_set)            # the number of observation type and state type
        self.O = {o:i for i, o in enumerate(O_set)}        # a dict to record o and its id
        self.S = {"<END>":len(S_set), len(S_set):"<END>"}  # since all states transit to end state, it should be included, and we want to make sure it is always at the last of the table for computation convenience
        for i, s in enumerate(S_set):                      # a two-way dict to record s and its id
            self.S[i] = s
            self.S[s] = i
        return zeros((self.N+1, self.N)) + 1, zeros((self.M, self.N)) + 1, zeros(self.N) + 1
    
    def forward(self, ob, T):
        F = zeros((self.N, T))
        F[:, 0] = self.P * self.E[ob[0]]    # initialize all states from Pi
        for t in range(1, T):               # for each t, for each state, sum(all prev-state * transition * ob)
            F[:, t] = [dot(F[:, t-1], self.T[s]) * self.E[ob[t], s] for s in range(self.N)]
        return F                            # return dot(F[:, -1], self.T[-1])
    
    def viterbi(self, ob, T):
        V = zeros((self.N, T))
        B = zeros((self.N, T))
        V[:, 0] = self.P * self.E[ob[0]]
        for t in range(1, T):               # for each t, for each state, choose the biggest from all prev-state * transition * ob, remember the best prev
            for s in range(self.N):
                V[s, t], B[s, t] = max([(p, s) for s, p in enumerate(V[:, t-1] * self.T[s] * self.E[ob[t], s])])
        return V, B
#         best = max([(p, s) for s, p in enumerate(V[:, -1] * self.T[-1])])[1]
#         return self.backtrace([], best, B, T - 1)
    
    def backtrace(self, path, best, B, t):
        if t == -1: return path
        path.insert(0, self.S[best])
        return self.backtrace(path, int(B[best][t]), B, t-1)
    
    def backward(self, ob, T):
        B = zeros((self.N, T))
        B[:, -1] = self.T[-1]               # np.ones(self.T.shape[1])
        for t in range(T - 2, -1, -1):
            B[:, t] = [sum(B[:, t+1] * self.T[:, s][:-1] * self.E[ob[t+1]]) for s in range(self.N)]
        return B                            # return sum(B[:, 0] * self.P * self.E[ob[0]])
    
    def checkOb(self, seq):
        '''Handle unk, >> also convert feature to id.'''
        ob = []
        for o in seq.features:
            if o in self.O: ob.append(self.O[o])
            else: ob.append(self.O['<unk>'])
        return ob, len(ob)
    
    def likelihood(self, seq):
        '''Likelihood: compute the observation probability.'''
        ob, T = self.checkOb(seq)
        F = self.forward(ob, T)
        return dot(F[:, -1], self.T[-1])
    
    def data_likelihood(self, data):
        '''Compute the likelihood of the entire data.'''
        return sum([self.likelihood(seq) for seq in data])
    
    def classify(self, seq):
        '''Decoding: predict the state sequence of given observation sequence.'''
        ob, T = self.checkOb(seq)
        V, B = self.viterbi(ob, T)
        best = max([(p, s) for s, p in enumerate(V[:, -1] * self.T[-1])])[1]
        return self.backtrace([], best, B, T - 1)

if __name__ == '__main__':
    from nltk.corpus import brown

    data = []
    for seq in brown.tagged_sents():
        features, labels = [], []
        for w, pos in seq:
            features.append(w)
            labels.append(pos)
        data.append(Document(features, labels))

    bound = int(round(len(data)*0.8))
    train = data[:bound]
    test = data[bound:]

    hmm = HMM()
    hmm.train(train)

    print(hmm.T[:, 10].sum()) # test if all transitions from one state add to 1 in Transition matrix.
    print(hmm.E[:, 10].sum()) # test if all emissions from one state add to 1 in Emission maxtrix.
    print(hmm.P.sum())       # test if all transitions from START state add to 1 in Pi maxtrix.
    # def test_(data):
    #     correct_num = 0.0
    #     token_num = 0.0
    #     for seq in data[10:20]:
    #         true = seq.labels
    #         sen = seq.features
    #         ob, T = hmm.checkOb(seq)
    #         print(sen)
    #         result = hmm.classify(seq)
    #         print(true)
    #         print(result)
    #         B = hmm.backward(ob, T)
    #         print(sum(B[:, 0] * hmm.P * hmm.E[ob[0]]))
    #         print(hmm.likelihood(seq))
    #         print()
    #         token_num += len(sen)
    #         for i in range(len(sen)):
    #             if true[i] == result[i]:
    #                 correct_num += 1
    #     print ('\nAccuracy: ' + str(correct_num / token_num))
    # test_(data)

  