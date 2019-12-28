import numpy as np

np.random.seed(42)

class EM:
    def __init__(self):
        self.a = 0
        self.b = 0
        self.c = 0
    
    def train(self, data, init_parameters=None, iterations=100):
        if init_parameters:
            self.a, self.b, self.c = init_parameters
        else:
            self.a, self.b, self.c= np.random.random(3)
            
        for i in range(1, iterations + 1):
            print('Parameters at iteration {}: \n\ta: {}\n\tb: {}\n\tc: {}'.
                format(i, self.a, self.b, self.c))
            eps = self.E_step(data)
            self.a, self.b, self.c = self.M_step(data, eps)

    
    def E_step(self, data):
        eps = np.zeros(len(data))
        for i in range(len(data)):
            P_a_head = self.a * pow(self.b, data[i]) * pow(1 - self.b, 1 - data[i])
            P_a_tail = (1 - self.a) * pow(self.c, data[i]) * pow(1 - self.c, 1 - data[i])
            eps[i] = P_a_head / (P_a_head + P_a_tail)
        return eps

    def M_step(self, data, eps):
        a = np.sum(eps) / len(data)
        b = np.sum(eps * data) / np.sum(eps)
        c = np.sum((1 - eps) * data) / np.sum(1 - eps)
        return a, b, c

    def is_converge(self):
        pass

if __name__ == '__main__':
    data = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
    em = EM()
    em.train(data, init_parameters=[0.4, 0.6, 0.7])
