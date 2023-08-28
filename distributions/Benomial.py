import numpy as np

class Binomial:
    def __init__(self, N, m, mu=0.25):
        """
        N: the number total of the tosses
        m the of the obtaining heads
        mu the parameter(probability) of obtain head """
        self.N=N
        self.m=m
        self.mu=mu

    def bernoulli(self):
         return np.random.choice([0,1])

    def factorial(self, n):
        # recursive algorithm
        if  np.array(n).any()==1:
            return 1
        else:
            return n*self.factorial(n-1)

    def C(self):
        return self.factorial(self.N)/(self.factorial(self.N-self.m)*self.factorial(self.m))

    def Binomial(self):
        return self.C(self.N, self.m) * self.mu ** self.m * (1 - self.mu) ** (self.N - self.m)
