import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
class Mixture_Gaussian1D:

      def __init__(self, X, N  ,K, p=None, Mu=None, Sigma=None):
          ''' K is the number of classes,
          Mixture Distribution is a linear combination of K gaussian
          N is the number of samples'''
          self.K=K
          self.N=N
          self.X=X
          if p is None:
              self.p = np.abs(np.random.randn(self.K))
              # you can set np.ones(self.K) instead
              # p is the probabilities of classes so we must normalize it
              self.p = self.p / self.p.sum()
          else:
              self.p=p
              self.p=self.p/self.p.sum()
          if Mu is None:
               self.Mu = np.random.randn(self.K)
               # you can se self.Mu=np.ones(self.K) instead
          if Sigma is None:
               self.Sigma = np.abs(np.random.randn(self.K))
               # you can set self.Sigma=np.ones(self.K)
               # all Sigma's components must be positive(due Sigma is the variance(square))
          else:
              self.Sigma=Sigma

      def Gaussian(self, mu, sigma):
          D = self.X.shape[0]
          const = 1 / (np.pi ** (D / 2) * sigma)
          return const * np.exp(-0.5 * (X - mu) ** 2 /sigma**2)

      def Distribution(self):
          print(self.p, self.Mu, self.Sigma)
          return np.array([self.p[k] * self.Gaussian(self.Mu[k], self.Sigma[k]) for k in range(self.K)]).sum(axis=0)

      def ploting(self, with_components=True):
          try:
              import matplotlib.pyplot as plt
          except ImportError:
              print('you must to install matplotlib')
          if with_components:
              data = Mixture_Gaussian1D(self.X, self.N, self.K).Distribution()
              plt.plot(X, data, label='MixtureDistribution')
              plt.legend()
              if with_components:
                  for k in range(self.K):
                     data=Mixture_Gaussian1D(self.X, self.N, self.K).Gaussian(self.Mu[k], self.Sigma[k])
                     plt.plot(X, data, label='components'+str(k+1))
                     plt.legend()
              plt.show()
              plt.axis('off')



K=3
N=100
X=np.linspace(-4, 4, N)
Mixture_Gaussian1D(X, N=N  ,K=K).ploting()
