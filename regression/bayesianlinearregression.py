import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import random

f=lambda x:-0.3+0.5*x

class bayesianlinearregression:

      def __init__(self, alpha=2. , beta=25, EPOCHS=4):
          self.alpha=alpha
          self.beta=beta
          self.EPOCHS=EPOCHS

      def fit(self, X_train, y_train, mean_0=None, S_0=None):
          # define the basis-function Phi
          Phi=np.column_stack([ np.ones_like(X_train), X_train])
          N = X_train.shape[0]
          if mean_0 is None:
             mean_0=np.zeros(N)
          if S_0 is None:
              S_0=self.alpha*np.eye(N)
          S_N =inv(inv(S_0)+self.beta*Phi.T@Phi)
          mean_N=S_N@(inv(S_0)@mean_0+self.beta*Phi.T@y_train)

          return mean_N, S_N

      def predict(self, x_test):
          mean_N, S_N=self.fit(self.X. self.y)
          w=multivariate_normal.pdf(mean=mean_N.ravel(), cov=S_N)
          y_pred=w@x_test
          return y_pred

      def ploting_results(self ):
          mean_N, S_N=np.array([0.0, 0.0]), np.array([[1., .0], [0.0, 1.]])
          x=np.linspace(-1, 1, 50)
          y=f(x)
          w_0, w_1 = np.mgrid[-1:1:.01, -1:1:.01]
          # w=(w_0, w_1)
          w = np.dstack((w_0, w_1))
          for i in range(self.EPOCHS):
               if i==0:
                 rv = multivariate_normal(mean=mean_N.ravel(), cov=S_N)
                 fig = plt.figure()
                 ax =fig.add_subplot(111)
                 ax.contourf(w_0, w_1, rv.pdf(w))
                 plt.scatter(-0.3, 0.5, marker="x")
               #mean_N, S_N = self.fit(X_train=x, y_train=y, mean_0=mean_N.ravel(), S_0=S_N)
               if  i>0:
                 rv = multivariate_normal(mean=mean_N.ravel(), cov=S_N)
                 fig = plt.figure()
                 ax = fig.add_subplot(111)
                 ax.contourf(w_0, w_1, rv.pdf(w))
                 plt.scatter(-0.3, 0.5, marker="x")
                 plt.show()
               mean_N, S_N = self.fit(X_train=x, y_train=y, mean_0=mean_N.ravel(), S_0=S_N)
          print(mean_N, S_N)

if __name__=='__main__':
  baeslinear=bayesianlinearregression()
  #baeslinear.ploting_results()
