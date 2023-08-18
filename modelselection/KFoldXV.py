# @ author said koussi
''' implementing Kfoding cross-validation based on the book Deep Learning 2016'''

import numpy as np

class KfoldXV:
     def __init__(self, X, y, K, Algorithm, Loss):

        '''
        cross-validation is a practical technique that uses to jusitify
        that algorithm A better than algorithm B in a specific task
        X:the data's features(attribus), the input
        y: the target data
        k:number of partitions(subdatates doesn't overlaping) X=Union{x_i} for i in {1, ..K} and x_i intersect x_j is empty if i not equal j
        Algorithm:algorithm learning, strategy to optimize the problem
        loss:erro function, cost function (e,g: binarycrossentropy, mean_squred_error...) it mesure the difference
        between prediction y_pred madeed by algorithm and the target y(regression), or count number(y_pred==y) (accuracy classification
        '''
        self.X=X
        self.y=y
        self.Algorithm=Algorithm
        self.K=K
        self.Loss=Loss


     def split(self):
          X_partitions=[]
          y_partitions = []
          N=self.X.shape[0]
          self.l=N//self.K
          for i in range(self.K):
              X_partitions.append(self.X[i*self.l:(i+1)*self.l])
              y_partitions.append(self.y[i * self.l:(i + 1) * self.l])
          return np.concatenate(X_partitions), np.concatenate(y_partitions)

     def loss(self, y_pred, y_real):
          return self.Loss(y_pred, y_real)

     def fit(self):
          e=0
          for i in range(self.K):
              X_partitions, y_partitions = self.split()
              X_partitions=np.delete(X_partitions, i)
              y_partitions = np.delete(y_partitions, i)
              self.Algorithm.fit( X_partitions, y_partitions)
              for j in range(self.l):
                  ej=self.loss((self.Algorithm.predict(self.X[i*self.l+j])), self.y[i*self.l+j])
                  e+=ej
          return e

###############################-----Evaluation of the algorithm(regressor)------#######################
def squared_error(X, y):
   return ((X - y)**2).mean()

class PolynomialRegression:

   def __init__(self, M: int = 3):
    'M is really depend to your task'
    self.M = M
    # intializationthe weight with M random vaues between -1 and 1
    self.w = np.random.rand(M)

   def fit(self, X, t):
    # convert X, y to numpy array
    X, t = np.array(X), np.array(t)
    assert X.ndim == t.ndim
    # degree M must be less than N number of data ponits(M<N)
    #assert self.M < X.shape[0]
    # w is weights (coefficients)that matching(fitting) to the polynomial regession
    self.w = np.polyfit(X, t, deg=self.M)  # deg is the degree of the polynom set it a M
    return self.w

   def predict(self, x):
    y = np.poly1d(self.w)(x)
    return y
       
X = np.linspace(0,1,50)
y1= np.sin(2*np.pi*X)
y2=X**9-3*X+5
K=10
M=1

for i in range(10):
     M+=1
     Kfold=KfoldXV(X=X, y=y1, K=K, Algorithm=PolynomialRegression(M=M), Loss=squared_error)
     error=Kfold.fit()
     print('error1='+str(M), error)
M=1
for i in range(10):
     M+=1
     Kfold=KfoldXV(X=X, y=y2, K=K, Algorithm=PolynomialRegression(M=M), Loss=squared_error)
     error=Kfold.fit()
     print('error2='+str(M), error, 'M=', M)
