# @author: said koussi
import numpy as np
import sys
if not sys.warnoptions:
       import warnings
       warnings.simplefilter('ignore', np.RankWarning)
#import matplotlib.pyplot as plt
class PolynomialRegression:
    def __init__(self, M=3):
        self.M = M

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        assert X.ndim==y.ndim
        # degree M must be less than N number of data ponits(M<N)
        assert self.M<X.shape[0]
        X = np.vander(X, self.M + 1, increasing=True)
        self.w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))

    def predict(self, X):
        X = np.array(X)
        X = np.vander(X, self.M + 1, increasing=True)
        return np.dot(X, self.w)

X = np.linspace(0,1,50)
y = np.sin(2*np.pi*X)

model = PolynomialRegression(M=5)
model.fit(X, y)

x_test = np.array([-1, 0.5, 0.78, 1])
y_pred = model.predict(x_test)
print(y_pred)
# use matplotlib.pyplot to test the model
