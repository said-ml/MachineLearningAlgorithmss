import numpy as np
import numba as nb

class CNN:
  def __init__(self):
    pass

  # compile the code with the decorator jit(just-in-time)
  @nb.jit
  def conv(self, X, K):
        '''
        X: the input
        K: the kernel
        '''
       h, w = K.shape
       Y = np.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
       for i in range(Y.shape[0]):
           for j in range(Y.shape[1]):
               Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
       return Y

  # compile the code with the decorator jit(just-in-time)
  @nb.jit
  def tconv(X, K):
        '''
        X: the input
        K: the kernel
        '''
        h, w = K.shape
        Y = np.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Y[i:i + h, j:j + w] += X[i, j] * K
        return Y
