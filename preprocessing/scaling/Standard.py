import numpy as np

class Standard:
     def __init__(self, X):
        self.X=X

      def fit(self):
          return self.X-self.X.mean())/self.X.std()
        
    
