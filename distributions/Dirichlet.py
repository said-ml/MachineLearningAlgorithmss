import numpy as np
# importing the base function gamma
from scipy.special import gamma


class dirichlet:

    def __init__(self, alpha):
        """ alpha is the hyperparameter that control the distribution """
        self._alpha = alpha
        # we make sure that _alpha is numpy.array
        assert isinstance(alpha, np.ndarray)
        # alpha_0 is sum of all alpha_{k}
        self._alpha_0 = alpha.sum()

    # the decorator poperty give us the possibility to change(setter method) ,
    # getting(getter method) the values of intances(e,g alpha) or even deleter method

    @property
    # getter method
    def alpha(self):
        return self._alpha

    # setter method
    @alpha.setter
    def alpha(self, val):
        self._alpha = val
        return self._alpha

    # deleter method
    @alpha.deleter
    def alpha_del(self):
        del self._alpha

    def _pdf(self, mu):
        # mu must be numpy.ndarray
        assert isinstance(mu, np.ndarray)
        # all component of mu must be positive
        assert (mu >= 0).all()
        # sum of components of mu must be equal one
        assert mu.sum() == 1
        # we cacul firstly the firt term of the (1) that indepent of mu
        beta = gamma(self._alpha_0) / gamma(self._alpha).prod()
        # we calcul the second term
        mu_power_alpgha_1 = mu ** (self._alpha - 1).prod()
        return beta*mu_power_alpgha_1
