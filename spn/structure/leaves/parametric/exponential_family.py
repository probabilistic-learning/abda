import numpy as np
from spn.structure.Base import Leaf, Sum
from spn.structure.StatisticalTypes import Type


class ExponentialFamily(Leaf):
    """
    Represent a univariate Exponential Family distribution with P parameters \thetas
    and with: 

    * \eta as a Px1 array containing the natural parameters (as a function of \thetas)
    * T(x) as a NxP array returning the sufficient statistics for N samples x
    * h(x) returning a Nx1 array for the base measure for N samples x
    * A(\eta) representing the log-partition function as a scalar function of \eta
    * A(\theta) representing the log-partition function as a scalar function of \theta
    """

    def __init__(self, type, scope=None):
        Leaf.__init__(self, scope=scope)
        self._type = type

    @property
    def type(self):
        return self._type

    @property
    def params(self):
        raise NotImplementedError("Params is not implemented")

    @property
    def P(self):
        return len(self.params)

    @property
    def eta(self):
        raise NotImplementedError("Eta is not implemented")

    def t(self, x):
        raise NotImplementedError("Sufficient statistics are not implemented")

    def h(self, x):
        raise NotImplementedError("Base measure are not implemented")

    @property
    def A_eta(self):
        raise NotImplementedError("Log-partition function (\eta) not implemented")

    # def A_theta(self):
    #     raise NotImplementedError("Log-partition function (\theta) not implemented")

    def log_p(self, x):
        #
        # TODO: we could just stay in the log domain and push h(x) into the exp
        T_x = self.t(x)
        etas = self.eta
        assert T_x.shape[1] == etas.shape[0]
        p = self.h(x) * np.exp(np.dot(T_x, etas) - self.A_eta)
        return np.log(p)


class Gaussian(ExponentialFamily):
    """
    Implements a univariate gaussian distribution with parameters
    \thetas = [\mu(mean), \sigma ^ 2 (variance)]
    """

    def __init__(self, mean, stdev, scope=None):
        ExponentialFamily.__init__(self, Type.REAL, scope=scope)

        # parameters
        self.mean = mean
        self.stdev = stdev

    @property
    def params(self):
        return {'mean': self.mean, 'stdev': self.stdev}

    @property
    def precision(self):
        return 1.0 / self.variance

    @property
    def variance(self):
        return self.stdev * self.stdev

    @property
    def eta(self):
        """
        """
        return np.array([self.mean / self.variance, -1 / (2 * self.variance)])

    def t(self, x):
        """
        """
        assert x.shape[1] == 1
        return np.hstack((x, x * x))

    def h(self, x):
        """
        """
        return 1 / np.sqrt(2 * np.pi)

    @property
    def A_eta(self):
        """
        """
        eta_1, eta_2 = self.eta
        return - eta_1 * eta_1 / (4 * eta_2) - np.log(-2 * eta_2) / 2


if __name__ == '__main__':

    import scipy.stats

    rand_gen = np.random.RandomState(42)
    N = 100
    #
    # testing the gaussian as exponential family
    mu = 10
    sigma = 1.5
    g = Gaussian(mean=mu, stdev=sigma)
    x = rand_gen.normal(loc=11, scale=1, size=N)
    true_log_p = scipy.stats.norm(loc=mu, scale=sigma).logpdf(x)

    print('True pdf', true_log_p)

    print(g.log_p(x.reshape(N, 1)))
