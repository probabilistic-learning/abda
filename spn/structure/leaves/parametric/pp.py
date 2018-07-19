"""
Routines for likelihood models samples within PP frameworks

Starting to work on PyMC3 and Pyro
"""

import numpy as np
import pymc3
import theano
import pyro
import torch

from spn.structure.Base import Leaf
from spn.structure.StatisticalTypes import Type
from spn.structure.leaves.parametric.Parametric import Parametric


def orderedGaussianCutpoints(label, mu, sd, shape):
    return pymc3.Normal(label, mu=mu, sd=sd, shape=shape,
                        transform=pymc3.distributions.transforms.ordered)


PYMC3_LIKELIHOOD_MAP = {'Gamma': pymc3.Gamma,
                        'Gaussian': pymc3.Normal,
                        'Beta': pymc3.Beta,
                        'Gumbel': pymc3.Gumbel,
                        'Laplace': pymc3.Laplace,
                        'Weibull': pymc3.Weibull,
                        'Wald': pymc3.Wald}

PYRO_LIKELIHOOD_MAP = {'Gamma': pyro.distributions.Gamma,
                       'Gaussian': pyro.distributions.Normal,
                       'Beta': pyro.distributions.Beta,
                       'Gumbel': pyro.distributions.Gumbel,
                       'Laplace': pyro.distributions.Laplace,
                       # 'Weibull': pyro.distributions.Weibull,
                       # 'Wald': pymc3.Wald,
                       }


def create_prior_pymc3(prior_name, label, model, params):

    prior = None

    with model:

        prior = PYMC3_LIKELIHOOD_MAP[prior_name](label, **params)

    return prior


def get_pyro_dist(dist_name, dist_params):

    dist = PYRO_LIKELIHOOD_MAP[dist_name]
    if dist_name == 'Gaussian':
        return dist, {"loc": dist_params['mean'], "scale": dist_params['stdev']}

    elif dist_name == 'Gamma':
        return dist, {"concentration": dist_params['alpha'], "rate": dist_params['beta']}

    elif dist_name == 'Beta':
        return dist, {"concentration0": dist_params['alpha'], "concentration1": dist_params['beta']}

    elif dist_name == 'Gumbel':
        return dist, {"loc": dist_params['mu'], "scale": dist_params['beta']}

    elif dist_name == 'Laplace':
        return dist, {"loc": dist_params['mu'], "scale": dist_params['b']}

    else:
        raise Exception("unknown distribution %s " % type(dist_name))


def create_prior_pyro(prior_name, label, params):

    prior = None
    pyro_dist, pyro_params = get_pyro_dist(prior_name, params)
    prior = pyro.sample(label, pyro_dist(**pyro_params))

    return prior


def create_pp_prior(prior, prior_label, pp_engine='pymc3', context=None, params=None):
    """
    Creates a prior object abstracting from the underlying PP framework

    """
    if pp_engine == 'pymc3':
        p = create_prior_pymc3(prior_name=prior,
                               label=prior_label,
                               model=context['model'], params=params)
    elif pp_engine == 'pyro':
        p = create_prior_pyro(prior_name=prior,
                              label=prior_label,
                              params=params)
    else:
        raise ValueError('Unrecognized PP engine: {}'.format(pp_engine))

    #
    # store string representation

    return p


def create_likelihood_pymc3(lik_name, data, label, model, params):
    """
    NOTE: data shall be some theano shared var so that we can update it later
    """

    lik_model = None

    with model:

        lik_model = PYMC3_LIKELIHOOD_MAP[lik_name](label, observed=data, **params)

    return lik_model


# def create_likelihood_pyro(lik_name, data, label, params):
#     """
#     NOTE: data shall be some torch tensor var
#     """

#     pyro_dist, pyro_params = get_pyro_dist(lik_name, params)

#     def lik_model(obs_data):
#         for p, p_val in pyro_params:
#             p = create_prior_pyro(p, , prior_params)
#             x = pyro.sample(label, pyro_dist(**pyro_params), obs=obs_data)
#         return x

#     return lik_model


def create_likelihood_model(likelihood_model, likelihood_label, observed_data,
                            pp_engine='pymc3',
                            context=None, params=None):

    lm = None
    if pp_engine == 'pymc3':
        lm = create_likelihood_pymc3(lik_name=likelihood_model,
                                     data=observed_data,
                                     label=likelihood_label,
                                     model=context['model'], params=params)
    # elif pp_engine == 'pyro':
    #     lm = create_likelihood_pyro(lik_name=likelihood_model,
    #                                 data=observed_data,
    #                                 label=likelihood_label, params=params)
    else:
        raise ValueError('Unrecognized PP engine: {}'.format(pp_engine))

    return lm


def create_pp_shared_data(name, dtype, shape, pp_engine='pymc3'):

    data = None
    if pp_engine == 'pymc3':
        #
        # creating a Theno shared variable
        val = np.zeros(shape, dtype=dtype)
        data = theano.shared(name=name, value=val)
    elif pp_engine == 'pyro':
        data = torch.from_numpy(data)
    else:
        raise ValueError('Unrecognized PP engine: {}'.format(pp_engine))

    return data


def update_pp_shared_data(pp_engine='pymc3', data_var=None, new_data=None):

    if pp_engine == 'pymc3':
        #
        # data_var must be a Theano shared variable
        data_var.set_value(new_data)

    elif pp_engine == 'pyro':
        #
        #  FIXME: is this inefficient?
        data_var = torch.from_numpy(new_data)
    else:
        raise ValueError('Unrecognized PP engine: {}'.format(pp_engine))

    return data_var


def create_pp_context(pp_engine='pymc3'):
    context = {}
    if pp_engine == 'pymc3':

        model = pymc3.Model()
        context['model'] = model
    else:
        raise ValueError('Unrecognized PP engine: {}'.format(pp_engine))

    return context


def create_pp_sampler(sampler, context, priors, pp_engine='pymc3'):

    s = None
    if pp_engine == 'pymc3':

        with context['model']:
            if sampler == 'nuts':
                s = pymc3.NUTS(vars=[priors[k]
                                     for k in sorted(priors.keys())],
                               max_treedepth=10, early_max_treedepth=8,)
            elif sampler == 'hmc':
                s = pymc3.HamiltonianMC(vars=[priors[k]
                                              for k in sorted(priors.keys())],
                                        path_length=2.0,
                                        adapt_step_size=True,
                                        gamma=0.05,
                                        k=0.75,
                                        t0=10,
                                        target_accept=0.8)
            elif sampler == 'mh':
                s = pymc3.Metropolis(vars=[priors[k]
                                           for k in sorted(priors.keys())],
                                     S=None,
                                     proposal_dist=None,
                                     scaling=1.0,
                                     tune=True,
                                     tune_interval=100,
                                     model=None,
                                     mode=None,)
            else:
                raise ValueError('Unrecognized sampler: {}'.format(sampler))
    else:
        raise ValueError('Unrecognized PP engine: {}'.format(pp_engine))

    return s


class PPParametric(Parametric):
    def __init__(self, type,
                 scope=None,
                 name=None,
                 pp_engine='pymc3',
                 priors=None,
                 n_samples=100,
                 burn_in=100,
                 dtype=np.float32,
                 shape=None,
                 n_chains=1,
                 sampler='nuts'):
        Parametric.__init__(self, type=type, scope=scope)
        # self._type = type

        self.pp_engine = pp_engine

        #
        # create engine context and store priors
        self.context = create_pp_context(self.pp_engine)
        self.pp_priors = {}
        self.priors = {}
        for param, (prior_name, prior_params) in priors.items():
            self.pp_priors[param] = create_pp_prior(prior=prior_name,
                                                    prior_label=param,
                                                    pp_engine=pp_engine,
                                                    context=self.context,
                                                    params=prior_params)
            self.priors[param] = (prior_name, prior_params)

            # self.context = create_pp_context(self.pp_engine)

        #
        # create data shared variable
        data_name = 'X-{}-{}'.format(name, scope[0], self.id)
        self.data = create_pp_shared_data(name=data_name,
                                          pp_engine=self.pp_engine, dtype=dtype, shape=shape)

        #
        # create likelihood model
        self.lik_model = create_likelihood_model(name, '{}-{}'.format(name, scope[0]), self.data,
                                                 pp_engine=self.pp_engine,
                                                 context=self.context, params=self.pp_priors)

        #
        # create sampler
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.sampler_name = sampler
        self.n_chains = n_chains
        self.sampler = create_pp_sampler(sampler,
                                         context=self.context,
                                         priors=self.pp_priors,
                                         pp_engine=self.pp_engine)

    # @property
    # def type(self):
    #     return self._type

    # @property
    # def params(self):
    #     raise Exception("Not Implemented")

    def update_param_value(self, param, value):
        setattr(self, param, value)

    def update_shared_data(self, data):
        self.data = update_pp_shared_data(self.pp_engine, self.data, data)

    # def create_prior(self, param_name, prior_name, prior_params, context, pp_engine='pymc3'):
    #     """
    #     Creates a prior object abstracting from the underlying PP framework

    #     """

    #     pp_prior =
    #     #
    #     # store string representation

    #     return pp_prior

    def update_posterior(self, rand_gen, n_cores=1, verbosity=1):

        if self.pp_engine == 'pymc3':

            progressbar = False if verbosity < 2 else True
            live_plot = False if verbosity < 3 else True

            # print('before', self.params)
            with self.context['model']:
                trace = pymc3.sample(draws=self.n_samples,
                                     tune=self.burn_in,
                                     discart_tuned_samples=True,
                                     live_plot=live_plot,
                                     compute_convergence_checks=False,
                                     cores=n_cores,
                                     progressbar=progressbar,
                                     step=[self.sampler], chains=self.n_chains,
                                     random_seed=rand_gen.get_state()[2])
                # print(trace)
                for p in self.params.keys():
                    setattr(self, p, trace[p].mean())

        elif self.pp_engine == 'pyro':
            #
            # retrieve sampler
            sampler_kernel = self.sampler()
            mcmc_run = pyro.infer.mcmc.MCMC(sampler_kernel,
                                            num_samples=self.n_samples,
                                            warmup_steps=self.burn_in).run(self.data)

            for p in self.params.keys():
                p_name = self.param_name_map(p)
                posterior = pyro.infer.abstract_infer.EmpiricalMarginal(mcmc_run, p_name)
                setattr(self, p, posterior.mean)

        else:
            raise ValueError('Unrecognized PP engine {}'.format(self.pp_engine))


class Beta(PPParametric):
    """
    Implements a univariate Beta distribution with parameters
    \alpha and \beta
    By using a Probabilistic Programming formalism
    """

    def __init__(self,
                 alpha,  beta,
                 pp_engine='pymc3',
                 priors=None,
                 n_samples=100,
                 burn_in=100,
                 n_chains=1,
                 n_instances=1000,
                 sampler='nuts',
                 scope=None):

        # parameters
        self.alpha = alpha
        self.beta = beta

        #
        # FIXME: this is useless, done not to broke the Text module
        self.n_instances = n_instances

        PPParametric.__init__(self, Type.REAL,
                              scope=scope,
                              name='Beta',
                              pp_engine=pp_engine,
                              priors=priors,
                              n_samples=n_samples,
                              burn_in=burn_in,
                              dtype=np.float32,
                              shape=n_instances,
                              n_chains=n_chains,
                              sampler=sampler)

    @property
    def params(self):
        return {'alpha': self.alpha, 'beta': self.beta}

    @property
    def mode(self):
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        elif np.isclose(self.alpha, 1) and np.isclose(self.beta, 1):
            return 0.5  # any value in 0...1 would do it....
        elif np.isclose(self.alpha, 1) and self.beta > 1:
            return 0
        elif np.isclose(self.beta, 1) and self.alpha > 1:
            return 1


class Gumbel(PPParametric):
    """
    Implements a univariate Gumbel (aka  type I Fisher-Tippett) distribution with parameters
    \mu (loc) and \beta (scale)
    By using a Probabilistic Programming formalism
    """

    def __init__(self,
                 mu, beta,
                 pp_engine='pymc3',
                 priors=None,
                 n_samples=100,
                 burn_in=100,
                 n_chains=1,
                 n_instances=1000,
                 sampler='nuts',
                 scope=None):

        # parameters
        self.mu = mu
        self.beta = beta

        #
        # FIXME: this is useless, done not to broke the Text module
        self.n_instances = n_instances

        PPParametric.__init__(self, Type.REAL,
                              scope=scope,
                              name='Gumbel',
                              pp_engine=pp_engine,
                              priors=priors,
                              n_samples=n_samples,
                              burn_in=burn_in,
                              dtype=np.float32,
                              shape=n_instances,
                              n_chains=n_chains,
                              sampler=sampler)

    @property
    def params(self):
        return {'mu': self.mu, 'beta': self.beta}

    @property
    def mode(self):
        return self.mu


class Laplace(PPParametric):
    """
    Implements a univariate Laplace distribution with parameters
    \mu (loc) and b (scale)
    By using a Probabilistic Programming formalism
    """

    def __init__(self,
                 mu, b,
                 pp_engine='pymc3',
                 priors=None,
                 n_samples=100,
                 burn_in=100,
                 n_chains=1,
                 n_instances=1000,
                 sampler='nuts',
                 scope=None):

        # parameters
        self.mu = mu
        self.b = b

        #
        # FIXME: this is useless, done not to broke the Text module
        self.n_instances = n_instances

        PPParametric.__init__(self, Type.REAL,
                              scope=scope,
                              name='Laplace',
                              pp_engine=pp_engine,
                              priors=priors,
                              n_samples=n_samples,
                              burn_in=burn_in,
                              dtype=np.float32,
                              shape=n_instances,
                              n_chains=n_chains,
                              sampler=sampler)

    @property
    def params(self):
        return {'mu': self.mu, 'b': self.b}

    @property
    def mode(self):
        return self.mu


class Wald(PPParametric):
    """
    Implements a univariate Wald distribution (aka Inverse gaussian) with parameters
    \mu (loc) and \lambda (scale)
    By using a Probabilistic Programming formalism
    """

    def __init__(self,
                 mu, lam,
                 pp_engine='pymc3',
                 priors=None,
                 n_samples=100,
                 burn_in=100,
                 n_chains=1,
                 n_instances=1000,
                 sampler='nuts',
                 scope=None):

        # parameters
        self.mu = mu
        self.lam = lam

        #
        # FIXME: this is useless, done not to broke the Text module
        self.n_instances = n_instances

        PPParametric.__init__(self, Type.REAL,
                              scope=scope,
                              name='Wald',
                              pp_engine=pp_engine,
                              priors=priors,
                              n_samples=n_samples,
                              burn_in=burn_in,
                              dtype=np.float32,
                              shape=n_instances,
                              n_chains=n_chains,
                              sampler=sampler)

    @property
    def params(self):
        return {'mu': self.mu, 'lam': self.lam}

    @property
    def mode(self):
        return self.mu * (np.sqrt(1 + 9 * self.mu * self.mu / (4 * self.lam * self.lam)) - 3 * self.mu / (2 * self.lam))


class Weibull(PPParametric):
    """
    Implements a univariate Weibull distribution (aka Frechet right or Weibull minimum) with parameters
    \alpha (a) and \beta (b)
    By using a Probabilistic Programming formalism
    """

    def __init__(self,
                 alpha, beta,
                 pp_engine='pymc3',
                 priors=None,
                 n_samples=100,
                 burn_in=100,
                 n_chains=1,
                 n_instances=1000,
                 sampler='nuts',
                 scope=None):

        # parameters
        self.alpha = alpha
        self.beta = beta

        #
        # FIXME: this is useless, done not to broke the Text module
        self.n_instances = n_instances

        PPParametric.__init__(self, Type.REAL,
                              scope=scope,
                              name='Weibull',
                              pp_engine=pp_engine,
                              priors=priors,
                              n_samples=n_samples,
                              burn_in=burn_in,
                              dtype=np.float32,
                              shape=n_instances,
                              n_chains=n_chains,
                              sampler=sampler)

    @property
    def params(self):
        return {'alpha': self.alpha, 'beta': self.beta}

    @property
    def mode(self):
        if self.alpha > 1:
            return self.beta * np.power((self.alpha - 1) / self.alpha, 1.0 / self.alpha)
        elif self.alpha <= 1:
            return 0
