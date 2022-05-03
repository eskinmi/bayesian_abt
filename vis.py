import numpy as np
import matplotlib.pyplot as plt

from evaluation import BayesianEvaluationFramework


def posteriors(framework: BayesianEvaluationFramework, size: int = 1000):
    """
    Visualize posterior distributions.

    Parameters
    ----------
    framework: BayesianEvaluationFramework
        bayesian evaluation framework
    size: int
        size of distribution
    """
    x = np.linspace(0, 1, size)
    fig, ax = plt.subplots(1, 1)
    evidence = framework.res
    for ix, variant in enumerate(evidence.variants):
        beta = framework.posterior_beta(evidence.converted[ix], evidence.n[ix])
        plt.plot(x, beta.pdf(x), label=variant)
    ax.set_xlabel('Probabilities')
    ax.set_ylabel('Density')
    ax.set_title('Posterior Distributions')
    ax.legend()
    plt.show()


def prior(framework: BayesianEvaluationFramework, size: int = 1000):
    """
    Visualize prior distribution.

    Parameters
    ----------
    framework: BayesianEvaluationFramework
        bayesian evaluation framework
    size: int
        size of distribution
    """
    x = np.linspace(0, 1, size)
    fig, ax = plt.subplots(1, 1)
    prior_beta = framework.conjugate_prior
    plt.plot(x, prior_beta.pdf(x), label='prior')
    ax.set_xlabel('Probabilities')
    ax.set_ylabel('Density')
    ax.set_title('Prior Distributions')
    ax.legend()
    plt.show()


