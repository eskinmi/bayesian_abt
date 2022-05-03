from typing import Union
import numpy as np
import scipy.stats

from beta import beta_parameters
from beta import BetaInequality
from utils import Evidence

# TODO:  FIND OUT CONFIDENCE INTERVALS FOR VARIANTS
# TODO: ADD BAYESIAN GAIN INTERVAL


class BayesianEvaluationFramework:
    """
    Bayesian evaluation framework for A/B tests.

    Attributes
    ----------
    prior: Union[float, int]
        prior probability, e.g. the baseline conversion rate believed to be had
        in a baseline A/B test.

    prior_strength: int
        prior strength, that is e.g. the amount of samples used to decide on the
        prior probability.

    loss_threshold: Union[float, int]
        acceptable loss threshold, that is the largest  expected loss we are willing
        to accept, if we were to make a mistake and choose the wrong variant. This is
        proportionate to prior probability (prior conversion rate).
        For instance:
            # assume that we are trying to be extremely cautious,
            # so be conservative in loss threshold
            prior_conversion = 0.30
            loss_threshold = 0.001
            # acceptable drop in conversion rate would be :
            0.30 * 0.001

    prior_a: float
        alpha parameter of the beta distribution for prior distribution

    prior_b: float
        beta parameter of the beta distribution for prior distribution

    conjugate_prior: scipy.stats.beta
        prior beta distribution

    res: Evidence
        results of the A/B test, stored in ExperimentSummary object

    random_posterior_dists: np.ndarray
        posterior distribution array with shape (n, 2)
    """

    def __init__(self,
                 prior: Union[float, int] = 0.5,
                 prior_strength: int = 50,
                 loss_threshold: Union[float, int] = 0.01
                 ):
        self.prior = prior
        self.prior_strength = prior_strength
        self.loss_threshold = loss_threshold
        self.prior_a, self.prior_b = beta_parameters(self.prior, self.prior_strength)
        self.conjugate_prior = self.generate_conjugate_prior()
        self.res = None
        self.random_posterior_dists = None

    def ingest_evidence(self, res: Evidence):
        """
        Ingest evidence (results of the A/B Test) into the
        evaluation framework.

        Parameters
        ----------
        res: Evidence
            results of the A/B Test

        Returns
        -------
        None
        """
        if isinstance(res, Evidence):
            self.res = res
            self.random_posterior_dists = self.generate_posteriors_rvs(10000)
        else:
            raise TypeError('please try injecting with ExperimentData.ingest(spark_df)')

    def generate_conjugate_prior(self):
        """
        Generate conjugate priors.

        Returns
        -------
        scipy.stats.beta
            beta distributed random variate sample
        """
        return scipy.stats.beta(self.prior_a, self.prior_b)

    def posterior_beta(self, converted, n):
        return scipy.stats.beta(self.prior_a + converted,
                                self.prior_b + n - converted,
                                )

    def generate_posteriors_rvs(self, size: int = None):
        """
        Generate posterior distribution.

        Parameters
        ----------
        size: int
            if None, the evidence sample is taken.

        Returns
        -------
        np.array
            posterior distributions of variants
        """
        posteriors = []
        for ix, variant in enumerate(self.res.variants):
            variant_size = 10000 if size is None else self.res.n[ix]
            rv = self.posterior_beta(self.res.converted[ix], self.res.n[ix])
            rvs = rv.rvs(size=variant_size)
            posteriors.append(rvs)
        return np.column_stack(posteriors)

    def expected_wins(self, variant: str, exact=False):
        """
        Calculate expected wins of the variant.

        Parameters
        ----------
        variant: str
            variant (name)
        exact: bool
            exact calculation if True, else approx. calculation

        Returns
        -------
        float
            probability of the given variant being greater or equal
            than the opponent variant
        """
        i = self.res.variants.index(variant)
        if not exact:
            expected_wins = np.greater_equal(self.random_posterior_dists[:, i], self.random_posterior_dists[:, 1 - i])
            return expected_wins.mean()
        else:
            a_v, b_v = beta_parameters(self.res.conversions[i], self.res.n[i])
            a_o, b_o = beta_parameters(self.res.conversions[1-i], self.res.n[1-i])
            return BetaInequality.g(a_v, b_v, a_o, b_o)

    def expected_loss_amount(self, variant: str, exact=False):
        """
        Expected loss in conversion rate, given that we've made a
        mistake and chose the wrong variant as the winner.

        Parameters
        ----------
        variant: str
            variant (name)
        exact: bool
            exact calculation if True, else approx. calculation

        Returns
        -------
        float
            expected loss amount
        """
        if not exact:
            i = self.res.variants.index(variant)
            proba_diff = np.subtract(self.random_posterior_dists[:, 1 - i], self.random_posterior_dists[:, i])
            expected_losses_proba = np.maximum(proba_diff, np.zeros(proba_diff.shape))
            return expected_losses_proba.mean()
        else:
            raise NotImplemented

    def evaluate(self, variant: str):
        """
        Evaluate the A/B Test results.

        Parameters
        ----------
        variant: str
            base variant name

        Returns
        -------
        None
        """
        win_proba = self.expected_wins(variant)
        expected_loss_amount = self.expected_loss_amount(variant)
        print(F'variant : {variant}')
        print(F'expected wins : {win_proba}')
        print(F'\tloss_threshold : {self.loss_threshold}')
        print(F'\texpected loss amount : {expected_loss_amount}')
        print(F'\twithin set limit : {self.loss_threshold > expected_loss_amount}')
