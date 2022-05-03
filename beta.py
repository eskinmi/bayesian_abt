import math
import numpy as np
from typing import Tuple


def beta_parameters(p: float, strength: int) -> Tuple[float, float]:
    """
    Calculate beta distribution parameters (alpha and beta) given the
    conversion rate and the amount of sample required to reach to it.

    Parameters
    ----------
    p: float
        probability (e.g. conversion rate)
    strength: int
        probability strength (e.g. amount of samples)

    Returns
    -------
    Tuple[float, float]
        alpha and beta parameters from beta distribution
    """
    return (
        math.ceil(strength * p) + 1,
        strength + 1 - math.ceil(strength * p)
    )


class BetaInequality:
    """
    Utilities for exact calculation of beta
    inequalities, which is to be used in calculation of
    the area under the two posterior distributions of
    independent beta random variables.

    Examples
    --------
    a_T, b_T = beta_parameters()
    a_C, b_C = beta_parameters()
    BetaInequality.g()

    References
    ----------
    Cook, John; Exact Calculation of Beta Inequalities, 2005.
    https://www.johndcook.com/UTMDABTR-005-05.pdf
    Code: https://gist.github.com/arsatiki/1395348 (Antti Rasinen.)
    """

    @staticmethod
    def h(a_1, b_1, a_2, b_2):
        num = math.lgamma(a_1 + a_2) + math.lgamma(b_1 + b_2) + math.lgamma(a_1 + b_1) + math.lgamma(a_2 + b_2)
        den = (
                math.lgamma(a_1) +
                math.lgamma(b_1) +
                math.lgamma(a_2) +
                math.lgamma(b_2) +
                math.lgamma(a_1 + b_1 + a_2 + b_2)
        )
        return np.exp(num - den)

    @staticmethod
    def g0(a_1, b_1, a_2):
        return np.exp(math.lgamma(a_1 + b_1) + math.lgamma(a_1 + a_2) - (
                math.lgamma(a_1 + b_1 + a_2) + math.lgamma(a_1)))

    @staticmethod
    def h_iter(a_1, b_1, a_2, b_2):
        while b_2 > 1:
            b_2 -= 1
            yield BetaInequality.h(a_1, b_1, a_2, b_2) / b_2

    @staticmethod
    def g(a_1, b_1, a_2, b_2):
        return BetaInequality.g0(a_1, b_1, a_2) + sum(BetaInequality.h_iter(a_1, b_1, a_2, b_2))