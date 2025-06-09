import numpy as np
import pandas as pd
import math

import functools

def get_expected_streaks(makes, misses):
    """
    Use Wald_Wolfowitz test to compute expected number of streaks.
    """
    return  (2 * (makes * misses) / (makes + misses)) + 1


def get_variance(makes, misses, expected_streaks):
    # https://en.wikipedia.org/wiki/Wald%E2%80%93Wolfowitz_runs_test#Definition
    
    numerator = (expected_streaks - 1) * (expected_streaks - 2)
    denominator = (makes + misses - 1)

    return np.divide(numerator, denominator)

@functools.cache
def get_exact_pmf(makes, misses):
    """
    returns the exact PMF of the number of runs in makes, misses combos.
    see https://online.stat.psu.edu/stat415/lesson/21/21.1 for derivation.
           
    """
    pmf = []

    assert makes  >= 0, "makes must be >=0"
    assert misses >= 0, "misses must be >=0"

    if (makes > 0) and (misses > 0):
        min_streaks = 2
    else:
        # there's only one way this could go.
        return pd.Series([1.0], index=[1])

    smaller_one = min(makes, misses)
    # we can interleave makes, misses 2x times the smaller_one
    # with the rest of the larger one as one streak at the end.
    max_streaks = (2 * smaller_one) + 1

    for r in range(min_streaks, max_streaks+1):
        if (r % 2) == 0:
            pmf.append(_get_even(r/2, makes, misses))
        else:
            pmf.append(_get_odd((r-1)/2, makes, misses))
    return pd.Series(pmf, index=range(min_streaks, max_streaks+1))

def get_percentile_rank(makes, misses, runs):
    """
    for a certain number of runs/makes/misses, what is the percentile value in the pmf?
    calculates percentile rank, see explanation here https://en.wikipedia.org/wiki/Percentile_rank
    """

    pmf = get_exact_pmf(makes, misses)
    calced = (sum(pmf[pmf.index <= runs])  - (.5 * pmf[pmf.index == runs])) * 100
    return calced.values[0]

@functools.cache
def _comb(x,y):
    return math.comb(x,y)

def _get_even(k, n1, n2):
    k = int(k)
    numer = 2 * _comb(n1-1, k-1) * _comb(n2-1, k-1)
    denom = _comb(n1+n2, n1)
    return numer/denom


def _get_odd(k, n1, n2):
    k = int(k)
    numer = (_comb(n1-1, k) * _comb(n2-1, k-1)) + (_comb(n2-1, k) * _comb(n1-1, k-1))
    denom = _comb(n1+n2, n1)
    return numer/denom