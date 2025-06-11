import numpy as np
import pandas as pd
import math

import functools

def get_expected_streaks(makes, misses):
    """
    Use Wald_Wolfowitz test to compute expected number of streaks.
    
    >>> get_expected_streaks(7,4)
    6.090909090909091
    
    """
    return  (2 * (makes * misses) / (makes + misses)) + 1


def get_variance(makes, misses, expected_streaks):
    """
    Calculate the variance for the Wald-Wolfowitz test. This takes 
    expected_streaks as an argument so the function can be vectorized.

    @see https://en.wikipedia.org/wiki/Wald%E2%80%93Wolfowitz_runs_test#Definition
    
    >>> get_variance(7, 4, get_expected_streaks(7,4))
    np.float64(2.0826446280991737)
    """
    
    numerator = (expected_streaks - 1) * (expected_streaks - 2)
    denominator = (makes + misses - 1)

    return np.divide(numerator, denominator)

@functools.cache
def get_exact_pmf(makes, misses):
    """
    returns the exact PMF of the number of runs in makes, misses combos.
    see https://online.stat.psu.edu/stat415/lesson/21/21.1 for derivation.

    >>> get_exact_pmf(7,4)    
    2    0.006061
    3    0.027273
    4    0.109091
    5    0.190909
    6    0.272727
    7    0.227273
    8    0.121212
    9    0.045455
    dtype: float64
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

@functools.cache
def _comb(x,y):
    return math.comb(x, y)

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

def get_percentile_rank(makes, misses, runs):
    """
    for a certain number of runs/makes/misses, what is the percentile value in the pmf?
    calculates percentile rank, see explanation here https://en.wikipedia.org/wiki/Percentile_rank

    >>> get_percentile_rank(7,4,7)
    np.float64(71.96969696969697)
    """
    pmf = get_exact_pmf(makes, misses)
    calced = (sum(pmf[pmf.index <= runs])  - (.5 * pmf[pmf.index == runs])) * 100
    return calced.values[0]


if __name__ == "__main__":
    import doctest
    doctest.testmod()