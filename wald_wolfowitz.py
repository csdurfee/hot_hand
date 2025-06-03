import numpy as np

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