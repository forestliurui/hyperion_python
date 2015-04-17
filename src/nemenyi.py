"""
Implements the formula to compare models with the Nemenyi test. "The performance
of two classifiers is significantly different if the corresponding average ranks
differ by at least the critical difference" from:

  Demsar, J. "Statistical comparisons of classifiers over multiple data sets."
      The Journal of Machine Learning Research 7 (2006): 1-30.

Critical values taken from:

  http://nikolaos.kourentzes.com/files/Nemenyi_critval.pdf
"""
import math

CRITICAL_VALUES = [
# p   0.01   0.05   0.10  Models
    [2.576, 1.960, 1.645], # 2
    [2.913, 2.344, 2.052], # 3
    [3.113, 2.569, 2.291], # 4
    [3.255, 2.728, 2.460], # 5
    [3.364, 2.850, 2.589], # 6
    [3.452, 2.948, 2.693], # 7
    [3.526, 3.031, 2.780], # 8
    [3.590, 3.102, 2.855], # 9
    [3.646, 3.164, 2.920], # 10
    [3.696, 3.219, 2.978], # 11
    [3.741, 3.268, 3.030], # 12
    [3.781, 3.313, 3.077], # 13
    [3.818, 3.354, 3.120], # 14
    [3.853, 3.391, 3.159], # 15
    [3.884, 3.426, 3.196], # 16
    [3.914, 3.458, 3.230], # 17
    [3.941, 3.489, 3.261], # 18
    [3.967, 3.517, 3.291], # 19
    [3.992, 3.544, 3.319], # 20
    [4.015, 3.569, 3.346], # 21
    [4.037, 3.593, 3.371], # 22
    [4.057, 3.616, 3.394], # 23
    [4.077, 3.637, 3.417], # 24
    [4.096, 3.658, 3.439], # 25
    [4.114, 3.678, 3.459], # 26
    [4.132, 3.696, 3.479], # 27
    [4.148, 3.714, 3.498], # 28
    [4.164, 3.732, 3.516], # 29
    [4.179, 3.749, 3.533], # 30
    [4.194, 3.765, 3.550], # 31
    [4.208, 3.780, 3.567], # 32
    [4.222, 3.795, 3.582], # 33
    [4.236, 3.810, 3.597], # 34
    [4.249, 3.824, 3.612], # 35
    [4.261, 3.837, 3.626], # 36
    [4.273, 3.850, 3.640], # 37
    [4.285, 3.863, 3.653], # 38
    [4.296, 3.876, 3.666], # 39
    [4.307, 3.888, 3.679], # 40
    [4.318, 3.899, 3.691], # 41
    [4.329, 3.911, 3.703], # 42
    [4.339, 3.922, 3.714], # 43
    [4.349, 3.933, 3.726], # 44
    [4.359, 3.943, 3.737], # 45
    [4.368, 3.954, 3.747], # 46
    [4.378, 3.964, 3.758], # 47
    [4.387, 3.973, 3.768], # 48
    [4.395, 3.983, 3.778], # 49
    [4.404, 3.992, 3.788], # 50
]

def critical_value(pvalue, models):
    """
    Returns the critical value for the two-tailed Nemenyi test for a given
    p-value and number of models being compared.
    """
    if pvalue == 0.01:
        col_idx = 0
    elif pvalue == 0.05:
        col_idx = 1
    elif pvalue == 0.10:
        col_idx = 2
    else:
        raise ValueError('p-value must be one of 0.01, 0.05, or 0.10')

    if not (2 <= models and models <= 50):
        raise ValueError('number of models must be in range [2, 50]')
    else:
        row_idx = models - 2

    return CRITICAL_VALUES[row_idx][col_idx]

def critical_difference(pvalue, models, datasets):
    """
    Returns the critical difference for the two-tailed Nemenyi test for a
    given p-value, number of models being compared, and number of datasets over
    which model ranks are averaged.
    """
    cv = critical_value(pvalue, models)
    return cv*math.sqrt((models*(models + 1))/(6.0*datasets))
