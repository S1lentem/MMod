import numpy as np

from math import sqrt
from scipy.stats import chi2

def get_intervals(a, b, count, values):
    intervals = [[i, []] for i in np.arange(a, b, (b-a)/count)] + [[float(b), []]]
    
    for value in values:
        for i in range(len(intervals) - 1):
            if value >= intervals[i][0] and value < intervals[i+1][0]:
                intervals[i][-1].append(value)
                break
    
    return intervals


def pearson_criterion(interval, theory_func, item_count, q=0.99):

    result = 0
    for i, item in enumerate(interval[:-1:]):
        o = len(item[-1]) / item_count
        
        temp = item[0] + interval[i+1][0] / 2
        e = theory_func(temp)

        result += (o - e)**2 / e

    return result


def get_assessment_of_mathematical_expectation(values, count=None):
    count = count if count is not None else len(values)
    
    return sum(values) / count


def get_variance_estimate(values, M=None, count=None):
    count = count if count is not None else len(values)
    M = M if M is not None else get_assessment_of_mathematical_expectation(values, count)
    
    return sum(value**2 - (M**2) for value in values) / count


def get_correlation(values, M=None, D=None, count=None, step=3):
    count = count if count is not None else len(values)
    M = M if M is not None else get_assessment_of_mathematical_expectation(values, count)
    D = D if D is not None else get_variance_estimate(values, M, count)
    
    Mxy = sum(val * values[i+step] for i, val in enumerate(values[:-step])) / (count - step)
    return (Mxy - M**2)/D

def get_interval_assessment(values, M=None, count=None):
    count = count if count is not None else len(values)
    M = M if M is not None else get_assessment_of_mathematical_expectation(values, count)
    
    d = sum([(value - M)**2 for value in values]) / (count - 1)
    c1 = count * d / chi2.isf((1-0.99)/2, count-1) 
    c2 = count * d / chi2.isf((1+0.99)/2, count-1)

    return c1, c2