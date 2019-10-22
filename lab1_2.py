import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np

from generators import ContinuousRandomNumberGenerator       
from funcs import *

from math import exp, pi, sqrt, log
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import chi2


A_PARAM = 4
A = 0
COUNT = 1_000_000
STEP = 0.1

def rayleign_distribution(x):
    return (x/A_PARAM**2)*exp(-(x**2)/(2*A_PARAM**2))

def rayleign_func(x):
    return (1-exp(-(x**2)/(2*A_PARAM**2)))


def get_end_point_for_x(func, start_point, eps=0.0001, step=0.1):
    b = start_point + step
    while (func(b) > eps):
        b += step
    return b

def get_end_point_for_y(func, a, b):
    x_for_max_y = fmin_l_bfgs_b(lambda x: -func(x), 
                                (b - a) / 2, bounds=[(a,b)],
                                approx_grad=True)[0][0]
    return func(x_for_max_y)


if __name__ == '__main__':
    b = get_end_point_for_x(rayleign_distribution, A)
    max_y = get_end_point_for_y(rayleign_distribution, A, b)

    generator = ContinuousRandomNumberGenerator(rayleign_distribution, max_y)
     
    all_x = np.arange(A, b, STEP)
    all_y = [rayleign_distribution(x) for x in all_x]

    values = list(generator.get_iterator(COUNT, b, max_y))

    M = get_assessment_of_mathematical_expectation(values, COUNT)
    D = get_variance_estimate(values, M, COUNT)
    R = get_correlation(values, M, D, COUNT)

    test_m = sqrt(pi/2)*A_PARAM
    test_d = (2 - pi/2)*A_PARAM**2

    print('M) ', M, test_m)
    print('D) ', D, test_d)
    print('R) ', R)

    plt.plot(all_x, all_y)
    plt.hist(values, weights=np.zeros_like(values) + 1. / len(values), bins=20)
    plt.show()

    l = sqrt(COUNT) if COUNT <= 100 else 1 + log(COUNT, 2)
    intervals = get_intervals(A, b, l, values)
    chi_2_value = pearson_criterion(intervals, rayleign_func, COUNT)
    chi_2_theory = chi2.ppf(0.99, l - 3)
    print(f'Pirson) {chi_2_value} < {chi_2_theory}')

    c1, c2 = get_interval_assessment(values, M, COUNT)
    print(f'{c1} < {D} < {c2}')
