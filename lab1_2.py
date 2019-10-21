import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np

from generators import ContinuousRandomNumberGenerator       
from funcs import chi_2, get_intervals

from math import exp, pi, sqrt, log
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import chi2


A_PARAM = 2


def rayleign_distribution(x):
    return (x/A_PARAM**2)*exp(-(x**2)/(2*A_PARAM**2))

def rayleign_func(x):
    return (1-exp(-(x**2)/(2*A_PARAM**2)))

    

A = 0
B = 8
STEP = 0.1
COUNT = 1_000_000
DELTA = 0.99

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

def pearson_consent(a, b, count, values):
    intervals = [[i, []] for i in np.arange(a, b, (b-a)/count)] + [[float(b), []]]
    
    for value in values:
        for i in range(len(intervals) - 1):
            if value >= intervals[i][0] and value < intervals[i+1][0]:
                intervals[i][-1].append(value)
                break
    
    return intervals


def chi_2(interval, theory_func, item_count):
    result = 0
    for i, item in enumerate(interval[:-1:]):
        o = len(item[-1]) / item_count
        
        temp = item[0] + interval[i+1][0] / 2
        e = theory_func(temp)

        result += (o - e)**2 / e

    return result

if __name__ == '__main__':
    step = 3
    
    b = get_end_point_for_x(rayleign_distribution, A)
    max_y = get_end_point_for_y(rayleign_distribution, A, b)

    generator = ContinuousRandomNumberGenerator(rayleign_distribution, max_y)
     
    all_x = np.arange(A, b, STEP)
    all_y = [rayleign_distribution(x) for x in all_x]

    values =list(generator.get_iterator(COUNT, b, max_y))

    M = sum(values) / COUNT
    D = sum(value**2 - (M**2) for value in values) / COUNT
    
    Mxy = sum(val * values[i+step] for i, val in enumerate(values[:-step])) / (COUNT - step)
    R = (Mxy - M**2)/D

    test_m = sqrt(pi/2)*A_PARAM
    test_d = (2 - pi/2)*A_PARAM**2

    print('M) ', M, test_m)
    print('D) ', D, test_d)
    print('R) ', R)

    plt.plot(all_x, all_y)
    plt.hist(values, weights=np.zeros_like(values) + 1. / len(values))
    plt.show()

    l = sqrt(COUNT) if COUNT <= 100 else 1 + log(COUNT, 2)

    intervals = pearson_consent(A, b, l, values)

    chi_2_value = chi_2(intervals, rayleign_func, COUNT)
    chi_2_theory = chi2.ppf(0.99, COUNT - 1)
    print(f'{chi_2_value} < {chi_2_theory}')


    d = sum([(value - M)**2 for value in values]) / (COUNT - 1)
    S = sqrt(d)

    k = S* chi2.ppf(0.99, COUNT-3) / sqrt(COUNT-1)

    c1 = COUNT * d / chi2.isf((1-0.99)/2, COUNT-1) 
    c2 = COUNT * d / chi2.isf((1+0.99)/2, COUNT-1)
    
    print(f'{c1} < {D} < {c2}')
