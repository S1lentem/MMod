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
    return (x/A_PARAM**2)*exp(-x**2/(2*A_PARAM**2))

A = 0
B = 8
STEP = 0.1
COUNT = 10_000

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
    generator = ContinuousRandomNumberGenerator(rayleign_distribution)
    
    b = get_end_point_for_x(rayleign_distribution, A)
    max_y = get_end_point_for_y(rayleign_distribution, A, b)
     
    all_x = np.arange(A, b, STEP)
    all_y = [rayleign_distribution(x) for x in all_x]

    plt.plot(all_x, all_y)
    plt.show()


    values =list(generator.get_iterator(COUNT, b, max_y))

    M = sum(values) / COUNT
    D = sum(value**2 - (M**2) for value in values) / COUNT
    
    test_m = sqrt(pi/2)*A_PARAM
    test_d = (2 - pi/2)*A_PARAM**2

    print(M, test_m)
    print(D, test_d)

    plt.hist(values)
    plt.show()

    l = sqrt(COUNT) if COUNT <= 100 else 1 + log(COUNT, 2)

    intervals = get_intervals(A, b, l, values)

    chi_2_value = chi_2(intervals, rayleign_distribution, COUNT) / COUNT**2
    print(chi_2_value)

    l = chi2.pdf(0.99, l-1)
    print(l)