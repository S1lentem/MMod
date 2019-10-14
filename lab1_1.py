import matplotlib.pyplot as plt
import numpy as np

import generators

from math import sqrt, log
from scipy.stats import chi2
from funcs import chi_2, get_intervals

COUNT_NUMBER = 1_000_000
INTERVAL_COUNT = l = int(sqrt(COUNT_NUMBER) if COUNT_NUMBER <= 100 else 1 + log(COUNT_NUMBER, 2))


MATHEMATICAL_EXPECTATION = 1 / 2 
DESPERSION = 1 / 12

if __name__ == '__main__':
    generator = generators.BaseRandomGeneretor()

    random_values = list(generator.get_iterator(COUNT_NUMBER))
    plt.hist(random_values)
    plt.show()

    k_array = [[] for _ in range(INTERVAL_COUNT)]
    
    for value in random_values:
        k_array[int(value*10)].append(value)
    
    all_n = [len(items) for items in k_array]
    all_p = [value / COUNT_NUMBER for value in all_n]

    M = sum(random_values) / COUNT_NUMBER
    D = sum(value**2 - (M**2) for value in random_values) / COUNT_NUMBER

    print(MATHEMATICAL_EXPECTATION, M)
    print(DESPERSION, D)

    step = 2

    pair_sum = sum(val * random_values[i+step] for i, val in enumerate(random_values[COUNT_NUMBER-step:]))
    R = 12 / (COUNT_NUMBER - step) * pair_sum
    print(R)

    intervals = get_intervals(0, 1, INTERVAL_COUNT, random_values)
    
    chi_2_value = chi_2(intervals, lambda x: COUNT_NUMBER / INTERVAL_COUNT / COUNT_NUMBER, COUNT_NUMBER) 
    print(chi_2_value)

    l = chi2.pdf(0.99, INTERVAL_COUNT-1-2)
    print(l)