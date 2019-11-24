import matplotlib.pyplot as plt
import numpy as np

import generators

from math import sqrt, log
from scipy.stats import chi2

delta = 0.99

STEP = 0.05
COUNT = 1_000_000
INTERVAL_COUNT = int(sqrt(COUNT) if COUNT <= 100 else 1 + log(COUNT, 2))

FUNC = lambda a, b: 1 / (b - a)

MATHEMATICAL_EXPECTATION = 1 / 2 
DESPERSION = 1 / 12

if __name__ == '__main__':
    all_x = np.arange(0, 1, STEP)
    generator = generators.BaseRandomGeneretor()
    step = 2

    values = list(generator.get_iterator(COUNT))
    plt.hist(values, weights=np.zeros_like(values) + 1. / len(values))
    plt.show()

    k_array = [[] for _ in range(INTERVAL_COUNT)]
    
    for value in values:
        k_array[int(value*10)].append(value)
    
    all_n = [len(items) for items in k_array]
    all_p = [value / COUNT for value in all_n]

    M = sum(values) / COUNT
    D = sum(value**2 - (M**2) for value in values) / COUNT
    Mxy = sum(val * values[i+step] for i, val in enumerate(values[:-step])) / (COUNT - step)
    R = (Mxy - M**2)/D

    print('M) ', MATHEMATICAL_EXPECTATION, M)
    print('D) ', DESPERSION, D)
    print('R) ', R)
