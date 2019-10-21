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

def pearson_consent(a, b, count, values):
    intervals = [[i, []] for i in np.arange(a, b, (b-a)/INTERVAL_COUNT)] + [[float(b), []]]
    
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

        result += (o + e)**2 / e

    return result

if __name__ == '__main__':
    all_x = np.arange(0, 1, STEP)
    generator = generators.BaseRandomGeneretor()
    step = 2

    values = list(generator.get_iterator(COUNT))
    plt.plot(all_x, [FUNC(0, 1) for _ in all_x])
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

   

    intervals = pearson_consent(0, 1, INTERVAL_COUNT, values)
    
    chi_2_value = chi_2(intervals, lambda x: (x - 0)/(1-0) , COUNT) 
    print(chi_2_value)

    l = chi2.pdf(0.99, INTERVAL_COUNT-1)
    print(l)


    c1 = chi2.pdf((1-delta)/2, INTERVAL_COUNT-3)
    c2 = chi2.pdf((1+delta)/2, INTERVAL_COUNT-3)

    left = (INTERVAL_COUNT - 1)*M**2/c2
    right = (INTERVAL_COUNT - 1)*M**2/c1

    print(left, D, right)