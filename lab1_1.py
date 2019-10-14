import matplotlib.pyplot as plt
import numpy as np

import generators

from math import sqrt, log
from scipy.stats import chi2


COUNT_NUMBER = 1_000_000
INTERVAL_COUNT = l = int(sqrt(COUNT_NUMBER) if COUNT_NUMBER <= 100 else 1 + log(COUNT_NUMBER, 2))


MATHEMATICAL_EXPECTATION = 1 / 2 
DESPERSION = 1 / 12

def pearson_consent(a, b, count, values):
    intervals = [[i, []] for i in np.arange(a, b, (b-a)/INTERVAL_COUNT)] + [[float(b), []]]
    
    for value in random_values:
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

    intervals = pearson_consent(0, 1, INTERVAL_COUNT, random_values)
    
    chi_2_value = chi_2(intervals, lambda x: COUNT_NUMBER / INTERVAL_COUNT / COUNT_NUMBER, COUNT_NUMBER) 
    print(chi_2_value)

    l = chi2.pdf(0.99, INTERVAL_COUNT-1)
    print(l)