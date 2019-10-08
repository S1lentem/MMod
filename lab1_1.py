import matplotlib.pyplot as plt
import numpy as np

import generators

COUNT_NUMBER = 10000
INTERVAL_COUNT = 10

MATHEMATICAL_EXPECTATION = 1 / 2 
DESPERSION = 1 / 12


if __name__ == '__main__':
    generator = generators.BaseRandomGeneretor()

    random_values = [generator.next() for _ in range(COUNT_NUMBER)]

    plt.hist(random_values)
    plt.show()

    generator.reset()
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

        