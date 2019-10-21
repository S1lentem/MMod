import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

from generators import DiscreteRandomGenerator

COUNT = 100_000

GEOMETRIC_DISTRIBUTION = {
    'func': lambda p, q, i: q**i*p,
    'args': {
        'p': 0.6,
        'q': 0.4
    }
}

if __name__ == '__main__':
    all_x = range(30)
    step = 3

    generator = DiscreteRandomGenerator(range(20), 
                                        GEOMETRIC_DISTRIBUTION['func'],
                                        **GEOMETRIC_DISTRIBUTION['args'])

    values = [generator.next() for _ in range(COUNT)]

    all_y = [GEOMETRIC_DISTRIBUTION['func'](i=x, **GEOMETRIC_DISTRIBUTION['args']) 
             for x in all_x]

    plt.plot(all_x, all_y)
    plt.hist(values, weights=np.zeros_like(values) + 1/len(values))
    plt.show()

    M = sum(values) / COUNT
    D = sum(value**2 - (M**2) for value in values) / COUNT

    Mxy = sum(val * values[i+step] for i, val in enumerate(values[:-step])) / (COUNT - step)
    R = (Mxy - M**2)/D

    test_m = GEOMETRIC_DISTRIBUTION['args']['q']/GEOMETRIC_DISTRIBUTION['args']['p']
    test_d = GEOMETRIC_DISTRIBUTION['args']['q']/GEOMETRIC_DISTRIBUTION['args']['p']**2

    print('M) ', M, test_m)
    print('D) ', D, test_d)
    print('R) ', R)

    step = 2
    pair_sum = sum(val*values[i+step] for i, val in enumerate(values[:-step]))
    R = 12 / (COUNT - step) * pair_sum
    print(R)