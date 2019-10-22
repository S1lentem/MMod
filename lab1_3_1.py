import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

from generators import DiscreteRandomGenerator
from funcs import *

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
    plt.hist(values, weights=np.zeros_like(values) + 1/len(values), bins=20)
    plt.show()

    M = get_assessment_of_mathematical_expectation(values, COUNT)
    D = get_variance_estimate(values, M, COUNT)
    R = get_correlation(values, M, D, COUNT)

    test_m = GEOMETRIC_DISTRIBUTION['args']['q']/GEOMETRIC_DISTRIBUTION['args']['p']
    test_d = GEOMETRIC_DISTRIBUTION['args']['q']/GEOMETRIC_DISTRIBUTION['args']['p']**2

    print('M) ', M, test_m)
    print('D) ', D, test_d)
    print('R) ', R)
