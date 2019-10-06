import matplotlib.pyplot as plt
import numpy as np

import generators

COUNT_NUMBER = 10000
INTERVAL_COUNT = 10

if __name__ == '__main__':
    generator = generators.BaseRandomGeneretor()

    random_values = [generator.get_base_random_value() for _ in range(COUNT_NUMBER)]

    plt.hist(random_values)
    plt.show()

    generator.reset()
    k_array = [[] for _ in range(INTERVAL_COUNT)]
    for _ in range(COUNT_NUMBER):
        value = generator.get_base_random_value()
        index = int()