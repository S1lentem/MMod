import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

from math import exp
from generators import BaseRandomGeneretor


BEGIN = 0
END = 32
STEP = 0.1

A_PARAM = 4

GENERATOR = BaseRandomGeneretor()


def rayleigh_distribution(x, a):
    return (x/a**2)*exp(-x**2/(2*a**2))

if __name__ == '__main__':
    random_values = [GENERATOR.get_base_random_value() for _ in range(10000)]

    plt.hist(random_values)
    plt.show()

    GENERATOR.reset()
    max_x = opt.fmin_l_bfgs_b(lambda x: -rayleigh_distribution(x, A_PARAM),
                            0,
                            bounds=[(BEGIN, END)],
                            approx_grad=True)[0][0]

    x_list = np.arange(BEGIN, END, STEP)
    y_list = [rayleigh_distribution(x, A_PARAM) for x in x_list]

    plt.plot(x_list, y_list)
    plt.show()
    
    x2 = []
    for _ in range(1000000):
        x_value = BEGIN + GENERATOR.get_base_random_value() * (END - BEGIN)
        y_value = GENERATOR.get_base_random_value() * max_x
        print(x_value, y_value, rayleigh_distribution(x_value, A_PARAM))
        input()
        
        if rayleigh_distribution(x_value, A_PARAM) < y_value:
            x2.append(x_value)
